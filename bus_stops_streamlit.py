# streamlit_bus_stops.py

import streamlit as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans, DBSCAN
import zipfile
import tempfile

# Streamlit Setup
st.set_page_config(page_title="School Bus Route Generator", layout="wide")
st.title("🚍 School Bus Route Generator for Athens High School")

# User Inputs
num_students = st.sidebar.number_input("Number of students to sample:", min_value=10, max_value=200, value=80, step=5)
num_buses = st.sidebar.number_input("Number of buses (KMeans clusters):", min_value=1, max_value=10, value=2, step=1)
max_students_per_bus = 40

# Upload Files
geojson_file = st.file_uploader("Upload students_near_athens.geojson file", type=["geojson"])
shapefile_zip = st.file_uploader("Upload city boundary shapefile ZIP file", type=["zip"])

if geojson_file and shapefile_zip:

    # Load Students Data
    gdf = gpd.read_file(geojson_file)
    gdf = gdf.to_crs(epsg=4326)

    # Process City Shapefile
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(shapefile_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shapefiles = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
        if shapefiles:
            shp_path = os.path.join(tmpdir, shapefiles[0])
            places = gpd.read_file(shp_path)
            troy = places[places["NAME"] == "Troy"].to_crs(epsg=4326)
        else:
            st.error("No shapefile found in uploaded ZIP.")

    # Filter Students
    gdf_filtered = gdf[gdf.within(troy.unary_union)].reset_index(drop=True)
    st.write(f"Total students inside Troy: {len(gdf_filtered)}")

    if len(gdf_filtered) >= num_students:
        gdf_students = gdf_filtered.sample(n=num_students, random_state=42).reset_index(drop=True)
    else:
        st.error(f"Not enough students inside Troy to sample {num_students}.")
        st.stop()

    # Load Road Network
    G = ox.graph_from_place("Troy, Michigan, USA", network_type="drive")

    # KMeans Clustering
    coords = np.column_stack((gdf_students.geometry.y.values, gdf_students.geometry.x.values))
    kmeans = KMeans(n_clusters=num_buses, random_state=42).fit(coords)
    gdf_students["bus_cluster"] = kmeans.labels_

    # --- Functions ---
    def compute_group_centroids(gdf, cluster_column="bus_cluster"):
        centroids = {}
        for cid in gdf[cluster_column].unique():
            group = gdf[gdf[cluster_column] == cid]
            centroids[cid] = (group.geometry.y.mean(), group.geometry.x.mean())
        return centroids

    def compute_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_cluster_indices(gdf, cluster_column="bus_cluster"):
        cluster_indices = {}
        for i, label in enumerate(gdf[cluster_column]):
            cluster_indices.setdefault(label, []).append(i)
        return cluster_indices

    def find_best_student_to_move(gdf, cluster_indices, centroids, max_size=40):
        for cid, students in cluster_indices.items():
            if len(students) <= max_size:
                continue
            for student_index in students:
                student_point = (gdf.geometry.y[student_index], gdf.geometry.x[student_index])
                possible_moves = []
                for other_cid, other_students in cluster_indices.items():
                    if other_cid == cid or len(other_students) >= max_size:
                        continue
                    dist_to_other = compute_distance(student_point, centroids[other_cid])
                    possible_moves.append((dist_to_other, student_index, cid, other_cid))
                if possible_moves:
                    possible_moves.sort()
                    _, student_index, from_cid, to_cid = possible_moves[0]
                    return student_index, from_cid, to_cid
        return None, None, None

    def reassign_out_of_place_students(gdf, max_distance_factor=0.8):
        changed = False
        centroids = compute_group_centroids(gdf, cluster_column="bus_cluster")
        for idx, row in gdf.iterrows():
            current_cluster = row["bus_cluster"]
            student_point = (row.geometry.y, row.geometry.x)
            own_centroid = centroids[current_cluster]
            dist_to_own = compute_distance(student_point, own_centroid)
            closer_cluster = current_cluster
            min_dist = dist_to_own
            for other_cid, other_centroid in centroids.items():
                if other_cid == current_cluster:
                    continue
                dist_to_other = compute_distance(student_point, other_centroid)
                if dist_to_other < min_dist * max_distance_factor:
                    closer_cluster = other_cid
                    min_dist = dist_to_other
            if closer_cluster != current_cluster:
                gdf.at[idx, "bus_cluster"] = closer_cluster
                changed = True
        return changed

    def strictly_enforce_capacity(gdf, max_size=40):
        while True:
            cluster_indices = get_cluster_indices(gdf)
            centroids = compute_group_centroids(gdf)
            overloaded_clusters = {cid: idxs for cid, idxs in cluster_indices.items() if len(idxs) > max_size}
            underloaded_clusters = {cid: idxs for cid, idxs in cluster_indices.items() if len(idxs) < max_size}
            if not overloaded_clusters:
                break
            for cid, student_indices in overloaded_clusters.items():
                while len(student_indices) > max_size:
                    own_centroid = centroids[cid]
                    distances = [(compute_distance((gdf.geometry.y[idx], gdf.geometry.x[idx]), own_centroid), idx)
                                 for idx in student_indices]
                    distances.sort(reverse=True)
                    _, student_to_move = distances[0]
                    student_point = (gdf.geometry.y[student_to_move], gdf.geometry.x[student_to_move])
                    nearest_cluster = None
                    min_dist = float('inf')
                    for other_cid, idxs in underloaded_clusters.items():
                        if len(idxs) >= max_size:
                            continue
                        dist = compute_distance(student_point, centroids[other_cid])
                        if dist < min_dist:
                            nearest_cluster = other_cid
                            min_dist = dist
                    if nearest_cluster is not None:
                        gdf.at[student_to_move, "bus_cluster"] = nearest_cluster
                        cluster_indices = get_cluster_indices(gdf)
                        student_indices = cluster_indices[cid]
                        underloaded_clusters[nearest_cluster] = cluster_indices[nearest_cluster]
                    else:
                        break

    # Rebalancing
    while True:
        cluster_indices = get_cluster_indices(gdf_students)
        centroids = compute_group_centroids(gdf_students)
        student, from_cid, to_cid = find_best_student_to_move(gdf_students, cluster_indices, centroids, max_size=max_students_per_bus)
        if student is None:
            break
        gdf_students.at[student, "bus_cluster"] = to_cid

    while True:
        changed = reassign_out_of_place_students(gdf_students)
        if not changed:
            break

    strictly_enforce_capacity(gdf_students, max_size=max_students_per_bus)

    # Plot Clusters
    st.subheader("Student Clusters (Max 40 per bus)")
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color="#333333")
    colors = plt.colormaps.get_cmap('tab10')
    for bus_id in sorted(gdf_students["bus_cluster"].unique()):
        group = gdf_students[gdf_students["bus_cluster"] == bus_id]
        ax.scatter(group.geometry.x, group.geometry.y,
                   color=colors(bus_id),
                   s=40, edgecolors="black", linewidths=0.5,
                   label=f"Bus {bus_id}")
    plt.legend()
    st.pyplot(fig)

    # Generate Shared Stops with DBSCAN
    eps_deg = 800 / 111320
    bus_stops = []
    for cluster_id in sorted(gdf_students["bus_cluster"].unique()):
        group = gdf_students[gdf_students["bus_cluster"] == cluster_id]
        coords = np.column_stack((group.geometry.y.values, group.geometry.x.values))
        db = DBSCAN(eps=eps_deg, min_samples=2, metric='euclidean').fit(coords)
        group["stop_group"] = db.labels_.astype(int)
        for label in sorted(group["stop_group"].unique()):
            if label == -1:
                for idx, row in group[group["stop_group"] == -1].iterrows():
                    bus_stops.append({"bus_id": cluster_id, "students": [idx], "geometry": row.geometry})
            else:
                sub_group = group[group["stop_group"] == label]
                centroid = Point(sub_group.geometry.x.mean(), sub_group.geometry.y.mean())
                nearest_node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
                x, y = G.nodes[nearest_node]['x'], G.nodes[nearest_node]['y']
                intersection_point = Point(x, y)
                bus_stops.append({"bus_id": cluster_id, "students": list(sub_group.index), "geometry": intersection_point})

    gdf_stops = gpd.GeoDataFrame(bus_stops, crs=gdf_students.crs)
    gdf_stops["num_students"] = gdf_stops["students"].apply(len)

    # Plot Stops
    st.subheader("Shared Stops")
    fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color="#333333")
    gdf_students.plot(ax=ax, color="white", markersize=5, label="Students", alpha=0.6)
    for bus_id in sorted(gdf_stops["bus_id"].unique()):
        group = gdf_stops[gdf_stops["bus_id"] == bus_id]
        ax.scatter(group.geometry.x, group.geometry.y,
                   color=colors(bus_id),
                   s=80, edgecolors="black", linewidths=0.5,
                   label=f"Bus {bus_id} Stops")
    plt.legend()
    st.pyplot(fig)

    # Routing (TSP)
    school_lat, school_lon = 42.5841, -83.1250
    school_point = ox.distance.nearest_nodes(G, school_lon, school_lat)
    gdf_stops["osmid"] = gdf_stops.geometry.apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))
    gdf_stops["cluster"] = gdf_stops["bus_id"].astype(int)
    bus_routes = {}
    for cluster_id in sorted(gdf_stops["cluster"].unique()):
        cluster_df = gdf_stops[gdf_stops["cluster"] == cluster_id]
        stop_nodes = list(cluster_df["osmid"])
        all_nodes = [school_point] + stop_nodes
        tsp_graph = nx.complete_graph(len(all_nodes))
        for i in tsp_graph.nodes:
            for j in tsp_graph.nodes:
                if i != j:
                    try:
                        length = nx.shortest_path_length(G, all_nodes[i], all_nodes[j], weight='length')
                        tsp_graph[i][j]['weight'] = length
                    except:
                        tsp_graph[i][j]['weight'] = float('inf')
        tsp_cycle = nx.approximation.traveling_salesman_problem(tsp_graph, cycle=True)
        ordered_osmids = [all_nodes[i] for i in tsp_cycle]
        bus_routes[cluster_id] = ordered_osmids

    # Plot Routes
    st.subheader("Final Bus Routes from Athens High School")
    fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='white', node_size=0)
    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(bus_routes))]
    for i, (cluster_id, route_nodes) in enumerate(bus_routes.items()):
        full_path = []
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            try:
                segment = nx.shortest_path(G, u, v, weight='length')
                full_path += segment[:-1]
            except:
                continue
        full_path.append(route_nodes[-1])
        ox.plot_graph_route(G, full_path, route_linewidth=2, route_color=colors[i], node_size=0, ax=ax, show=False, close=False)
    st.pyplot(fig)


