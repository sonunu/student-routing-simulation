# streamlit_bus_stops.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import zipfile

st.set_page_config(page_title="Bus Stops Generator", layout="wide")
st.title("🚍 Bus Stops and Routing from Athens High School, Troy")

# Upload Files
student_file = st.file_uploader("Upload Student Points (GeoJSON):", type=["geojson"])
city_zip_file = st.file_uploader("Upload Troy City Boundaries ZIP File:", type=["zip"])

if student_file and city_zip_file:

    # Load Troy boundaries
    with zipfile.ZipFile(city_zip_file, 'r') as zip_ref:
        zip_ref.extractall("troy_boundary")
    gdf_cities = gpd.read_file("troy_boundary/tl_2019_26_place.shp")

    gdf_students = gpd.read_file(student_file).to_crs(epsg=4326)

    # Hardcoded school coordinates
    school_lat, school_lon = 42.5841, -83.1250

    # Filter to Troy
    troy_geom = gdf_cities[gdf_cities["NAME"] == "Troy"].unary_union
    gdf_students = gdf_students[gdf_students.within(troy_geom)].reset_index(drop=True)

    num_students = st.slider("Number of Students:", min_value=10, max_value=len(gdf_students), value=min(80, len(gdf_students)))
    num_buses = st.number_input("Number of Buses:", min_value=1, max_value=20, value=2)
    max_capacity = st.number_input("Max Students per Bus:", min_value=10, max_value=100, value=40)

    gdf_students = gdf_students.sample(n=num_students, random_state=42).reset_index(drop=True)

    if st.button("Generate Bus Stops and Routes"):

        # Load network and school node
        G = ox.graph_from_place("Troy, Michigan, USA", network_type="drive")
        school_point = ox.distance.nearest_nodes(G, school_lon, school_lat)

        # KMeans Clustering
        coords = np.column_stack((gdf_students.geometry.y.values, gdf_students.geometry.x.values))
        kmeans = KMeans(n_clusters=num_buses, random_state=42).fit(coords)
        gdf_students["bus_cluster"] = kmeans.labels_

        # Capacity Balancing Functions (same as original)
        def compute_group_centroids(gdf):
            return {cid: (gdf[gdf["bus_cluster"] == cid].geometry.y.mean(),
                          gdf[gdf["bus_cluster"] == cid].geometry.x.mean())
                    for cid in gdf["bus_cluster"].unique()}

        def compute_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def get_cluster_indices(gdf):
            cluster_indices = {}
            for i, label in enumerate(gdf["bus_cluster"]):
                cluster_indices.setdefault(label, []).append(i)
            return cluster_indices

        def find_best_student_to_move(gdf, cluster_indices, centroids, max_size):
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

        def strictly_enforce_capacity(gdf, max_size):
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
                        nearest_cluster, min_dist = None, float('inf')
                        for other_cid, idxs in underloaded_clusters.items():
                            if len(idxs) >= max_size:
                                continue
                            dist = compute_distance(student_point, centroids[other_cid])
                            if dist < min_dist:
                                nearest_cluster, min_dist = other_cid, dist
                        if nearest_cluster is not None:
                            gdf.at[student_to_move, "bus_cluster"] = nearest_cluster
                            cluster_indices = get_cluster_indices(gdf)
                            student_indices = cluster_indices[cid]
                            underloaded_clusters[nearest_cluster] = cluster_indices[nearest_cluster]
                        else:
                            break

        # Capacity Enforcement
        while True:
            cluster_indices = get_cluster_indices(gdf_students)
            centroids = compute_group_centroids(gdf_students)
            student, from_cid, to_cid = find_best_student_to_move(gdf_students, cluster_indices, centroids, max_capacity)
            if student is None:
                break
            gdf_students.at[student, "bus_cluster"] = to_cid

        strictly_enforce_capacity(gdf_students, max_capacity)

        # DBSCAN Stop Grouping
        eps_deg = 800 / 111320
        bus_stops = []
        for cluster_id in sorted(gdf_students["bus_cluster"].unique()):
            group = gdf_students[gdf_students["bus_cluster"] == cluster_id]
            coords = np.column_stack((group.geometry.y.values, group.geometry.x.values))
            db = DBSCAN(eps=eps_deg, min_samples=2, metric='euclidean').fit(coords)
            group["stop_group"] = db.labels_.astype(int)
            for label in sorted(group["stop_group"].unique()):
                if label == -1:
                    for _, row in group[group["stop_group"] == -1].iterrows():
                        bus_stops.append({"bus_id": cluster_id, "geometry": row.geometry})
                else:
                    sub_group = group[group["stop_group"] == label]
                    centroid = Point(sub_group.geometry.x.mean(), sub_group.geometry.y.mean())
                    nearest_node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
                    x, y = G.nodes[nearest_node]['x'], G.nodes[nearest_node]['y']
                    bus_stops.append({"bus_id": cluster_id, "geometry": Point(x, y)})

        # Final GeoDataFrame
        gdf_stops = gpd.GeoDataFrame(bus_stops, crs=gdf_students.crs)
        gdf_stops["bus_id"] = gdf_stops["bus_id"].astype(int)
        gdf_stops["osmid"] = gdf_stops.geometry.apply(lambda pt: ox.distance.nearest_nodes(G, pt.x, pt.y))

        # TSP Routing
        bus_routes = {}
        for cluster_id in sorted(gdf_stops["bus_id"].unique()):
            cluster_df = gdf_stops[gdf_stops["bus_id"] == cluster_id]
            stop_nodes = list(cluster_df["osmid"])
            all_nodes = [school_point] + stop_nodes
            tsp_graph = nx.complete_graph(len(all_nodes))
            for i in tsp_graph.nodes:
                for j in tsp_graph.nodes:
                    if i != j:
                        try:
                            tsp_graph[i][j]['weight'] = nx.shortest_path_length(G, all_nodes[i], all_nodes[j], weight='length')
                        except:
                            tsp_graph[i][j]['weight'] = float('inf')
            tsp_cycle = nx.approximation.traveling_salesman_problem(tsp_graph, cycle=True)
            ordered_osmids = [all_nodes[i] for i in tsp_cycle]
            bus_routes[cluster_id] = ordered_osmids

        # Plot
        fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='white', node_size=0)
        cmap = plt.cm.get_cmap('tab10')
        for i, (cluster_id, route_nodes) in enumerate(bus_routes.items()):
            full_path = []
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                try:
                    segment = nx.shortest_path(G, u, v, weight='length')
                    full_path.extend(segment[:-1])
                except:
                    continue
            full_path.append(route_nodes[-1])
            ox.plot_graph_route(G, full_path, route_linewidth=2, route_color=cmap(i), ax=ax, show=False, close=False)
        for idx, row in gdf_stops.iterrows():
            ax.scatter(row.geometry.x, row.geometry.y, c=[cmap(row.bus_id)], edgecolors='black', s=40)
        st.pyplot(fig)

        # Download
        st.download_button("Download Bus Stops GeoJSON", data=gdf_stops.to_json(), file_name="bus_stops.geojson", mime="application/geo+json")
