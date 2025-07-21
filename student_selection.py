import streamlit as st
import pandas as pd
import geopandas as gpd
import random
from shapely.geometry import Point
import requests

st.set_page_config(page_title="Nationwide Student Simulation Tool", layout="wide")
st.title("Nationwide Student Simulation Tool")

# ======== Load Schools CSV (Local File) ========
@st.cache_data
def load_schools_data():
    return pd.read_csv('US_Schools_Cleaned.csv')


# ======== File Uploads for Large Files ========

st.header("🔼 Upload City Boundaries GeoJSON")
uploaded_city = st.file_uploader("Upload city boundaries file (.geojson)", type="geojson")
if uploaded_city:
    city_boundaries = gpd.read_file(uploaded_city)
    st.success("City boundaries loaded successfully.")
else:
    st.stop()  # Stop app until city boundaries are uploaded


st.header("🔼 Upload Tabblock Centroids GeoJSON")
uploaded_centroids = st.file_uploader("Upload tabblock centroids file (.geojson)", type="geojson")
if uploaded_centroids:
    tabblock_centroids = gpd.read_file(uploaded_centroids)
    st.success("Centroids loaded successfully.")
else:
    st.stop()  # Stop app until centroids are uploaded


# ======== Load Schools Dataset ========
schools_df = load_schools_data()


# ======== School Selection ========
st.header("1️⃣ Select a School")

school_name = st.text_input("Enter school name:")

if school_name:
    matches_df = schools_df[schools_df['NAME'].str.contains(school_name, case=False, na=False)]

    if not matches_df.empty:
        matches_df['Display'] = matches_df.apply(lambda row: f"{row['NAME']} ({row['CITY']}, {row['STATE']})", axis=1)
        selected_display = st.selectbox("Select your school:", matches_df['Display'])

        selected_school = matches_df[matches_df['Display'] == selected_display].iloc[0]

        school_lat = selected_school['LAT']
        school_lon = selected_school['LON']
        st.success(f"Selected: {selected_school['NAME']} in {selected_school['CITY']}, {selected_school['STATE']}")
        st.map(pd.DataFrame({'lat': [school_lat], 'lon': [school_lon]}))

    else:
        st.warning("No matching schools found.")
        st.stop()

else:
    st.stop()


# ======== City Selection ========
st.header("2️⃣ Select the City for This School")

# Filter cities containing selected city name
possible_cities = city_boundaries[city_boundaries['NAME'].str.contains(selected_school['CITY'], case=False, na=False)]

if possible_cities.empty:
    st.error("City boundary not found. This school might not have an exact city match.")
    st.stop()

city_options = possible_cities['NAME'].unique()
selected_city = st.selectbox("Select city boundary:", city_options)

city_polygon = city_boundaries[city_boundaries['NAME'] == selected_city].unary_union


# ======== Radius Selection ========
st.header("3️⃣ Set Pickup Radius")

radius_miles = st.select_slider("Select radius around school:", options=[1, 2, 3], value=3)
radius_meters = radius_miles * 1609.34

school_point = Point(school_lon, school_lat)
school_buffer = school_point.buffer(radius_meters / 111320)  # approximate conversion to degrees

# Restrict area to within both city boundary AND radius buffer
allowed_area = city_polygon.intersection(school_buffer)


# ======== Generate Students ========
st.header("4️⃣ Generate Simulated Students")

num_students = st.number_input("How many students to generate?", min_value=1, max_value=20000, value=1000, step=100)

def generate_random_point_within(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            return random_point

# Filter centroids within allowed area
filtered_centroids = tabblock_centroids[tabblock_centroids.geometry.within(allowed_area)]

if filtered_centroids.empty:
    st.error("No centroids found within city and radius. Expand your radius or check the selected city.")
    st.stop()

if st.button(f"Generate {num_students} Student Locations"):
    simulated_points = []
    for _ in range(num_students):
        sampled_centroid = filtered_centroids.sample(1).geometry.iloc[0]
        buffer = sampled_centroid.buffer(0.0005)  # Random spread around centroid (~50 meters)
        random_point = generate_random_point_within(buffer)
        simulated_points.append(random_point)

    students_gdf = gpd.GeoDataFrame(geometry=simulated_points, crs='EPSG:4326')

    st.success(f"Generated {num_students} student locations.")

    # Visualize
    st.map(pd.DataFrame({'lat': students_gdf.geometry.y, 'lon': students_gdf.geometry.x}))

    # Optional: Save simulated students as GeoJSON
    students_gdf.to_file("simulated_students.geojson", driver='GeoJSON')
    st.download_button("Download Simulated Students GeoJSON", data=students_gdf.to_json(), file_name="simulated_students.geojson")




# ======== School Selection ========
st.header("1️⃣ Select a School")

school_name = st.text_input("Enter school name:")

if school_name:
    matches_df = schools_df[schools_df['NAME'].str.contains(school_name, case=False, na=False)]

    if not matches_df.empty:
        matches_df['Display'] = matches_df.apply(lambda row: f"{row['NAME']} ({row['CITY']}, {row['STATE']})", axis=1)
        selected_display = st.selectbox("Select your school:", matches_df['Display'])

        selected_school = matches_df[matches_df['Display'] == selected_display].iloc[0]

        school_lat = selected_school['LAT']
        school_lon = selected_school['LON']
        st.success(f"Selected: {selected_school['NAME']} in {selected_school['CITY']}, {selected_school['STATE']}")
        st.map(pd.DataFrame({'lat': [school_lat], 'lon': [school_lon]}))

    else:
        st.warning("No matching schools found.")
        st.stop()

else:
    st.stop()


# ======== City Selection ========
st.header("2️⃣ Select the City for This School")

# Filter cities containing selected city name
possible_cities = city_boundaries[city_boundaries['NAME'].str.contains(selected_school['CITY'], case=False, na=False)]

if possible_cities.empty:
    st.error("City boundary not found. This school might not have an exact city match.")
    st.stop()

city_options = possible_cities['NAME'].unique()
selected_city = st.selectbox("Select city boundary:", city_options)

city_polygon = city_boundaries[city_boundaries['NAME'] == selected_city].unary_union


# ======== Radius Selection ========
st.header("3️⃣ Set Pickup Radius")

radius_miles = st.select_slider("Select radius around school:", options=[1, 2, 3], value=3)
radius_meters = radius_miles * 1609.34

school_point = Point(school_lon, school_lat)
school_buffer = school_point.buffer(radius_meters / 111320)  # approximate conversion to degrees

# Restrict area to within both city boundary AND radius buffer
allowed_area = city_polygon.intersection(school_buffer)


# ======== Generate Students ========
st.header("4️⃣ Generate Simulated Students")

num_students = st.number_input("How many students to generate?", min_value=1, max_value=20000, value=1000, step=100)

def generate_random_point_within(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            return random_point

# Filter centroids within allowed area
filtered_centroids = tabblock_centroids[tabblock_centroids.geometry.within(allowed_area)]

if filtered_centroids.empty:
    st.error("No centroids found within city and radius. Expand your radius or check the selected city.")
    st.stop()

if st.button(f"Generate {num_students} Student Locations"):
    simulated_points = []
    for _ in range(num_students):
        sampled_centroid = filtered_centroids.sample(1).geometry.iloc[0]
        buffer = sampled_centroid.buffer(0.0005)  # Random spread around centroid (~50 meters)
        random_point = generate_random_point_within(buffer)
        simulated_points.append(random_point)

    students_gdf = gpd.GeoDataFrame(geometry=simulated_points, crs='EPSG:4326')

    st.success(f"Generated {num_students} student locations.")

    # Visualize
    st.map(pd.DataFrame({'lat': students_gdf.geometry.y, 'lon': students_gdf.geometry.x}))

    # Optional: Save simulated students as GeoJSON
    students_gdf.to_file("simulated_students.geojson", driver='GeoJSON')
    st.download_button("Download Simulated Students GeoJSON", data=students_gdf.to_json(), file_name="simulated_students.geojson")

