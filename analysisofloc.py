import folium
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
import pandas as pd
from math import sin, cos, sqrt, atan2, radians

spacex_df = pd.read_csv('spacex_launch_geo.csv')
print("Dataset loaded successfully!")
print(spacex_df.head())


spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]

spacex_df['Lat'] = spacex_df['Lat'].astype(float)
spacex_df['Long'] = spacex_df['Long'].astype(float)

def assign_marker_color(launch_outcome):
    return 'green' if launch_outcome == 1 else 'red'

spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)

launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site','Lat','Long']]

mean_lat = spacex_df['Lat'].mean()
mean_long = spacex_df['Long'].mean()
site_map = folium.Map(location=[mean_lat, mean_long], zoom_start=2)

nasa_coordinate = [29.559684888503615, -95.0830971930759]
folium.Circle(
    nasa_coordinate,
    radius=50000,  # 50 km
    color='#d35400',
    fill=True,
    fill_opacity=0.3,
    popup='NASA Johnson Space Center'
).add_to(site_map)

folium.map.Marker(
    nasa_coordinate,
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>NASA JSC</b></div>'
    )
).add_to(site_map)


for lat, lng, label in zip(launch_sites_df['Lat'], launch_sites_df['Long'], launch_sites_df['Launch Site']):
    folium.Circle(
        location=[lat, lng],
        radius=50000,
        color='#1f77b4',
        fill=True,
        fill_opacity=0.3,
        popup=f"Launch Site: {label}"
    ).add_to(site_map)


marker_cluster = MarkerCluster().add_to(site_map)
for index, row in spacex_df.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=f"Launch Site: {row['Launch Site']}\nOutcome: {'Success' if row['class']==1 else 'Failure'}\nLat: {row['Lat']}, Long: {row['Long']}",
        icon=folium.Icon(color=row['marker_color'])
    ).add_to(marker_cluster)

formatter = "function(num) {return L.Util.formatNum(num, 5);};" 
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=True,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)
mouse_position.add_to(site_map)


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


latitudewater = 28.56378
longitudewater = -80.56805
latitudesite = 28.56325
longitudesite = -80.57681

distance_coastline = calculate_distance(latitudesite, longitudesite, latitudewater, longitudewater)
print(f"Distance to coastline: {distance_coastline:.2f} km")

distance_marker = folium.Marker(
    [latitudewater, longitudewater],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
    )
)

lines = folium.PolyLine(
    locations=[[latitudesite, longitudesite], [latitudewater, longitudewater]],
    weight=2,
    color='black'
)


latitudehighway=28.56265
longitudehighway=-80.57069
latitudesite2=28.56325
longitudesite2=-80.57681
distance=calculate_distance(latitudesite2,longitudesite2,latitudehighway,longitudehighway)
print(f"Distance from launch site to the highway: {distance:.2f} km")

distance_marker2=folium.Marker(
    [latitudehighway,longitudehighway],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance),
    )
)
lines2=folium.PolyLine(
    locations=[[latitudesite2,longitudesite2],[latitudehighway,longitudehighway]],
    weight=2,
    color='blue'
)

latituderoad=28.55793
longituderoad=-80.58032
latitudesite3=28.56325
longitudesite3=-80.57681
distance_road=calculate_distance(latitudesite3,longitudesite3,latituderoad,longituderoad)
print(f"Distance from launch site to the road: {distance_road:.2f} km")
distance_marker3=folium.Marker(
    [latituderoad,longituderoad],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_road),
    )
)
lines3=folium.PolyLine(
    locations=[[latitudesite3,longitudesite3],[latituderoad,longituderoad]],
    weight=2,
    color='green'
)

latitudeharrison=28.55263
longitudeharrison=-80.58892
latitudesite4=28.56325
longitudesite4=-80.57681
distance_harrison=calculate_distance(latitudesite4,longitudesite4,latitudeharrison,longitudeharrison)
print(f"Distance from launch site to the city Harrison: {distance_harrison:.2f} km")
distance_marker4=folium.Marker(
    [latitudeharrison,longitudeharrison],
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_harrison),
    )
)
lines4=folium.PolyLine(
    locations=[[latitudesite4,longitudesite4],[latitudeharrison,longitudeharrison]],
    weight=2,
    color='orange'
)

site_map.add_child(lines)
site_map.add_child(distance_marker)
site_map.add_child(lines2)
site_map.add_child(distance_marker2)
site_map.add_child(lines3)
site_map.add_child(distance_marker3)
site_map.add_child(lines4)
site_map.add_child(distance_marker4)
site_map.save("complete_launch_map.html")
print("Map saved as complete_launch_map.html")

