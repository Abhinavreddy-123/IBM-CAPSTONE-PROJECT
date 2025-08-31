import requests
import pandas as pd
import numpy as np
import datetime
import json
pd.set_option('display.max_columns', None)  # Show all columns in DataFrame
pd.set_option('display.max_rows', None)     # Show all rows in DataFrame
pd.set_option('display.width', 1000)       # Set display width for better readability
pd.set_option('display.max_colwidth', None)  # Show full content of each cell

spacex_url = "https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
data = response.json()

data_df = pd.json_normalize(data)
print("Launches shape:", data_df.shape)

data_df = data_df[['rocket','payloads','launchpad','cores','flight_number','date_utc']]


data_df = data_df[data_df['cores'].map(len) == 1]
data_df = data_df[data_df['payloads'].map(len) == 1]


data_df['cores'] = data_df['cores'].map(lambda x : x[0])
data_df['payloads'] = data_df['payloads'].map(lambda x : x[0])


data_df['date'] = pd.to_datetime(data_df['date_utc']).dt.date
data_df = data_df[data_df['date'] <= datetime.date(2020, 11, 13)]

rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()
rockets_dict = {r['id']: r for r in rockets}

launchpads = requests.get("https://api.spacexdata.com/v4/launchpads").json()
launchpads_dict = {l['id']: l for l in launchpads}

payloads = requests.get("https://api.spacexdata.com/v4/payloads").json()
payloads_dict = {p['id']: p for p in payloads}

cores = requests.get("https://api.spacexdata.com/v4/cores").json()
cores_dict = {c['id']: c for c in cores}

BoosterVersion = data_df['rocket'].map(lambda x: rockets_dict[x]['name']).tolist()
Longitude = data_df['launchpad'].map(lambda x: launchpads_dict[x]['longitude']).tolist()
Latitude  = data_df['launchpad'].map(lambda x: launchpads_dict[x]['latitude']).tolist()
LaunchSite = data_df['launchpad'].map(lambda x: launchpads_dict[x]['name']).tolist()
PayloadMass = data_df['payloads'].map(lambda x: payloads_dict[x].get('mass_kg')).tolist()
Orbit = data_df['payloads'].map(lambda x: payloads_dict[x].get('orbit')).tolist()

Block = []
ReusedCount = []
Serial = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []

for core in data_df['cores']:
    if core['core'] and core['core'] in cores_dict:
        Block.append(cores_dict[core['core']]['block'])
        ReusedCount.append(cores_dict[core['core']]['reuse_count'])
        Serial.append(cores_dict[core['core']]['serial'])
    else:
        Block.append(None)
        ReusedCount.append(None)
        Serial.append(None)
#Block,Reusedcount and serial are same for a core, no matter which flight
#per launch attributes like outcome,flights,gridfins,resued,legs and landingpad change from one flight to another.
    Outcome.append(str(core['landing_success']) + ' ' + str(core['landing_type']))
    Flights.append(core['flight'])
    GridFins.append(core['gridfins'])
    Reused.append(core['reused'])
    Legs.append(core['legs'])
    LandingPad.append(core['landpad'])

launch_dict = {
    'FlightNumber': list(data_df['flight_number']),
    'Date': list(data_df['date']),
    'BoosterVersion': BoosterVersion,
    'PayloadMass': PayloadMass,
    'Orbit': Orbit,
    'LaunchSite': LaunchSite,
    'Outcome': Outcome,
    'Flights': Flights,
    'GridFins': GridFins,
    'Reused': Reused,
    'Legs': Legs,
    'LandingPad': LandingPad,
    'Block': Block,
    'ReusedCount': ReusedCount,
    'Serial': Serial,
    'Longitude': Longitude,
    'Latitude': Latitude
}

launch_df = pd.DataFrame(launch_dict)
print("Final dataset shape:", launch_df.shape)
print(launch_df.head())

data_falcon9 = launch_df[launch_df['BoosterVersion']=="Falcon 9"].copy()
data_falcon9['FlightNumber'] = range(1, len(data_falcon9)+1)

mean_mass = data_falcon9['PayloadMass'].mean()
data_falcon9['PayloadMass'].fillna(mean_mass, inplace=True)

print("Falcon 9 shape:", data_falcon9.shape)
print(data_falcon9.head())
print(data_falcon9.isnull().sum())

data_falcon9.to_csv('data_falcon9.csv', index=False)
print("✅ Data saved to data_falcon9.csv")

