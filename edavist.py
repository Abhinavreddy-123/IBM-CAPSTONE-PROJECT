import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('dataset_part_2.csv')
print(df.head())

sns.catplot(y="PayloadMass",x="FlightNumber",hue="Class",data=df,aspect=5)
#the hue paramater is used to add another dimension to the plot, it marks success/failure of mission
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)

plt.xticks(rotation=90)
plt.show()

sns.catplot(y="LaunchSite",x="FlightNumber",hue="Class",data=df,aspect=5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.xticks(rotation=90)
plt.show()

sns.catplot(x="PayloadMass",y="LaunchSite",hue="Class",data=df,aspect=3)
plt.xlabel("Payload Mass (kg)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.xticks(rotation=90)
plt.show()

orbit_sucess= df.groupby("Orbit")["Class"].mean().reset_index()
orbit_sucess=orbit_sucess.rename(columns={"Class":"Success Rate"})
print(orbit_sucess)
plt.figure(figsize=(10,6))
sns.barplot(x="Orbit", y="Success Rate", data=orbit_sucess, palette="viridis")
plt.title("Success Rate per Orbit", fontsize=16)
plt.ylabel("Success Rate", fontsize=14)
plt.xlabel("Orbit Type", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.show()


sns.catplot(x="FlightNumber",y="Orbit",hue="Class",data=df,aspect=5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.xticks(rotation=90)
plt.show()

sns.catplot(x="PayloadMass",y="Orbit",hue="Class",data=df,aspect=3)
plt.xlabel("Payload Mass (kg)",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.xticks(rotation=90)
plt.show()


def Extract_year(df):
    return df["Date"].apply(lambda x: x.split("-")[0]) 
df["Year"] = Extract_year(df)
yearly_success = df.groupby("Year")["Class"].mean().reset_index()
print(yearly_success)
plt.figure(figsize=(10,6))
sns.lineplot(x="Year", y="Class", data=yearly_success, marker="o")
plt.title("Yearly Launch Success Trend", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Success Rate", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()

features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
print(features.head())

features_one_hot = pd.get_dummies(
    features, 
    columns=["Orbit", "LaunchSite", "LandingPad", "Serial"]
)
print(features_one_hot.shape[1])
features_one_hot=features_one_hot.astype("float64")
print(features_one_hot.dtypes)
print(features_one_hot.head())

features_one_hot.to_csv('dataset_part_3.csv', index=False)

