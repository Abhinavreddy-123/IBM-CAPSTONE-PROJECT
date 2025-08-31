import pandas as pd
import numpy as np
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df=pd.read_csv('data_falcon9.csv')
print(df.head())
print(df.isnull().sum()/len(df)*100)
print(df.dtypes)
print(df['LaunchSite'].value_counts())
print(df['Orbit'].value_counts()) #do not count GTO , as it is a transfer orbit and not itself geostationary.

landing_outcomes=df['Outcome'].value_counts()
print(landing_outcomes)
print(landing_outcomes.keys())
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
print(bad_outcomes)

numlist=list()
for outcome in df['Outcome']:
    if outcome in bad_outcomes:
        numlist.append(0)
    else:
        numlist.append(1)
print(numlist)
landing_class=numlist
print(landing_class)

df['Class']=landing_class
print(df[['Class']].head(8))

print(df.head())
print(df["Class"].mean())

df.to_csv('dataset_part_1.csv', index=False)
print("Wrangled CSV created successfully")


