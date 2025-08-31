import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import json
import unicodedata

def date_time(table_cells):
    #defines a function that takes a table cell(td),looking inside cell for <span> tags(wikipedia uses <span> tags for date/time info)
    #i.text.strip() extracts text and removes spaces., return[...]..a list of date/time strings
    return [i.text.strip() for i in table_cells.find_all('span')]


static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
response=requests.get(static_url)
print("Status code:", response.status_code)

soup=BeautifulSoup(response.content,"html.parser")
print(soup.prettify()[:300]) #prints the first 300 characters of the html content..for preview
print(soup.title.string) #prints the title of the page

launch_tables=soup.find_all("table",class_="wikitable") 
#finds all tables with the wikitable class which wikipedia uses for structed tables.
print("Number of tables found: ",len(launch_tables))

launch_rows=[]
for table in launch_tables:
    rows=table.find_all("tr")
    launch_rows.extend(rows)
print("Number of rows extracted: ",len(launch_rows))

launch_dates=[]
launch_sites=[]
payloads=[]
orbits=[]
customers=[]
launch_outcomes=[]
for row in launch_rows:
    cells=row.find_all("td")
    if len(cells)>0:
        date=date_time(cells[0])
        launch_dates.append(date[0] if date else None)
        launch_sites.append(cells[2].text.strip() if len(cells)>2 else None)
        payloads.append(cells[3].text.strip() if len(cells)>3 else None)
        orbits.append(cells[4].text.strip() if len(cells)>4 else None)
        customers.append(cells[5].text.strip() if len(cells)>5 else None)
        launch_outcomes.append(cells[-1].text.strip())
        #last cell is usually the outcome, so we use -1 to get it.
# Create a DataFrame with the collected data
df = pd.DataFrame({
    "Launch Date": launch_dates,
    "Launch Site": launch_sites,
    "Payload": payloads,
    "Orbit": orbits,
    "Customer": customers,
    "Launch Outcome": launch_outcomes
})

print(df.head())
df.dropna(how="all", inplace=True)
df.reset_index(drop=True, inplace=True)
df.head(10)
df.to_csv("falcon9_launches.csv", index=False)

print("Data saved to falcon9_launches.csv")


