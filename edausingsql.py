import csv,sqlite3
import prettytable
import pandas as pd
prettytable.DEFAULT='DEFAULT'
con=sqlite3.connect("mydata1.db")
curr=con.cursor()
df=pd.read_csv('Spacex.csv')
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")
curr.execute("DROP TABLE IF EXISTS SPACEXTABLE")
curr.execute("create table spacextable as select * from SPACEXTBL where Date is not null")
#Printing the first five rows of the table
print("First Five rows: ")
for row in curr.execute("SELECT * FROM SPACEXTABLE LIMIT 5"):
    print(row)

#taking the distinct launch sites from the table...
print("Distinct Launch Sites: ")
for row in curr.execute("SELECT DISTINCT(Launch_Site) FROM SPACEXTABLE"):
    print(row)


print("5 records where launch site that begins with CCA: ")
for row in curr.execute("SELECT * FROM SPACEXTABLE WHERE Launch_Site LIKE 'CCA%' LIMIT 5"):
    print(row)

print("Total payload mass carried by boosters launched by NASA (CRS): ")
for row in curr.execute("SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Customer='NASA (CRS)'"):
    print(row)

print("Average payload mass carried by bosster version F9 V1.1: ")
for row in curr.execute("SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Booster_Version='F9 v1.1'"):
    print("Average payload mass of F9 v1.1: ",row[0])

print("Date when first successful landing outcome in ground pad was acheived: ")
for row in curr.execute("SELECT MIN(Date) FROM SPACEXTABLE WHERE Landing_Outcome='Success (ground pad)'"):
    print(row)

print("Name of the booster of success in drone ship and has payload mass between 4000 and 6000: ")
for row in curr.execute("SELECT Booster_Version FROM SPACEXTABLE WHERE Landing_Outcome='Success (drone ship)' AND PAYLOAD_MASS__KG_>=4000 AND PAYLOAD_MASS__KG_<=6000"):
    print(row)

print("Total number of mission outcomes with faulure (in flight): ")
for row in curr.execute("SELECT COUNT(Mission_Outcome) FROM SPACEXTABLE WHERE Mission_Outcome='Failure (in flight)'"):
    print("Total number of successful missions: ",row[0])

print("Total number of mission outcomes with success: ")
for row in curr.execute("SELECT COUNT(Mission_Outcome) FROM SPACEXTABLE WHERE Mission_Outcome='Success'"):
    print("Total number of failed missions: ",row[0])

print("Name of booster of max payload mass using subquery with suitable aggregate function: ")
for row in curr.execute("SELECT Booster_Version FROM SPACEXTABLE WHERE PAYLOAD_MASS__KG_=(SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTABLE)"):
    print(row)

print("Drone Ship Failure Landings in 2015 (with month)")
query = '''
SELECT substr(Date, 6, 2) AS Month,
       "Landing_Outcome",
       "Booster_Version",
       "Launch_Site"
FROM SPACEXTABLE
WHERE substr(Date, 0, 5) = '2015'
  AND "Landing_Outcome" LIKE 'Failure (drone ship)%'
'''
for row in curr.execute(query):
    print(row)


print("Count of landing outcomes between dates 2010-06-04 and 2017-03-20: ")
query = '''
SELECT "Landing_Outcome",
         COUNT("Landing_Outcome")
FROM SPACEXTABLE
WHERE Date BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY "Landing_Outcome"
'''
for row in curr.execute(query):
    print(row)

con.commit()
con.close()

