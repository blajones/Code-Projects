# This is a sample Python script.

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("2016 Crime Data.csv",
                   dtype={"FIPS": str})
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

fig = px.choropleth(df, geojson=counties, locations='FIPS', color='crime_rate_per_100000',
                           color_continuous_scale="Oranges",
                           scope="usa",
                           hover_name = 'county_name',
                           hover_data = ['MURDER', 'RAPE', 'ROBBERY', 'AGASSLT', 'BURGLRY'],
                           labels={'FIPS':'county_ID','AGASSLT':'ASSAULT','BURGLRY':'BURGLARY',
                                   'crime_rate_per_100000':'Crime Rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

c_name = df['county_name'].tolist()
crime_rate = df['crime_rate_per_100000'].tolist()
FIPS = df['FIPS'].tolist()

df2 = pd.read_csv("poverty_2016.csv", dtype={"FIPS": str})

poverty_per = df2['Poverty Percent, All Ages'].tolist()
poverty_FIPS = df2['FIPS'].tolist()
c_name.remove('Wade Hampton Census Area, AK')
c_name.remove('Shannon County, SD')
c_name.remove('Kalawao County, HI')
c_name.remove('Bedford city, VA')
FIPS.remove('51515')
FIPS.remove('02270')
FIPS.remove('46113')
FIPS.remove('15005')
crime_rate.remove(crime_rate[2987])
crime_rate.remove(crime_rate[2995])
crime_rate.remove(crime_rate[3018])
crime_rate.remove(201.7484869)

res = []
for i in range(len(FIPS)):
    for e in range(len(poverty_FIPS)):
        if FIPS[i] == poverty_FIPS[e]:
            res.append(e)

FIPSC = [i for i in FIPS if i not in poverty_FIPS ]

res2 = []
for i in range(len(res)):
    res2.append(poverty_per[res[i]])

res3 = []
for i in range(len(res2)):
    res3.append(float(res2[i]))

fig = px.scatter(x=res3, y=crime_rate, trendline="ols", hover_name = c_name)
fig.update_yaxes(title= "Crime Rate")
fig.update_xaxes(title= "Poverty Rate")
fig.show()
