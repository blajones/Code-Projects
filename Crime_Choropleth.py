# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Python Apriori Algorithm Assignment
# Author: Blake Jones
# Date: Feb 12th, 2022

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("2016 Crime Data.csv",
                   dtype={"FIPS": str})
import plotly.express as px


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