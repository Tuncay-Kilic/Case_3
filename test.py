#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import requests
import json
import datetime
import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib import pyplot
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import branca
import branca.colormap as cm
import streamlit_folium
from streamlit_folium import folium_static


# In[23]:


df_laadpaal = pd.read_csv("laadpaaldata.csv")


# In[24]:


df_laadpaal.head()


# In[25]:


#Naar minuten converteren, makkelijker te lezen
df_laadpaal['ConnectedTime_InMin'] = df_laadpaal['ConnectedTime'] * 60 
df_laadpaal['ChargerTime_InMin'] = df_laadpaal['ChargeTime'] * 60


# In[26]:


#Onbenutte tijd.
df_laadpaal['noncharging'] = df_laadpaal['ConnectedTime_InMin'] - df_laadpaal['ChargerTime_InMin']


# In[27]:


#df_laadpaal["ChargerTime_InMin"].max()

#Laadpalen laadtijd gefilterd op reele laadtijden. 
df_laadpaal = df_laadpaal[df_laadpaal.ChargerTime_InMin > 0]
df_laadpaal = df_laadpaal[df_laadpaal.ChargerTime_InMin < 397]

print("Mean ", df_laadpaal["ChargerTime_InMin"].mean())
print("Median ", df_laadpaal["ChargerTime_InMin"].median())


#df_laadpaal = df_laadpaal.drop(df_laadpaal[df_laadpaal['ChargerTime_InMin'] < 0], inplace = True, axis = 1)


# In[28]:


#Boxplot om outliers te bepalen.
fig = px.box(df_laadpaal, y="ChargerTime_InMin")
fig.show()


# In[29]:


#Visualisatie van charging vs non charging
labels = ['Non-charge time', 'Charge time']
plt.hist([df_laadpaal['noncharging'], df_laadpaal['ChargerTime_InMin']],stacked = True, bins = 10, range = [1, 485])
plt.title('Charge time vs non-charge time')
plt.legend(labels)
plt.show()


# In[30]:


fig = px.histogram(df_laadpaal, x="ChargerTime_InMin", labels={"ChargerTime_InMin":"Oplaadtijd in minuten"})

fig.add_annotation(x=138, y=400,
            text="Mean : 140.09442977168854",
            showarrow=True,
            arrowhead=1)
fig.add_annotation(x=133, y=320,
            text="Median  133.998",
            showarrow=True)


fig.show()


# In[31]:


fig = px.histogram(df_laadpaal, x="ChargerTime_InMin", marginal="rug",
                   hover_data=df_laadpaal.columns, labels={"ChargerTime_InMin":"Oplaadtijd in minuten"})
fig.show()


# In[11]:


print(df_laadpaal.isna().sum())
#Geen duplicates, dus we hoeven bij deze dataset niks te vervangen/verwijderen.


# In[32]:


#Profiel laadgebruik over tijd, over een sample.
plt.plot(df_laadpaal['Ended'][0:50], df_laadpaal['MaxPower'][0:50])
plt.title('Laadprofiel 2018')
plt.xticks(np.arange(0,0, step=1), rotation = 90)
plt.xlabel('tijd')
plt.ylabel('Max power (W)')


# In[ ]:


#blablabla


# In[13]:


key='f3985c54-660c-43a6-9359-1f0c0bc73f0c'
url = "https://api.openchargemap.io/v3/poi/?key=key"
querystring = {"camelcase":"true","key":"",'countrycode':'NL', 'compact':True, 'maxresults':7990}
headers = {"Content-Type": "application/json"}

response = requests.request("GET", url,headers=headers, params=querystring)
data = response.text

opencharge_data = json.loads(data)

opencharge_data


# In[33]:


df_laadpalen = pd.DataFrame(opencharge_data)
df_laadpalen.columns


# In[15]:


df_adress = pd.DataFrame(df_laadpalen['addressInfo'])
data_folium = pd.concat([df_adress.addressInfo.apply(pd.Series), df_adress.drop('addressInfo', axis=1)], axis=1)
data_folium.info()


# In[34]:


data_folium.head()


# In[69]:


# !! Bij error RUN 1 code onder en run dan deze code opnieuw !!
#Maken van een nieuwe folium map genaamd i
i = folium.Map(location=[52.3545828, 4.7638778], zoom_start=4, top="2%", tiles="Stamen Toner")

#For loop gebruiken waarbij elke row wordt verbonden aan een marker + categorie kleur.
for index, row in data_folium.iterrows() :
    folium.CircleMarker(location=[row['latitude'], row['longitude']], radius = 0.8, popup=row['countryID'], fill=True, fill_opacity=0.2, opacity=0).add_to(i)
    

#Legenda

lgd_txt = '<span style="color: {col};">{txt}</span>'


for idx, color in enumerate(['Blauwe vlek']):  # color choice is limited
    fg = folium.FeatureGroup(name= lgd_txt.format( txt= color+' = Aanwezigheid van een laadpaal', col= 'blue'))
    i.add_child(fg)

folium.map.LayerControl('topleft', collapsed= False).add_to(i)
i


# In[68]:


i = folium.Map(location=[52.0898989, 5.1000000], zoom_start = 7, min_zoom = 7)

clustermod = MarkerCluster().add_to(i)

for row in data_folium.iterrows():
    row_values = row[1]
    location = [row_values['latitude'], row_values['longitude']]
    popup = popup = '<strong>' + row_values['title'] + '</strong>'
    marker = folium.Marker(location = location, popup = popup)
    marker.add_to(clustermod)
    
#Legenda
colormap = cm.LinearColormap(colors=['green', 'yellow','darkorange'], index=[0, 10, 100],vmin=0,vmax=100)
colormap.caption = 'Hoeveelheid laadpalen per dichtheid Nederland'
colormap
colormap.add_to(i)

folium_static(i)
#display(i)
#i


# In[ ]:


#blablabla


# # Lineaire regressie

# In[50]:


wegenbelastingdf = pd.read_csv("opbrengst-motorrijtuigenbelasting.csv")
elektrobelasting = pd.read_excel("elekautotovbelasting.xlsx")
mergedf = pd.read_excel("mergedf.xlsx")


# In[51]:


elektrobelasting.head()


# In[52]:


mergedf.head()


# In[53]:


wegenbelastingdf.head(15)


# In[59]:


sns.regplot(x=mergedf["Omzet (rijk)"], y=mergedf["Elektrische Autos"]).set(title="Lineaire regressie, Elektrische auto's t.o.v. Omzet motorrijtuigen belasting")


# # Dataset Electrische auto's

# In[44]:


df_auto = pd.read_csv('lijngrafiek_autos.csv')
df_auto = df_auto.assign(Datum_tenaamstelling = pd.to_datetime(df_auto['Datum_eerste_afgifte_Nederland']))


# In[45]:


df_auto.head()


# In[46]:


df_auto['Years'] = df_auto['Datum_tenaamstelling'].dt.year
df_auto['Month'] = df_auto['Datum_tenaamstelling'].dt.month 

df_auto.head()


# In[47]:


#Filter om het jaar op te stellen
df_auto2020 = df_auto[df_auto['Years']==2020]
df_auto2020.head()


# In[49]:


#dataset inladen
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Diesel),
               name="Diesel",
               line=dict(color="#33CFA5")))

fig.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Benzine),
               name="Benzine",
               visible=False,
               line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Hybride),
               name="Hybride",
               line=dict(color="#F06A6A")))
fig.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Elektriciteit),
               name="Elektriciteit",
               line=dict(color="#F06A6A")))


fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="ALL",
                     method="update",
                     args=[{"visible": [True, True, True, True]},
                           {"title": "ALL"}]),
                dict(label="Diesel",
                     method="update",
                     args=[{"visible": [True, False, False, False]},
                           {"title": "Diesel"}]),
                dict(label="Benzine",
                     method="update",
                     args=[{"visible": [False, True, False, False]},
                           {"title": "Benzine"}]),
                dict(label="Hybride",
                     method="update",
                     args=[{"visible": [False, False, True, False]},
                           {"title": "Hybride"}]),
                dict(label="Elektriciteit",
                     method="update",
                     args=[{"visible": [False, False, False,True]},
                           {"title": "Elektriciteit"}]),
                    ]),
        )
    ])
# Add annotation
fig.update_layout(
    annotations=[
        dict(text="Soort Brandstof:", showarrow=False,
        x=0, y=1.085, yref="paper", align="left")
    ]
)

fig.show()


# In[ ]:




