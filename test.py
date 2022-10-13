#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import streamlit as st
import streamlit_folium
from streamlit_folium import folium_static


# In[2]:


color_map = ["cyan", "darkcyan", "cornflowerblue"] 
st.title('2022-2023 sem-1 Case 3: laadpalen')
st.header('case 3')
st.subheader(' Team 27?: Tuncay, Umut, Serhat, Jullian') 


# In[3]:


st.subheader('Laadpaaldata opdracht') 


# In[4]:


code0= '''
df_laadpaal = pd.read_csv("laadpaaldata.csv")
df_laadpaal
#laat laadpaal data zien
'''
st.code(code0, language = 'python')


# In[62]:


df_laadpaal = pd.read_csv("laadpaaldata.csv")
df_laadpaal
st.dataframe(df_laadpaal)


# In[6]:


#Naar minuten converteren, makkelijker te lezen
df_laadpaal['ConnectedTime_InMin'] = df_laadpaal['ConnectedTime'] * 60 
df_laadpaal['ChargerTime_InMin'] = df_laadpaal['ChargeTime'] * 60

#Onbenutte tijd.
df_laadpaal['noncharging'] = df_laadpaal['ConnectedTime_InMin'] - df_laadpaal['ChargerTime_InMin']
#df_laadpaal["ChargerTime_InMin"].max()

#Laadpalen laadtijd gefilterd op reele laadtijden. 
df_laadpaal = df_laadpaal[df_laadpaal.ChargerTime_InMin > 0]
df_laadpaal = df_laadpaal[df_laadpaal.ChargerTime_InMin < 397]

print("Mean ", df_laadpaal["ChargerTime_InMin"].mean())
print("Median ", df_laadpaal["ChargerTime_InMin"].median())


#df_laadpaal = df_laadpaal.drop(df_laadpaal[df_laadpaal['ChargerTime_InMin'] < 0], inplace = True, axis = 1)


# In[7]:


code1= '''
#Naar minuten converteren, makkelijker te lezen
df_laadpaal['ConnectedTime_InMin'] = df_laadpaal['ConnectedTime'] * 60 
df_laadpaal['ChargerTime_InMin'] = df_laadpaal['ChargeTime'] * 60

#Onbenutte tijd.
df_laadpaal['noncharging'] = df_laadpaal['ConnectedTime_InMin'] - df_laadpaal['ChargerTime_InMin']
#df_laadpaal["ChargerTime_InMin"].max()

#Laadpalen laadtijd gefilterd op reele laadtijden. 
df_laadpaal = df_laadpaal[df_laadpaal.ChargerTime_InMin > 0]
df_laadpaal = df_laadpaal[df_laadpaal.ChargerTime_InMin < 397]

print("Mean ", df_laadpaal["ChargerTime_InMin"].mean())
print("Median ", df_laadpaal["ChargerTime_InMin"].median())


#df_laadpaal = df_laadpaal.drop(df_laadpaal[df_laadpaal['ChargerTime_InMin'] < 0], inplace = True, axis = 1)
'''
st.code(code1, language = 'python')


# In[8]:


st.subheader('Boxplot om te kijken hoe de data is gedistributeerd, outliers zijn weg gehaald')
        


# In[9]:


#Boxplot om outliers te bepalen.
fig_box = px.box(df_laadpaal, y="ChargerTime_InMin")
fig_box.show()
st.plotly_chart(fig_box)


# In[10]:


st.subheader('Hieronder een histogram waarin de laadtijd en extra laadtijd wordt afgebeeld')


# In[11]:


#Visualisatie van charging vs non charging
fig_rood, ax = plt.subplots()
labels = ['Non-charge time', 'Charge time']
ax.hist([df_laadpaal['noncharging'], df_laadpaal['ChargerTime_InMin']],stacked = True, bins = 10, range = [1, 485])
plt.title('Charge time vs non-charge time')
plt.legend(labels)
plt.show()
st.pyplot(fig_rood)


# In[12]:


st.subheader('histogram van de oplaadtijd')


# In[13]:


fig_oplaad = px.histogram(df_laadpaal, x="ChargerTime_InMin", labels={"ChargerTime_InMin":"Oplaadtijd in minuten"})

fig_oplaad.add_annotation(x=138, y=400,
            text="Mean : 140.09442977168854",
            showarrow=True,
            arrowhead=1)
fig_oplaad.add_annotation(x=133, y=320,
            text="Median  133.998",
            showarrow=True)


fig_oplaad.show()
st.plotly_chart(fig_oplaad)


# In[14]:


st.subheader('kansdichtheidsfunctie van oplaadtijd')


# In[15]:


fig_kans = px.histogram(df_laadpaal, x="ChargerTime_InMin", marginal="rug",
                   hover_data=df_laadpaal.columns, labels={"ChargerTime_InMin":"Oplaadtijd in minuten"})
fig_kans.show()
st.plotly_chart(fig_kans)


# In[ ]:





# In[65]:


df_nan = df_laadpaal.isna().sum()
df_nan
st.dataframe(df_nan)
#Geen duplicates, dus we hoeven bij deze dataset niks te vervangen/verwijderen.


# In[17]:


st.subheader("'Gemiddelde laadprofiel bij het opladen van auto's")


# In[18]:


fig_lijn, ax = plt.subplots()
ax.plot(df_laadpaal['Ended'][0:47], df_laadpaal['MaxPower'][0:47])
plt.title('Laadprofiel 2018-01-01')
plt.xticks(np.arange(0,0, step=1), rotation = 90)
plt.xlabel('tijd')
plt.ylabel('Max power (W)')
plt.show()
st.pyplot(fig_lijn)


# In[19]:


st.subheader('openchargemap data')
st.text('De data van de openchargemap wordt met api ingeladen')


# In[20]:


code2= '''
key='f3985c54-660c-43a6-9359-1f0c0bc73f0c'
url = "https://api.openchargemap.io/v3/poi/?key=key"
querystring = {"camelcase":"true","key":"",'countrycode':'NL', 'compact':True, 'maxresults':7990}
headers = {"Content-Type": "application/json"}

response = requests.request("GET", url,headers=headers, params=querystring)
data = response.text

opencharge_data = json.loads(data)
'''
st.code(code2, language = 'python')


# In[21]:


key='f3985c54-660c-43a6-9359-1f0c0bc73f0c'
url = "https://api.openchargemap.io/v3/poi/?key=key"
querystring = {"camelcase":"true","key":"",'countrycode':'NL', 'compact':True, 'maxresults':7990}
headers = {"Content-Type": "application/json"}

response = requests.request("GET", url,headers=headers, params=querystring)
data = response.text

opencharge_data = json.loads(data)

#opencharge_data


# In[22]:


st.text('Kolommen van de dataset')

code3= '''
df_laadpalen = pd.DataFrame(opencharge_data)
df_laadpalen.columns
'''
st.code(code3, language = 'python')


# In[23]:


df_laadpalen = pd.DataFrame(opencharge_data)
df_laadpalen.columns


# In[24]:


st.text('De kolom "addressInfo" wordt gebruikt om een map te kunnen maken met alle laadpalen in Nederland')


# In[25]:


code4= '''
df_adress = pd.DataFrame(df_laadpalen['addressInfo'])
data_folium = pd.concat([df_adress.addressInfo.apply(pd.Series), df_adress.drop('addressInfo', axis=1)], axis=1)
data_folium.info()
'''
st.code(code4, language = 'python')


# In[26]:


df_adress = pd.DataFrame(df_laadpalen['addressInfo'])
data_folium = pd.concat([df_adress.addressInfo.apply(pd.Series), df_adress.drop('addressInfo', axis=1)], axis=1)
data_folium.info()


# In[27]:


code5= '''
data_folium.isna().sum()
'''
st.code(code5, language = 'python')


# In[67]:


df_su=data_folium.isna().sum()
st.dataframe(df_su)


# In[29]:


st.text('Er is geen missende data voor latitude en longitude dus kan er een folium map gemaakt worden') 


# In[30]:


st.subheader('Map met alle laadpalen in nederland gevisualiseerd met geopandas')


# In[31]:


code6= '''
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
folium_static(i)
'''
st.code(code6, language = 'python')


# In[32]:



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
folium_static(i)


# In[33]:


st.subheader('Nu met Folium')


# In[34]:


code7= '''
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
'''
st.code(code7, language = 'python')


# In[35]:


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


# In[36]:


st.subheader('lineaire regressie')


# # Lineaire regressie

# In[37]:


st.text('Er zijn verschillende files geimporteerd')


# In[61]:


code8= '''
wegenbelastingdf = pd.read_csv("opbrengst-motorrijtuigenbelasting.csv")
elektrobelasting = pd.read_excel("elekautotovbelasting.xlsx")
mergedf = pd.read_excel("mergedf.xlsx")
'''
st.code(code8, language = 'python')


# In[39]:


wegenbelastingdf = pd.read_csv("opbrengst-motorrijtuigenbelasting.csv")
elektrobelasting = pd.read_excel("elekautotovbelasting.xlsx")
mergedf = pd.read_excel("mergedf.xlsx")


# In[40]:


code9= '''
elektrobelasting
'''
st.code(code9, language = 'python')


# In[68]:



st.dataframe(elektrobelasting)


# In[42]:


code10= '''
mergedf
'''
st.code(code10, language = 'python')


# In[69]:


st.dataframe(mergedf)


# In[44]:


code11= '''
wegenbelastingdf
'''
st.code(code11, language = 'python')


# In[71]:



st.dataframe(wegenbelastingdf)


# In[46]:


st.text('Er wordt een lineaire regressie gemaakt van')


# In[47]:


code12= '''
fig_reg = plt.figure(figsize=(10, 4))
sns.regplot(x=mergedf["Omzet (rijk)"], y=mergedf["Elektrische Autos"]).set(title="Lineaire regressie, Elektrische auto's t.o.v. Omzet motorrijtuigen belasting")
st.pyplot(fig_reg)
'''
st.code(code12, language = 'python')


# In[48]:


fig_reg = plt.figure(figsize=(10, 4))
sns.regplot(x=mergedf["Omzet (rijk)"], y=mergedf["Elektrische Autos"]).set(title="Lineaire regressie, Elektrische auto's t.o.v. Omzet motorrijtuigen belasting")
st.pyplot(fig_reg)


# # Dataset Electrische auto's

# In[49]:


st.subheader("Dataset elektrische auto's") 


# In[50]:


st.text('De api van het rdw lukte niet om in te laden, dus is er een csv van het internet gebruikt.')


# In[51]:


code13= '''
df_auto = pd.read_csv('lijngrafiek_autos.csv')
df_auto = df_auto.assign(Datum_tenaamstelling = pd.to_datetime(df_auto['Datum_eerste_afgifte_Nederland']))
df_auto.head()
'''
st.code(code13, language = 'python')


# In[72]:


df_auto = pd.read_csv('lijngrafiek_autos.csv')
df_auto = df_auto.assign(Datum_tenaamstelling = pd.to_datetime(df_auto['Datum_eerste_afgifte_Nederland']))
df_auto.head()
st.dataframe(df_auto)


# In[53]:


st.text('Vervolgens worden de kolommen "Year" en "Month" aangemaakt om de groei in autos te kunnen onderscheiden per jaar en maand.')


# In[54]:


code14= '''
df_auto['Years'] = df_auto['Datum_tenaamstelling'].dt.year
df_auto['Month'] = df_auto['Datum_tenaamstelling'].dt.month 

df_auto.head()
'''
st.code(code14, language = 'python')


# In[73]:


df_auto['Years'] = df_auto['Datum_tenaamstelling'].dt.year
df_auto['Month'] = df_auto['Datum_tenaamstelling'].dt.month 

df_auto.head()
st.dataframe(df_auto)


# In[56]:


st.text("De groei in auto's in het jaar 2020 wordt bekeken per brandstofsoort.")


# In[57]:


code15= '''
#Filter om het jaar op te stellen
df_auto2020 = df_auto[df_auto['Years']==2020]
df_auto2020.head()
'''
st.code(code15, language = 'python')


# In[74]:


#Filter om het jaar op te stellen
df_auto2020 = df_auto[df_auto['Years']==2020]
df_auto2020.head()
st.dataframe(df_auto2020)


# In[59]:


st.text("Dit wordt weergegeven in een lijnplot.")


# In[60]:


#dataset inladen
fig_brand = go.Figure()

fig_brand.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Diesel),
               name="Diesel",
               line=dict(color="#33CFA5")))

fig_brand.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Benzine),
               name="Benzine",
               visible=False,
               line=dict(color="#33CFA5", dash="dash")))

fig_brand.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Hybride),
               name="Hybride",
               line=dict(color="#F06A6A")))
fig_brand.add_trace(
    go.Scatter(x=list(df_auto2020.Month),
               y=list(df_auto2020.Elektriciteit),
               name="Elektriciteit",
               line=dict(color="#F06A6A")))


fig_brand.update_layout(
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
fig_brand.update_layout(
    annotations=[
        dict(text="Soort Brandstof:", showarrow=False,
        x=0, y=1.085, yref="paper", align="left")
    ]
)

fig_brand.show()
st.plotly_chart(fig_brand)

