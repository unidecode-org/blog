---
layout: post
title: "Visualizando datos con Python: crímenes en Philadelphia"
date: 2018-09-23 19:13:52
image: '/assets/img/philadelphia-crime-data/portada.jpg'
show_image_inside: false
description: Vamos a practicar ciencia de datos visualizando y entendiendo el registro
no_markdown_description: false
category: 'data science'
tags:
    - 'data science'
    - 'python'
twitter_text: Vamos a practicar ciencia de datos visualizando y entendiendo el registro
introduction: Vamos a practicar ciencia de datos visualizando y entendiendo el registro
---

En esta oportunidad se practica visualización y estudio de datos usamos el registro de crímenes en Philadelphia, el cual se puede descargar desde el sitio web de [Kaggle](https://www.kaggle.com/mchirico/philadelphiacrimedata). Pesa poco más de 70 MB. Para este ejercicio vamos a descubrir qué nos dice la data y hacer inferencias.

Para este ejemplo en concreto, se usa:

* Python
* Pandas
* Seaborn
* Matplotlib
* Leaflet

Primero, importamos las librerías que vamos a usar.


```python
# Lectura de datos y visualización de gráficos
import pandas as pd
import seaborn as sns

# Para interpretar fechas
from datetime import datetime

#Para controlar las propiedades de nuestros gráficos
import matplotlib.pyplot as plt

# Para visualizar imagenes
import matplotlib.cbook as cbook
from matplotlib.pyplot import imread

# Para visualizar mapas usando leaflet
from ipyleaflet import *
```

Por otro lado, necesitaremos interpretar fechas, así que definimos un lambda para hacerlo de manera funcional y leemos el registro usando pandas.


```python
def parseDatetime(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

parsedate = lambda x: parseDatetime(x)
```


```python
crimeData = pd.read_csv(\
    filepath_or_buffer="../resources/crime.csv",\
    header=0,\
    names=[\
        'Dc_Dist',\
        'Psa',\
        'Dispatch_Date_Time',\
        'Dispatch_Date',\
        'Dispatch_Time',\
        'Hour',\
        'Dc_Key',\
        'Location_Block',\
        'UCR_General',\
        'Text_General_Code',\
        'Police_Districts',\
        'Month',\
        'Lon',\
        'Lat'\
    ],\
    dtype={\
        'Dc_Dist':str,\
        'Psa':str,\
        'Dispatch_Date_Time':str,\
        'Dispatch_Date':str,\
        'Dispatch_Time':str,\
        'Hour':float,\
        'Dc_Key':str,\
        'Location_Block':str,\
        'UCR_General':str,\
        'Text_General_Code':str,\
        'Police_Districts':str,\
        'Month':str,\
        'Lon':float,\
        'Lat':float\
    },\
    parse_dates=["Dispatch_Date_Time"],\
    date_parser=parsedate\
)
```


```python
crimeData.head()
```




<div>

<div class="table-container">
    <table class="dataframe table">
    <thead class="thead-dark">
        <tr style="text-align: right;">
        <th>Index</th>
        <th>Dc_Dist</th>
        <th>Psa</th>
        <th>Dispatch_Date_Time</th>
        <th>Dispatch_Date</th>
        <th>Dispatch_Time</th>
        <th>Hour</th>
        <th>Dc_Key</th>
        <th>Location_Block</th>
        <th>UCR_General</th>
        <th>Text_General_Code</th>
        <th>Police_Districts</th>
        <th>Month</th>
        <th>Lon</th>
        <th>Lat</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <th>0</th>
        <td>18</td>
        <td>3</td>
        <td>2009-10-02 14:24:00</td>
        <td>2009-10-02</td>
        <td>14:24:00</td>
        <td>14.0</td>
        <td>200918067518</td>
        <td>S 38TH ST  / MARKETUT ST</td>
        <td>800</td>
        <td>Other Assaults</td>
        <td>NaN</td>
        <td>2009-10</td>
        <td>NaN</td>
        <td>NaN</td>
        </tr>
        <tr>
        <th>1</th>
        <td>14</td>
        <td>1</td>
        <td>2009-05-10 00:55:00</td>
        <td>2009-05-10</td>
        <td>00:55:00</td>
        <td>0.0</td>
        <td>200914033994</td>
        <td>8500 BLOCK MITCH</td>
        <td>2600</td>
        <td>All Other Offenses</td>
        <td>NaN</td>
        <td>2009-05</td>
        <td>NaN</td>
        <td>NaN</td>
        </tr>
        <tr>
        <th>2</th>
        <td>25</td>
        <td>J</td>
        <td>2009-08-07 15:40:00</td>
        <td>2009-08-07</td>
        <td>15:40:00</td>
        <td>15.0</td>
        <td>200925083199</td>
        <td>6TH CAMBRIA</td>
        <td>800</td>
        <td>Other Assaults</td>
        <td>NaN</td>
        <td>2009-08</td>
        <td>NaN</td>
        <td>NaN</td>
        </tr>
        <tr>
        <th>3</th>
        <td>35</td>
        <td>D</td>
        <td>2009-07-19 01:09:00</td>
        <td>2009-07-19</td>
        <td>01:09:00</td>
        <td>1.0</td>
        <td>200935061008</td>
        <td>5500 BLOCK N 5TH ST</td>
        <td>1500</td>
        <td>Weapon Violations</td>
        <td>20</td>
        <td>2009-07</td>
        <td>-75.130477</td>
        <td>40.036389</td>
        </tr>
        <tr>
        <th>4</th>
        <td>09</td>
        <td>R</td>
        <td>2009-06-25 00:14:00</td>
        <td>2009-06-25</td>
        <td>00:14:00</td>
        <td>0.0</td>
        <td>200909030511</td>
        <td>1800 BLOCK WYLIE ST</td>
        <td>2600</td>
        <td>All Other Offenses</td>
        <td>8</td>
        <td>2009-06</td>
        <td>-75.166350</td>
        <td>39.969532</td>
        </tr>
    </tbody>
    </table>
</div>

</div>



## Tipos de crímenes


```python
crimesCountedByType = crimeData.Text_General_Code\
    .value_counts()
```


```python
crimesCountedByType.head()
```

    All Other Offenses             437581
    Other Assaults                 277332
    Thefts                         257923
    Vandalism/Criminal Mischief    200345
    Theft from Vehicle             171135
    Name: Text_General_Code, dtype: int64



### Cantidad de crímenes por categoría


```python
# Seteamos el tamaño del gráfico
plt.subplots(figsize=(10, 10))

g = sns.barplot(x=crimesCountedByType.index.tolist(), y=crimesCountedByType.values);

# Hacemos que la leyenda sea mas facil de leer
g.set_xticklabels(rotation=90, labels=crimesCountedByType.index.tolist(), fontsize=12);
```


![png](/assets/img/philadelphia-crime-data/output_12_0.png)



```python
unidentifiedCrimes = crimesCountedByType\
    .drop("All Other Offenses")\
    .drop("Other Assaults")\
    .sum()
```


```python
identifiedCrimes = crimesCountedByType\
    .drop("Thefts")\
    .drop("Vandalism/Criminal Mischief")\
    .drop("Theft from Vehicle")\
    .drop("Narcotic / Drug Law Violations")\
    .drop("Fraud")\
    .drop("Recovered Stolen Motor Vehicle")\
    .drop("Burglary Residential")\
    .drop("Aggravated Assault No Firearm")\
    .drop("DRIVING UNDER THE INFLUENCE")\
    .drop("Robbery No Firearm")\
    .drop("Motor Vehicle Theft")\
    .drop("Robbery Firearm")\
    .drop("Disorderly Conduct")\
    .drop("Aggravated Assault Firearm")\
    .drop("Burglary Non-Residential")\
    .drop("Weapon Violations")\
    .drop("Other Sex Offenses (Not Commercialized)")\
    .drop("Prostitution and Commercialized Vice")\
    .drop("Rape")\
    .drop("Vagrancy/Loitering")\
    .drop("Arson")\
    .drop("Liquor Law Violations")\
    .drop("Forgery and Counterfeiting")\
    .drop("Embezzlement")\
    .drop("Public Drunkenness")\
    .drop("Homicide - Criminal")\
    .drop("Offenses Against Family and Children")\
    .drop("Gambling Violations")\
    .drop("Receiving Stolen Property")\
    .drop("Homicide - Justifiable")\
    .drop("Homicide - Gross Negligence")\
    .sum()
```


```python
# Porcentaje de crímenes no identificados
100.0 * identifiedCrimes / (identifiedCrimes + unidentifiedCrimes)
```




    31.95938920186576




```python
crimesGroupByType = pd.Series(crimeData\
    .Text_General_Code\
    .value_counts()\
).reset_index()

crimesGroupByType.rename(columns = {'Text_General_Code': 'Total', 'index': 'Category'}, inplace=True)

crimesGroupByType.set_index("Category", inplace=True)
```


```python
totalCrimes = crimesGroupByType.Total.sum()
```


```python
# Porcentaje de crímenes por categoría
meanCrimesInCategory = crimesGroupByType\
    .Total\
    .apply(lambda totalInCategory: (100.0 * totalInCategory / totalCrimes))
```

### Porcentaje de crímenes registrados por categoría


```python
plt.subplots(figsize=(10, 10))

g = sns.barplot(x=meanCrimesInCategory.index.tolist(), y=meanCrimesInCategory.values);

g.set_xticklabels(rotation=90, labels=meanCrimesInCategory.index.tolist(), fontsize=12);
```


![png](/assets/img/philadelphia-crime-data/output_20_0.png)


### Densidad de crímenes por categoría


```python
sns.heatmap(crimesGroupByType)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12aae1350>




![png](/assets/img/philadelphia-crime-data/output_22_1.png)


### Conclusiones

La mayoría de los crímenes en la ciudad de Filadelfia (69%) no están identificados bajo una categoría propia, esto puede significar que acumulan varias categorías menores, o bien, como en las categorías específicas de crímenes están los tipos más graves, es posible que "Otros" contengan los de menor gravedad. No obstante, entre los crímenes identificados más comunes están los Robos, el vandalismo y el uso, tenencia o distribución de narcóticos y entre los menos comunes están los homicidios.

## Incidencia de crímenes por año


```python
crimeData.Month = crimeData.Month.apply(lambda givenMonth: datetime.strptime(givenMonth, "%Y-%m"));
```


```python
crimesByYears = crimeData.Month.map(lambda x: x.year).value_counts()
```


```python
crimesByYears
```




    2006    234755
    2007    223902
    2008    223735
    2009    205044
    2010    199415
    2012    196755
    2011    195521
    2013    186489
    2014    186146
    2015    183300
    2016    169101
    2017     33442
    Name: Month, dtype: int64




```python
g = sns.barplot(y=crimesByYears.index.tolist(), x=crimesByYears.values, log=True);
g.set_xticklabels(rotation=90, labels=crimesByYears.index.tolist(), fontsize=12);
```


![png](/assets/img/philadelphia-crime-data/output_28_0.png)



```python
100.0 * crimesByYears[2016] / crimesByYears[2006]
```




    72.03297054375838



### Conclusiones

La incidencia de crímenes tiene una tendencia bajista de hecho, en 10 años se ha reducido en casi un 30%.

## Crímenes por ubicación

Veamos primeramente en un mapa los registros que tenemos, haciendo una ventana aleatorea de la mitad de los datos. 


```python
totalCrimes = crimeData.dropna().shape[0]

crimeDataSample = crimeData.sample(totalCrimes // 2)

lats = (crimeDataSample.Lat * 10e4)
longs = (crimeDataSample.Lon * 10e4)
```


```python
sns.scatterplot(y=lats, x=longs, alpha=0.1, s = 2, legend = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12a99f390>




![png](/assets/img/philadelphia-crime-data/output_34_1.png)


### ¿Donde se han registrado delitos en el 2016?


```python
recentCrimesMask = crimeData.Month.map(lambda x: x.year) == 2016
recentCrimes = crimeData[recentCrimesMask]

lats = (recentCrimes.Lat.dropna() * 10e4)
longs = (recentCrimes.Lon.dropna() * 10e4)

crimeMap = pd.DataFrame({ 'x': longs, 'y': lats })
```


```python
sns.scatterplot(y=crimeMap.y, x=crimeMap.x, alpha=0.1, s = 2, legend = False)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x127a2e310>




![png](/assets/img/philadelphia-crime-data/output_37_1.png)


### Conclusión

En la zona del aeropuerto de Filadelfia y en la zona limítrofe con Pensilvania, los crímenes son menores que en el resto de las zonas.


```python
recentCrimesTypes = recentCrimes.Text_General_Code
```


```python
sns.scatterplot(y=crimeMap.y, x=crimeMap.x, hue = recentCrimesTypes, alpha=.5, s = 5, legend = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12f13a890>




![png](/assets/img/philadelphia-crime-data/output_40_1.png)


## Zonas en Filadelfia con mayor criminalidad


```python
philadelphiaMap = cbook.get_sample_data('/Users/cnexans/personal/jupyter-guide/data_science_course/resources/map.jpeg')
philadelphiaImage = imread(philadelphiaMap)
plt.imshow(philadelphiaImage, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])
plt.show()

sns.kdeplot(lats, longs, cmap="Reds", shade=True, cut=0)
```


![png](/assets/img/philadelphia-crime-data/output_42_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x12d76bcd0>




![png](/assets/img/philadelphia-crime-data/output_42_2.png)



```python
crimesByPoliceDistricts = crimeData["Police_Districts"].value_counts().sort_values()
crimesByPoliceDistricts
```




    22      2818
    4      31113
    6      44444
    1      48008
    12     72198
    7      73207
    13     74514
    8      83426
    19     87183
    5      96025
    21     96956
    14    109907
    3     114689
    2     116180
    10    120481
    20    130293
    9     132875
    15    135628
    18    150186
    16    153103
    17    161245
    11    183196
    Name: Police_Districts, dtype: int64



## Estudiando por bloques específicos

Se puede apreciar en las gráficas anteriores que, existen dos zonas con mayor densidad de crímenes que en las demás (al centro al noreste) y también existe una zona densidad moderada-alta respecto a las demás. Veamos como podemos visualizar las zonas, utilizando el mapa de Filadelfia y para ello, tomaremos como ejemplo, la zona del centro.

### Crímenes en el centro de Filadelfia


```python
# Utilizamos las gráficas para tomar coordenadas de referencia

# Hacemos mascaras de índices que cumplan con estar dentro del centro
longitudeMask = crimeData.Lon.map(lambda x: (x > -75.2) and (x < -75.15)) == 1.0
latitudeMask = crimeData.Lat.map(lambda x: (x > 39.925) and (x < 39.975)) == 1.0

crimesInCenterBlockMask = (longitudeMask) & (latitudeMask)

# Aplicamos la máscara a la data
crimesInCenterBlock = crimeData[crimesInCenterBlockMask]

# Mostramos los bloques que hemos tomado
crimesInCenterBlock["Location_Block"].value_counts().head()
```




    1000 BLOCK MARKET ST              3970
    1300 BLOCK MARKET ST              2735
    1500 BLOCK MARKET ST              2331
    200 BLOCK S 13TH ST               1702
    1500 BLOCK JOHN F KENNEDY BLVD    1608
    Name: Location_Block, dtype: int64




```python
# Visualizamos el registro con puntos
sns.scatterplot(\
    y = crimesInCenterBlock.Lat, \
    x = crimesInCenterBlock.Lon, \
    hue = crimesInCenterBlock.Text_General_Code, \
    alpha = .5, \
    s = 5, \
    legend = False \
)

# La zona con menos puntos, es el Río Schuylkill
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12cb27f90>




![png](/assets/img/philadelphia-crime-data/output_47_1.png)



```python
# Para poner en contexto, utilizamos leaflet para dibujar un cuadro con la zona que estudiamos
m = Map(center=(39.925, -75.2), zoom=10, basemap=basemaps.OpenStreetMap.Mapnik);

polygon = Polygon(
    locations = [(39.925, -75.2), (39.975, -75.2), (39.975, -75.15), (39.925, -75.15)],
    color = "green",
    fill_color = "green"
);

m.add_layer(polygon);

m

```

{% include widgets/philadelphia-map.html %}


#### Tipos de crímenes en la zona del centro


```python
crimesInCenterBlock["Text_General_Code"].value_counts().head()
```




    All Other Offenses             67236
    Thefts                         64685
    Other Assaults                 34626
    Theft from Vehicle             30414
    Vandalism/Criminal Mischief    27365
    Name: Text_General_Code, dtype: int64




```python
100.0 * (crimesInCenterBlock.count() / crimeData.count())[0]
```




    15.315705855144227



### Conclusion

Los crímenes en la zona central de hecho, representan un 15% del total.

## Horas con mayor número de crímenes registrados


```python
totalCrimes = crimeData.dropna().shape[0]
sns.distplot(crimeData.sample(totalCrimes // 2).Hour);
```


![png](/assets/img/philadelphia-crime-data/output_54_0.png)


### Conclusion

Los horarios en los que se registran menos crímenes es en la mañana, concretamente entre las 4am y 6am.
