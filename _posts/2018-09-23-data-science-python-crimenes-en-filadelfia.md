---
layout: post
title: "Visualizando datos con Python: crímenes en Philadelphia"
date: 2018-09-23 19:13:52
image: '/assets/img/philadelphia-crime-data/portada.jpg'
show_image_inside: false
description: Vamos a practicar ciencia de datos visualizando y entendiendo el registro de crímenes de Filadelfia
no_markdown_description: false
category: 'data science'
tags:
    - 'data-science'
    - 'python'
twitter_text: Vamos a practicar ciencia de datos visualizando y entendiendo el registro de crímenes de Filadelfia
introduction: Vamos a practicar ciencia de datos visualizando y entendiendo el registro de crímenes de Filadelfia
---

En esta oportunidad se practica visualización y estudio de datos con el registro de crímenes en Philadelphia, que se puede descargar desde el sitio web de [Kaggle](https://www.kaggle.com/mchirico/philadelphiacrimedata). Para este ejercicio vamos a descubrir qué nos dice la data y hacer inferencias usando:

* Python
* Pandas
* Seaborn
* Matplotlib
* Leaflet

Primero, importamos las librerías:


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

Por otro lado, necesitaremos interpretar fechas, así que definimos un lambda para hacerlo de manera funcional mas adelante.


```python
def parseDatetime(x):
    return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

parsedate = lambda x: parseDatetime(x)
```

Leemos el registro de crímenes con pandas y echamos un primer vistazo a la tabla con el método [DataFrame.head](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html).

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

Accedemos a la columna de categorías de crímenes y con el método [Series.value_counts](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html) obtenemos el número de registros por cada tipo de crimen.

```python
# Cantidad de crímenes en cada categoría
crimesCountedByType = crimeData.Text_General_Code\
    .value_counts()

crimesCountedByType.head()
```

    All Other Offenses             437581
    Other Assaults                 277332
    Thefts                         257923
    Vandalism/Criminal Mischief    200345
    Theft from Vehicle             171135
    Name: Text_General_Code, dtype: int64



### Cantidad de crímenes por categoría

Utilizando la serie anterior, hacemos un gráfico de barras con el método [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html), para poner en cotexto estos datos de forma visual.

```python
# Seteamos el tamaño del gráfico
plt.subplots(figsize=(10, 10))

g = sns.barplot(x=crimesCountedByType.index.tolist(), y=crimesCountedByType.values);

# Hacemos que la leyenda sea mas facil de leer
g.set_xticklabels(rotation=90, labels=crimesCountedByType.index.tolist(), fontsize=12);
```


![png](/assets/img/philadelphia-crime-data/output_12_0.png)


En el gráfico se puede apreciar con mucha facilidad que la mayoría de los crímenes están registrados bajo *All other offenses* y *Other assaults* es decir, que no describen específicamente de qué crímenes se trata. **¿Cuantos crímenes estan bajo éste código de clasificación?**.

La clase [Series](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html) de pandas nos ofrece el método [Series.drop](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.drop.html#pandas.Series.drop) que remueve registros y [Series.sum](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.sum.html#pandas.Series.sum) el cual devuelve la cantidad de registros.


```python
# Cantidad de crímenes bajo categorías bien definidas
unidentifiedCrimes = crimesCountedByType\
    .drop("All Other Offenses")\
    .drop("Other Assaults")\
    .sum()
```


```python
# Cantidad de crímenes bajo las categorías de "Otros"
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

De ésta manera, podemos ver el porcentaje de crímenes que no están bien identificados.


```python
# Calculo de la relación porcentual
100.0 * identifiedCrimes / (identifiedCrimes + unidentifiedCrimes)
>> 31.95938920186576
```

Ahora bien, en el gráfico anterior no se pueden advertir, con exactitud, las relaciones de la cantidad de crímenes en cada categoría con respecto al total. **¿Cuáles son estas relaciones y cómo podemos visualizarlas?**.


```python
# Serie de datos con el conteo de crímenes
crimesGroupByType = pd.Series(crimeData\
    .Text_General_Code\
    .value_counts()\
).reset_index()

# Renombramos las columnas
crimesGroupByType.rename(columns = {\
    'Text_General_Code': 'Total',\
    'index': 'Category'
    },\
    inplace=True\
)

# Seteamos el índice en la columna de categorías
crimesGroupByType.set_index("Category", inplace=True)

# Sumamos la cantidades para obtener el total entre todos las categorías
totalCrimes = crimesGroupByType.Total.sum()

# Calculamos el porcentaje de crímenes por categoría
meanCrimesInCategory = crimesGroupByType\
    .Total\
    .apply(lambda totalInCategory: (100.0 * totalInCategory / totalCrimes))
```

### Porcentaje por categoría usando gráfico de barras


```python
plt.subplots(figsize=(10, 10))

g = sns.barplot(x=meanCrimesInCategory.index.tolist(), y=meanCrimesInCategory.values);

g.set_xticklabels(rotation=90, labels=meanCrimesInCategory.index.tolist(), fontsize=12);
```


![png](/assets/img/philadelphia-crime-data/output_20_0.png)


### Porcentaje por categoría usando mapa de calor


```python
sns.heatmap(crimesGroupByType)
```


![png](/assets/img/philadelphia-crime-data/output_22_1.png)


### Conclusiones

La mayoría de los crímenes en la ciudad de Filadelfia (69%) no están identificados bajo una categoría propia, esto puede significar que acumulan varias categorías menores, o bien, como en las categorías específicas de crímenes están los tipos más graves, es posible que "Otros" contenga los de menor gravedad.

No obstante, entre los crímenes identificados más comunes están los Robos, el vandalismo y el uso, tenencia o distribución de narcóticos y entre los menos comunes están los homicidios.

## Incidencia de crímenes por año

Estudiemos como se comportan los crímenes a través de los años usando métodos en la sección anterior. Recordemos que la columna que indica el mes del crimen, incluye también el año.

```python
# Interpretamos la columna de mes usando datetime
crimeData.Month = crimeData\
    .Month\
    .apply(lambda givenMonth: datetime.strptime(givenMonth, "%Y-%m"));

# Mapeamos la columna al año que indica
crimesByYears = crimeData\
    .Month\
    .map(lambda x: x.year)\
    .value_counts()

# Echamos un vistazo al resultado
crimesByYears
```

Como resultado, tenemos una *Serie* en la que cada año se relaciona con los crímenes registrados en esas fechas y, en primera instancia, se puede ver que los crímenes van decreciendo poco a poco en el tiempo. **¿Cómo podemos graficar y evidenciar éste resultado?**

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


### Crímenes por año utilizando gráfico de barras

```python
# Con el parámetro log, indicamos a seaborn que use una
g = sns.barplot(\
    y=crimesByYears.index.tolist(),\
    x=crimesByYears.values,\
    log=True\
);

# Rotamos las etiquetas para hacer mas facil la lectura
g.set_xticklabels(\
    rotation=90,\
    labels=crimesByYears.index.tolist(),\
    fontsize=12\
);
```


![png](/assets/img/philadelphia-crime-data/output_28_0.png)

**¿Cómo cambió la criminalidad entre el 2006 y el 2016?**

```python
100.0 * crimesByYears[2016] / crimesByYears[2006]
>> 72.03297054375838
```



### Conclusiones

La incidencia de crímenes tiene una tendencia bajista de hecho, en 10 años se ha reducido en casi un 30%.

## Crímenes por ubicación

Otra perspectiva que nos conscierne estudiar es **¿Cuales son las zonas en las que hay más crímen?**. Veamos primeramente en un gráfico de puntos las latitudes y longitudes, haciendo una ventana aleatorea de la mitad de los datos. Para ello, utilizamos el método [seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html).


```python
# Sacamos los registros sin latitud y longitud
totalCrimes = crimeData.dropna().shape[0]

# Hacemos una ventana de los datos
crimeDataSample = crimeData.sample(totalCrimes // 2)

# Tomamos la ubicación geográfica
lats = (crimeDataSample.Lat * 10e4)
longs = (crimeDataSample.Lon * 10e4)

sns.scatterplot(y=lats, x=longs, alpha=0.1, s = 2, legend = False)
```

Como resultado, se pueden contemplar que los puntos que rellenan todo el mapa de Filadelfia, con algunos sitios mas poblados que otros.

![png](/assets/img/philadelphia-crime-data/output_34_1.png)


### Mapa de crímenes en el 2016 usando gráfico de puntos


```python
# Generamos una mascara de los crímenes en el año 2016
recentCrimesMask = crimeData.Month.map(lambda x: x.year) == 2016

# Aplicamos la mascara a la data
recentCrimes = crimeData[recentCrimesMask]

# Obtenemos la ubicación correspondiente para cada caso
lats = (recentCrimes.Lat.dropna() * 10e4)
longs = (recentCrimes.Lon.dropna() * 10e4)

# Generamos un DataFrame con estos datos
crimeMap = pd.DataFrame({ 'x': longs, 'y': lats })

# Creamos un gráfico de puntos con scatterplot
sns.scatterplot(y=crimeMap.y, x=crimeMap.x, alpha=0.1, s = 2, legend = False)
```

![png](/assets/img/philadelphia-crime-data/output_37_1.png)

### Mapa de crímenes usando puntos coloreados por categoría


```python
# Obtenemos la columna de categorías desde el DataFrame
recentCrimesTypes = recentCrimes.Text_General_Code

# Utilizamos el parámetro hue en scatterplot, para diferenciar los tipos de puntos
sns.scatterplot(\
    y=crimeMap.y,\
    x=crimeMap.x,\
    hue = recentCrimesTypes,\
    alpha=.5,\
    s = 5,\
    legend = False\
)
```

![png](/assets/img/philadelphia-crime-data/output_40_1.png)

### Conclusión

En la zona del aeropuerto de Filadelfia y en la zona limítrofe con Pensilvania, los crímenes son menores que en el resto de las zonas.


## Zonas en Filadelfia con mayor criminalidad

Otro enfoque posible para la pregunta anterior es utilizar gráficos de densidad. Veamos dos cosas: un mapa de Filadelfia real y un gráfico de estimación de densidad por kernel.

```python
# Leemos la data que corresponde a la imagen
philadelphiaMap = cbook.get_sample_data(\
    'resources/map.jpeg'\
)

philadelphiaImage = imread(philadelphiaMap)

# Mostramos la imagen
plt.imshow(\
    philadelphiaImage,\
    zorder=0,\
    extent=[0.5, 8.0, 1.0, 7.0]\
)

plt.show()

# Creamos un gráfico de densidad por kernel
sns.kdeplot(\
    lats,\
    longs,\
    cmap="Reds",\
    shade=True,\
    cut=0\
)
```


![png](/assets/img/philadelphia-crime-data/output_42_0.png)


![png](/assets/img/philadelphia-crime-data/output_42_2.png)


## Estudiando por bloques específicos

Se puede apreciar en las gráficas anteriores que existen dos zonas con mayor densidad de crímenes que en las demás (al centro al noreste) y también existe una zona con densidad moderada-alta respecto a las demás. Veamos cómo podemos visualizar estas zonas, utilizando el mapa de Filadelfia y para ello, tomaremos como ejemplo, la zona del centro.

### Crímenes en el centro de Filadelfia

Comencemos con un gráfico de puntos como los que hicimos anteriormente. Notaremos que en esta zona, hay una curva en la que no aparecen crímenes, esto es porque ahí pasa el Río Schuylkill.


```python
# Utilizamos las gráficas para tomar coordenadas de referencia

# Hacemos mascaras de índices que cumplan con estar dentro del centro
longitudeMask = crimeData.Lon.map(lambda x: (x > -75.2) and (x < -75.15)) == 1.0
latitudeMask = crimeData.Lat.map(lambda x: (x > 39.925) and (x < 39.975)) == 1.0

crimesInCenterBlockMask = (longitudeMask) & (latitudeMask)

# Aplicamos la máscara a la data
crimesInCenterBlock = crimeData[crimesInCenterBlockMask]

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


![png](/assets/img/philadelphia-crime-data/output_47_1.png)

Para poner en contexto, utilizamos Leaflet para dibujar un cuadro con la zona que hemos estudiado. En una notebook de Jupyter, está la libreria *ipyleaflet*, que permite la exportación de widgets de mapas.

```python
# from ipyleaflet import *

# Creamos un mapa
m = Map(\
    center=(39.925, -75.2),\
    zoom=10,\
    basemap=basemaps.OpenStreetMap.Mapnik\
);

# Creamos un polígono con el cuadro
polygon = Polygon(\
    locations = [\
        (39.925, -75.2),\
        (39.975, -75.2),\
        (39.975, -75.15),\
        (39.925, -75.15)\
    ],\
    color = "green",\
    fill_color = "green"\
);

# Agregamos el polìgono al mapa
m.add_layer(polygon);

# Y lo mostramos
m

```

{% include widgets/philadelphia-map.html %}


#### Tipos de crímenes en la zona del centro

Contando por categoría y viendo los primeros resultados, hacemos otro hallazgo.

```python
crimesInCenterBlock.Text_General_Code\
    .value_counts()\
    .head()
```


    All Other Offenses             67236
    Thefts                         64685
    Other Assaults                 34626
    Theft from Vehicle             30414
    Vandalism/Criminal Mischief    27365
    Name: Text_General_Code, dtype: int64


Es claro que los robos *Thefts* representan una parte importante de los crímenes en esta zona. No obstante, volvamos a la pregunta **¿Qué porcentaje de crímenes representan los del centro de Filadelfia?**


```python
100.0 * (crimesInCenterBlock.count() / crimeData.count())[0]
>> 15.315705855144227
```

### Conclusion

Los crímenes en la zona central de hecho, representan un 15% del total.

## Crímenes por hora

Otra pregunta interesante que nos podemos hacer es **¿Cual es la distribución de ocurrencia de crímenes respecto las horas?**. Podemos conjeturar momentos mas seguros o inseguros en el día. Veamos una gráfica de distribución que nos explique los hechos con el método [seaborn.distplot](https://seaborn.pydata.org/generated/seaborn.distplot.html)


```python
totalCrimes = crimeData.dropna().shape[0]
sns.distplot(crimeData.sample(totalCrimes // 2).Hour);
```

![png](/assets/img/philadelphia-crime-data/output_54_0.png)


### Conclusion

Los horarios en los que se registran menos crímenes es en la mañana, concretamente entre las 4am y 6am.
