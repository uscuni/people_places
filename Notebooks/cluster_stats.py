import geopandas as gpd
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

clusters = gpd.read_parquet('clusters_freiburg_300.pq').reset_index()
clusters['label']=clusters.index

def assign_clusters(data,clusters):

    data = data.copy()
    
    data = data.rename(columns={'': 'ID'})
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[("ID", "E")], data[("ID", "N")]),crs='EPSG:3035')
    gdf = gdf.rename(columns={'': 'geometry'})
    gdf = gdf.drop(['N','E'],axis=1,level=1)
    
    # Buffer the points using a square cap style
    # Note cap_style: round = 1, flat = 2, square = 3
    gdf['geometry'] = gdf['geometry'].buffer(50, cap_style = 3)
    
    cells = gdf[['ID','geometry']]
    cells.columns =  cells.columns.get_level_values(0)
    
    # Overlay cells with clusters to get intersections
    overlap = gpd.overlay(cells, clusters, how='intersection')
    
    # Calculate the area of overlap
    overlap['area'] = overlap.geometry.area

    # Find the cluster with the largest overlap for each cell
    largest_overlap = overlap.loc[overlap.groupby(overlap['ID'])['area'].idxmax()]

    # Merge this back with the original cells DataFrame to attach cluster information
    data['cluster','cluster'] =  data.droplevel(0, axis=1).merge(largest_overlap[['ID', 'label']], on='ID')['label']

    return largest_overlap, data

### DEMOGRAPHICS

demographics = pd.read_parquet('/data/processed_data/Bevoelkerung100M.parquet').reset_index()
largest_overlap, data = assign_clusters(demographics,clusters)

#calculate % german
data['STATS','% German']=data['NATIONALITY','Germany'].fillna(0)/data[' INSGESAMT','Total']

#calculate average age
age = data['AGE_10']

data['STATS','Average Age'] = ((age['Under 10'].fillna(0)*5 + age['10 - 19'].fillna(0)*15 + age['20 - 29'].fillna(0)*25 
                               + age['30 - 39'].fillna(0)*35 + age['40 - 49'].fillna(0)*45 + age['50 - 59'].fillna(0)*55 
                               + age['60 - 69'].fillna(0)*65 + age['70 - 79'].fillna(0)*75 + age['80 and older'].fillna(0)*85) 
                               / age[['Under 10', '20 - 29', '60 - 69', '10 - 19','40 - 49', '50 - 59','30 - 39', '70 - 79',
                                      '80 and older']].sum(axis=1))

d = []
for i in range(0,int(data['cluster','cluster'].max())):
    d.append(
        {
            '% German': data[data['cluster','cluster']==i]['STATS','% German'].mean(),
            '% German count': data[data['cluster','cluster']==i]['STATS','% German'].count(),
            'Average age': data[data['cluster','cluster']==i]['STATS','Average Age'].mean(),
            'Average age count': data[data['cluster','cluster']==i]['STATS','Average Age'].count()
        }
    )

demographics_stats = pd.DataFrame(d)
demographics_stats.to_parquet('demographics_stats.parquet')

### BUILDINGS

data = pd.read_parquet('/data/processed_data/Geb100m.parquet').reset_index()
data = data.rename(columns={'': 'ID'})
data['cluster','cluster'] =  data.droplevel(0, axis=1).merge(largest_overlap[['ID', 'label']], on='ID')['label']

age = data['BUILDING_YEAR']
data['STATS','Building year'] = ((age['1919 - 1948'].fillna(0)*1934 + age['1949 - 1978'].fillna(0)*1964 
                                 + age['2001 - 2004'].fillna(0)*2003 + age['1987 - 1990'].fillna(0)*1989 
                                 + age['1996 - 2000'].fillna(0)*1998 + age['Before 1919'].fillna(0)*1900 
                                 + age['1979 - 1986'].fillna(0)*1983 + age['2005 - 2008'].fillna(0)*2007 
                                 + age['2009 and later'].fillna(0)*2010 + age['1991 - 1995'].fillna(0)*1993)
                                 /age[['1919 - 1948','1949 - 1978', '2001 - 2004', '1987 - 1990', '1996 - 2000', 'Before 1919',
                                       '1979 - 1986', '2005 - 2008', '2009 and later', '1991 - 1995']].sum(axis=1))

data['STATS','Apartment no'] = (data['APARTMENT_NO']['1 apartment'].fillna(0)*1 
                                + data['APARTMENT_NO']['2 apartments'].fillna(0)*2 
                                + data['APARTMENT_NO']['3 - 6 apartments'].fillna(0)*4.5 
                                + data['APARTMENT_NO']['7 - 12 apartments'].fillna(0)*9.5 
                                + data['APARTMENT_NO']['13 and more apartments'].fillna(0)*15)/data['APARTMENT_NO'].sum(axis=1)

d = []
for i in range(0,int(data['cluster','cluster'].max())):
    d.append(
        {
            'Building year': data[data['cluster','cluster']==i]['STATS','Building year'].mean(),
            'Building year count': data[data['cluster','cluster']==i]['STATS','Building year'].count(),
            'Apartment no. mean': data[data['cluster','cluster']==i]['STATS','Apartment no'].mean(),
            'Apartment no. count': data[data['cluster','cluster']==i]['STATS','Apartment no'].count()
        }
    )

buildings_stats = pd.DataFrame(d)
buildings_stats.to_parquet('buildings_stats.parquet')

### APARTMENTS

data = pd.read_parquet('/data/processed_data/Wohnungen100m.parquet').reset_index()
data = data.rename(columns={'': 'ID'})
data['cluster','cluster'] =  data.droplevel(0, axis=1).merge(largest_overlap[['ID', 'label']], on='ID')['label']

data['STATS','Rented for residential purposes'] = ((data['USE_TYPE','Rented: with currently managed household'].fillna(0) 
  + data['USE_TYPE','Rented: without currently managed household'].fillna(0)) 
  / data['USE_TYPE'].sum(axis=1))

data['STATS','Holiday apartment'] = (data['USE_TYPE','Holiday and leisure apartment'].fillna(0) 
                                                 / data['USE_TYPE'].sum(axis=1))

data['STATS','Average floor space'] = ((data['FLOOR_SPACE']['Under 30'].fillna(0)*20 
+ data['FLOOR_SPACE']['30 - 39'].fillna(0)*34.5
+ data['FLOOR_SPACE']['40 - 49'].fillna(0)*44.5 + data['FLOOR_SPACE']['50 - 59'].fillna(0)*54.5
+ data['FLOOR_SPACE']['60 - 69'].fillna(0)*64.5 + data['FLOOR_SPACE']['70 - 79'].fillna(0)*74.5
+ data['FLOOR_SPACE']['80 - 89'].fillna(0)*84.5 + data['FLOOR_SPACE']['90 - 99'].fillna(0)*94.5
+ data['FLOOR_SPACE']['100 - 109'].fillna(0)*104.5 + data['FLOOR_SPACE']['110 - 119'].fillna(0)*114.5
+ data['FLOOR_SPACE']['120 - 129'].fillna(0)*124.5 + data['FLOOR_SPACE']['130 - 139'].fillna(0)*134.5
+ data['FLOOR_SPACE']['140 - 149'].fillna(0)*144.5 + data['FLOOR_SPACE']['150 - 159'].fillna(0)*154.5
+ data['FLOOR_SPACE']['160 - 169'].fillna(0)*164.5 + data['FLOOR_SPACE']['170 - 179'].fillna(0)*174.5
+ data['FLOOR_SPACE']['180 and more'].fillna(0)*200) / data['FLOOR_SPACE'].sum(axis=1))

d = []
for i in range(0,int(data['cluster','cluster'].max())):
    d.append(
        {
            'Rented for residential purposes': data[data['cluster','cluster']==i]['STATS','Rented for residential purposes'].mean(),
            'Rented for residential purposes count': data[data['cluster','cluster']==i]['STATS','Rented for residential purposes'].count(),
            'Holiday apartment': data[data['cluster','cluster']==i]['STATS','Holiday apartment'].mean(),
            'Holiday apartment count': data[data['cluster','cluster']==i]['STATS','Holiday apartment'].count(),
            'Average floor space': data[data['cluster','cluster']==i]['STATS','Average floor space'].mean(),
            'Average floor space count': data[data['cluster','cluster']==i]['STATS','Average floor space'].count()
        }
    )

apartments_stats = pd.DataFrame(d)
apartments_stats.to_parquet('apartments_stats.parquet')