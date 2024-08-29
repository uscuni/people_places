import geopandas as gpd
import pandas as pd
import numpy as np

def assign_clusters(data,clusters):
    '''
    example: largest_overlap, data = assign_clusters(demographics,clusters)
    '''
    data = data.copy()
    
    data = data.rename(columns={'': 'ID'})
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[("ID", "E")], 
                                            data[("ID", "N")]),crs='EPSG:3035')
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
    data['cluster','cluster'] =  data.droplevel(0, axis=1).merge(
        largest_overlap[['ID', 'label']], how='left', on='ID')['label']
    gdf['cluster','cluster'] =  gdf.droplevel(0, axis=1).merge(
        largest_overlap[['ID', 'label']], how='left', on='ID')['label']

    return gdf, largest_overlap, data