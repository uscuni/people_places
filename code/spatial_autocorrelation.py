import geopandas as gpd
import esda
import numpy as np
from libpysal import graph

def lisa(gdf,column_name,sig):
    '''
    Calculates moran's I for census geodataframes and returns a geodataframe of local spatially 
    autocorrelated values
    
    gdf: geodataframe of values
    column_name: string of column to be spatially autocorrelated
    sig: level of significance e.g. 0.05 (integer)

    example:
    print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")
    gdf_sig.explore("cluster", prefer_canvas=True, cmap=["#d7191c","#fdae61","#abd9e9","#2c7bb6","lightgrey"])
    '''
    
    not_nan = ~np.isnan(gdf[column_name])
    gdf = gdf[not_nan]
    contiguity = graph.Graph.build_contiguity(gdf, rook=False)
    contiguity_r = contiguity.transform("r")
    mi = esda.Moran(gdf[column_name], contiguity_r.to_W())

    gdf_sig = gdf.copy()
    lisa = esda.Moran_Local(gdf[column_name], contiguity_r.to_W())
    gdf_sig.loc[lisa.p_sim < sig, 'cluster'] = lisa.q[lisa.p_sim < sig]
    gdf_sig["cluster"] = gdf_sig["cluster"].fillna(0)
    gdf_sig["cluster"] = gdf_sig["cluster"].map(
        {
            0: "Not significant",
            1: "High-high",
            2: "Low-high",
            3: "Low-low",
            4: "High-low",
        }
    )
    return mi, gdf_sig