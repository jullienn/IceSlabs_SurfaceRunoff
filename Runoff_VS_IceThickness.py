# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:14:01 2022

@author: jullienn
"""
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Transformer
import cartopy.crs as ccrs
import datetime as dt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import BallTree
import pickle

### -------------------------- Load GrIS DEM ----------------------------- ###
#This is from paper Greenland Ice Sheet Ice Slab Expansion and Thickening, function 'extract_elevation.py'
#https://towardsdatascience.com/reading-and-visualizing-geotiff-images-with-python-8dcca7a74510
import rasterio
from rasterio.plot import show
path_GrIS_DEM = r'C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/elevations/greenland_dem_mosaic_100m_v3.0.tif'
GrIS_DEM = rasterio.open(path_GrIS_DEM)
### -------------------------- Load GrIS DEM ----------------------------- ###

#Define path flowlines
path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'
#Open flowlignes
flowlines_polygones=gpd.read_file(path_data+'flowlines_stripes/flowlines_nicolas/Ys_polygons__test_W20km.shp')

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

### ---------------------------- Load dataset ---------------------------- ###
#Dictionnaries have already been created, load them
path_df_with_elevation='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/' 

#Load 2010-2018 high estimate
f_20102018_high = open(path_df_with_elevation+'final_excel/high_estimate/df_20102018_with_elevation_high_estimate_rignotetalregions', "rb")
df_2010_2018_high = pickle.load(f_20102018_high)
f_20102018_high.close()
### ---------------------------- Load dataset ---------------------------- ###

#Load max Ys from Machguth et al., (2022)
table_complete_annual_max_Ys=pd.read_csv(path_data+'_table_complete_annual_max_Ys.csv',delimiter=';',decimal=',')

#Load Emax from Tedstone and Machguth (2022)
Emax_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd.csv',delimiter=',',decimal='.')
#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})
#Emax_TedMach.drop(columns=['Unnamed: 0'])

Emax_plus_Mad_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_plus_mad.csv',delimiter=',',decimal='.')
Emax_Ted_minus_Mad_Mach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_minus_mad.csv',delimiter=',',decimal='.')

#Plot to check
fig = plt.figure(figsize=(10,5))
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)

#Display GrIS drainage bassins
flowlines_polygones.plot(ax=ax1,color='orange', edgecolor='black',linewidth=0.5)
ax1.scatter(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413'],c=df_2010_2018_high['20m_ice_content_m'],s=0.1)
ax1.scatter(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y'],c=table_complete_annual_max_Ys['year'],s=10,cmap='magma')
ax1.scatter(Emax_TedMach['x'],Emax_TedMach['y'],c=Emax_TedMach['year'],s=5,cmap='magma')

plt.show()
#Define width of the buffer for ice slabs pick up
buffer=10

#Define df_2010_2018_high, Emax_TedMach, table_complete_annual_max_Ys as being a geopandas dataframes
points_ice = gpd.GeoDataFrame(df_2010_2018_high, geometry = gpd.points_from_xy(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413']),crs="EPSG:3413")
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")
points_Ys = gpd.GeoDataFrame(table_complete_annual_max_Ys, geometry = gpd.points_from_xy(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y']),crs="EPSG:3413")

for indiv_index in flowlines_polygones.index:
    
    if (indiv_index != 10):
        continue
    
    print(indiv_index)
    indiv_polygon=flowlines_polygones[flowlines_polygones.index==indiv_index]
    
    #Plot to check
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(10, 9)
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    ax1 = plt.subplot(gs[0:10, 0:3],projection=crs)
    ax2 = plt.subplot(gs[0:10, 3:6],projection=crs)
    ax3 = plt.subplot(gs[0:10, 6:9])

    #Display GrIS drainage bassins
    indiv_polygon.plot(ax=ax1,color='orange', edgecolor='black',linewidth=0.5)
    indiv_polygon.plot(ax=ax2,color='orange', edgecolor='black',linewidth=0.5)

    #Intersection between ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_ice = gpd.sjoin(points_ice, indiv_polygon, op='within')
    #plot
    ax1.scatter(within_points_ice['lon_3413'],within_points_ice['lat_3413'],c=within_points_ice['20m_ice_content_m'],s=0.1)
    
    #Intersection between Emax and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Emax = gpd.sjoin(points_Emax, indiv_polygon, op='within')
    #plot
    ax1.scatter(within_points_Emax['x'],within_points_Emax['y'],c=within_points_Emax['year'],s=5,cmap='magma')

    #Intersection between Ys and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Ys = gpd.sjoin(points_Ys, indiv_polygon, op='within')
    #plot
    ax1.scatter(within_points_Ys['X'],within_points_Ys['Y'],c=within_points_Ys['year'],s=10,cmap='magma')
    
    plt.show()
    
    for indiv_year in list([2010,2011,2012,2013,2014,2017,2018]):#list([2012,2016,2019]):#np.asarray(within_points_Ys.year):
        #Select ice slabs data of the current indiv_year
        subset_iceslabs=within_points_ice[within_points_ice.year==indiv_year]
        
        #Display the tracks of the current year within the polygon
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],color='purple',s=10)
        
        #Display the Ys of the current indiv_year
        ax2.scatter(within_points_Ys[within_points_Ys.year==indiv_year]['X'],within_points_Ys[within_points_Ys.year==indiv_year]['Y'],color='black',s=10)
        
        '''
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],c=subset_iceslabs['20m_ice_content_m'],s=10)
        ax2.scatter(within_points_Emax[within_points_Emax.year==indiv_year]['x'],within_points_Emax[within_points_Emax.year==indiv_year]['y'],color='blue',s=10)
        #Display the whole track
        ax2.scatter(df_2010_2018_csv[df_2010_2018_csv.Track_name==indiv_track]['lon_3413'],df_2010_2018_csv[df_2010_2018_csv.Track_name==indiv_track]['lat_3413'],color='black',s=10)
        '''

        if (len(subset_iceslabs_retained)==0):
            #No slab for this particular year, continue
            continue
        
        #Define Ys point
        Ys_point=np.transpose(np.asarray([np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['X']),np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['Y'])]))   
        
        #Extract elevation of Ys
        for val in GrIS_DEM.sample(Ys_point): 
            #Calculate the corresponding elevation
            Ys_point_elevation=val[0]
        
        #Keep only data where elevation is within elevation+/-buffer
        subset_iceslabs_buffered=subset_iceslabs[np.logical_and(subset_iceslabs['elevation']<=(Ys_point_elevation+buffer),subset_iceslabs['elevation']>=(Ys_point_elevation-buffer))]
        
        #Display the ice slabs points that are inside this buffer
        ax2.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
        
        #Display the slab thickness distribution
        ax3.hist(subset_iceslabs_buffered['20m_ice_content_m'])
        print(indiv_year)
        plt.show()
        pdb.set_trace()
        
        #if vari > than x, add a label in df_2010_2018_csv to indicate not to use it, or just discard it
        # - do yearly maps!
    
    #Create a dataset of iceslabs, Emax and Ys for this stripe
    


#1. Select flowlines



#2. Pick up all perpendicular to contours transects

#3. Extract Emax and Ys for each year in each polygone

#4. Extract corresponding ice slab thickness