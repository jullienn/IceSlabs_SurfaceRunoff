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

'''
#Define path 2010 2018 data
path_2010_2018='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/final_excel/high_estimate/'
#Load all 2010-2018 data
df_2010_2018_csv = pd.read_csv(path_2010_2018+'Ice_Layer_Output_Thicknesses_2010_2018_jullienetal2021_high_estimate.csv',delimiter=',',decimal='.')
#Transform the coordinated from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
points=transformer.transform(np.asarray(df_2010_2018_csv["lon"]),np.asarray(df_2010_2018_csv["lat"]))
#Store lat/lon in 3413
df_2010_2018_csv['lon_3413']=points[0]
df_2010_2018_csv['lat_3413']=points[1]
#Extract year and create a new column
df_2010_2018_csv['year']=[int(df_2010_2018_csv.iloc[x]['Track_name'][0:4]) for x in range(len(df_2010_2018_csv))]
'''
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

'''
#Extract year, this is from https://stackoverflow.com/questions/41918115/how-do-i-extract-the-date-year-month-from-pandas-dataframe
Emax_TedMach['date']=pd.to_datetime(Emax_TedMach['date'])
Emax_TedMach['year'] = Emax_TedMach['date'].dt.year
'''
Emax_plus_Mad_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_plus_mad.csv',delimiter=',',decimal='.')
'''
Emax_plus_Mad_TedMach['date']=pd.to_datetime(Emax_plus_Mad_TedMach['date'])
Emax_plus_Mad_TedMach['year'] = Emax_plus_Mad_TedMach['date'].dt.year
'''
Emax_Ted_minus_Mad_Mach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_minus_mad.csv',delimiter=',',decimal='.')
'''
Emax_Ted_minus_Mad_Mach['date']=pd.to_datetime(Emax_Ted_minus_Mad_Mach['date'])
Emax_Ted_minus_Mad_Mach['year'] = Emax_Ted_minus_Mad_Mach['date'].dt.year
'''

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
        subset_iceslabs=within_points_ice[within_points_ice.year==indiv_year]
        #Create a blank dataframe
        subset_iceslabs_retained=pd.DataFrame()
        #plot
        '''
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],c=subset_iceslabs['20m_ice_content_m'],s=10)
        ax2.scatter(within_points_Emax[within_points_Emax.year==indiv_year]['x'],within_points_Emax[within_points_Emax.year==indiv_year]['y'],color='blue',s=10)
        
        '''
        ax2.scatter(within_points_Ys[within_points_Ys.year==indiv_year]['X'],within_points_Ys[within_points_Ys.year==indiv_year]['Y'],color='black',s=10)
        
        #To do:
        # - exclude ice slabs thickness transect not good for processing: compare variation of lon VS lat??
        for indiv_track in np.unique(subset_iceslabs['Track_name']):
            vari=np.abs(np.mean(np.diff(subset_iceslabs[subset_iceslabs.Track_name==indiv_track]['lat_3413']))/np.mean(np.diff(subset_iceslabs[subset_iceslabs.Track_name==indiv_track]['lon_3413'])))
            print('vari: ',str(np.round(vari,2)))
            if (vari <1): #If vari=1, angle is ~45°
                '''
                #Display the whole track
                ax2.scatter(df_2010_2018_csv[df_2010_2018_csv.Track_name==indiv_track]['lon_3413'],df_2010_2018_csv[df_2010_2018_csv.Track_name==indiv_track]['lat_3413'],color='black',s=10)
                '''
                #Display the track within the polygon
                ax2.scatter(subset_iceslabs[subset_iceslabs.Track_name==indiv_track]['lon_3413'],subset_iceslabs[subset_iceslabs.Track_name==indiv_track]['lat_3413'],color='purple',s=10)
                #Append the dataframe
                subset_iceslabs_retained=pd.concat([subset_iceslabs_retained,subset_iceslabs[subset_iceslabs.Track_name==indiv_track]])
            else:
                #Display the track that is not retained within the polygon
                ax2.scatter(subset_iceslabs[subset_iceslabs.Track_name==indiv_track]['lon_3413'],subset_iceslabs[subset_iceslabs.Track_name==indiv_track]['lat_3413'],color='red',s=10)
                
        if (len(subset_iceslabs_retained)==0):
            #No slab for this particular year, continue
            continue
        
        #Define Ys point
        Ys_point=np.transpose(np.asarray([np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['X']),np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['Y'])]))   
        
        '''
        #Define the elevation vector as full of nan
        subset_iceslabs_retained['elevation']=np.nan
        
        #Elevation extraction is from paper Greenland Ice Sheet Ice Slab Expansion and Thickening, function 'extract_elevation.py'
        for i in range(0,subset_iceslabs_retained.size):
            #This is from https://gis.stackexchange.com/questions/190423/getting-pixel-values-at-single-point-using-rasterio
            #Extract elevation in ice slabs dataset
            for val in GrIS_DEM.sample([(subset_iceslabs_retained['lon_3413'].iloc[i], subset_iceslabs_retained['lat_3413'].iloc[i])]): 
                #Calculate the corresponding elevation
                subset_iceslabs_retained['elevation'].iloc[i]=val
        '''
        #Extract elevation of Ys
        for val in GrIS_DEM.sample(Ys_point): 
            #Calculate the corresponding elevation
            Ys_point_elevation=val[0]
                
        #Keep only data where elevation is within elevation+/-buffer
        subset_iceslabs_retained_buffered=subset_iceslabs_retained[np.logical_and(subset_iceslabs_retained['elevation']<=(Ys_point_elevation+buffer),subset_iceslabs_retained['elevation']>=(Ys_point_elevation-buffer))]
        
        #Display the ice slabs points that are inside this buffer
        ax2.scatter(subset_iceslabs_retained_buffered['lon_3413'],subset_iceslabs_retained_buffered['lat_3413'],color='green',s=10)
        
        #Display the slab thickness distribution
        ax3.hist(subset_iceslabs_retained_buffered['20m_ice_content_m'])
        print(indiv_year)
        plt.show()
        pdb.set_trace()
        
        '''
        #OR EXTRACT ELEVATION OF Ys, AND EXCLUDE ALL SLABS WHOSE ELEVATION IS LARGER THAN THIS ELEVATION +/ A WINDOW

        #For this particular year, extract the distribution of ice slabs at the Emax and Ys location        
        #This is from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
        SlabsLocTree=np.transpose(np.asarray([np.asarray(subset_iceslabs_retained['lon_3413']),np.asarray(subset_iceslabs_retained['lat_3413'])]))
        tree = BallTree(SlabsLocTree, leaf_size=2)  
        # indices of neighbors within distance r
        ind = tree.query_radius(Ys_point, r=5000)         
        #Display the ice slabs points that are inside this radius
        ax2.scatter(subset_iceslabs_retained.iloc[ind[0]]['lon_3413'],subset_iceslabs_retained.iloc[ind[0]]['lat_3413'],color='green',s=10)
        
        #Display the slab thickness distribution
        ax3.hist(subset_iceslabs_retained.iloc[ind[0]]['20m_ice_content_m'])
        #continue
        plt.show()
        pdb.set_trace()
        '''
        #if vari > than x, add a label in df_2010_2018_csv to indicate not to use it, or just discard it
        # - do yearly maps!
    
    #Create a dataset of iceslabs, Emax and Ys for this stripe
    


#1. Select flowlines



#2. Pick up all perpendicular to contours transects

#3. Extract Emax and Ys for each year in each polygone

#4. Extract corresponding ice slab thickness