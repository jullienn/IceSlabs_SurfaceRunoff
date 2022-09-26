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
import seaborn as sns

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

#Load 2002-2003 dataset
path_2002_2003='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/'
df_2002_2003=pd.read_csv(path_2002_2003+'2002_2003_green_excel.csv')

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

#Define width of the buffer for ice slabs pick up (elevation-wise, in meters)
buffer=10

#Define df_2010_2018_high, Emax_TedMach, table_complete_annual_max_Ys as being a geopandas dataframes
points_ice = gpd.GeoDataFrame(df_2010_2018_high, geometry = gpd.points_from_xy(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413']),crs="EPSG:3413")
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")
points_Ys = gpd.GeoDataFrame(table_complete_annual_max_Ys, geometry = gpd.points_from_xy(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y']),crs="EPSG:3413")

#Define an empty summary dataframe
subset_iceslabs_buffered_summary=pd.DataFrame()

#Plot to check
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(20, 6)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax2 = plt.subplot(gs[0:20, 0:3],projection=crs)
ax3 = plt.subplot(gs[0:1, 3:6])
ax4 = plt.subplot(gs[1:2, 3:6])
ax5 = plt.subplot(gs[2:3, 3:6])
ax6 = plt.subplot(gs[3:4, 3:6])
ax7 = plt.subplot(gs[4:5, 3:6])
ax8 = plt.subplot(gs[5:6, 3:6])
ax9 = plt.subplot(gs[6:7, 3:6])
ax10 = plt.subplot(gs[7:8, 3:6])
ax11 = plt.subplot(gs[8:9, 3:6])
ax12 = plt.subplot(gs[9:10, 3:6])
ax13 = plt.subplot(gs[10:11, 3:6])
ax14 = plt.subplot(gs[11:12, 3:6])
ax15 = plt.subplot(gs[12:13, 3:6])
ax16 = plt.subplot(gs[13:14, 3:6])
ax17 = plt.subplot(gs[14:15, 3:6])
ax18 = plt.subplot(gs[15:16, 3:6])
ax19 = plt.subplot(gs[16:17, 3:6])
ax20 = plt.subplot(gs[17:18, 3:6])
ax21 = plt.subplot(gs[18:19, 3:6])
ax22 = plt.subplot(gs[19:20, 3:6])

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'2010': "#1a9850", '2011': "#66bd63", '2012': "#a6d96a", '2013':"#d9ef8b", '2014':"#fee08b", '2016':"#fdae61", '2017':"#f46d43", '2018':"#d73027", '2019':"#d73027"}

for indiv_index in flowlines_polygones.index:
    
    #Define axis to plot distribution
    if (indiv_index == 19):
        ax_distrib=ax3
    elif (indiv_index == 18):
        ax_distrib=ax4
    elif (indiv_index == 17):
        ax_distrib=ax5
    elif (indiv_index == 16):
        ax_distrib=ax6
    elif (indiv_index == 15):
        ax_distrib=ax7
    elif (indiv_index == 14):
        ax_distrib=ax8
    elif (indiv_index == 13):
        ax_distrib=ax9
    elif (indiv_index == 12):
        ax_distrib=ax10
    elif (indiv_index == 11):
        ax_distrib=ax11
    elif (indiv_index == 10):
        ax_distrib=ax12
    elif (indiv_index == 9):
        ax_distrib=ax13
    elif (indiv_index == 8):
        ax_distrib=ax14
    elif (indiv_index == 7):
        ax_distrib=ax15
    elif (indiv_index == 6):
        ax_distrib=ax16
    elif (indiv_index == 5):
        ax_distrib=ax17
    elif (indiv_index == 4):
        ax_distrib=ax18
    elif (indiv_index == 3):
        ax_distrib=ax19
    elif (indiv_index == 2):
        ax_distrib=ax20
    elif (indiv_index == 1):
        ax_distrib=ax21
    elif (indiv_index == 0):
        ax_distrib=ax22
    else:
        print('Should not arrive here')
    
    print(indiv_index)
    indiv_polygon=flowlines_polygones[flowlines_polygones.index==indiv_index]

    #Display GrIS drainage bassins
    indiv_polygon.plot(ax=ax1,color='orange', edgecolor='black',linewidth=0.5)
    indiv_polygon.plot(ax=ax2,color='orange', edgecolor='black',linewidth=0.5)

    #Intersection between ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_ice = gpd.sjoin(points_ice, indiv_polygon, op='within')
    #plot
    ax1.scatter(within_points_ice['lon_3413'],within_points_ice['lat_3413'],c=within_points_ice['20m_ice_content_m'],s=0.1)
    
    '''
    #Intersection between Emax and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Emax = gpd.sjoin(points_Emax, indiv_polygon, op='within')
    #plot
    ax1.scatter(within_points_Emax['x'],within_points_Emax['y'],c=within_points_Emax['year'],s=5,cmap='magma')
    '''
    
    #Intersection between Ys and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Ys = gpd.sjoin(points_Ys, indiv_polygon, op='within')
    #plot
    ax1.scatter(within_points_Ys['X'],within_points_Ys['Y'],c=within_points_Ys['year'],s=10,cmap='magma')
    
    plt.show()
    #15h50
    
    for indiv_year in list([2019]):#,2012,2016,2019]): #list([2010,2011,2012,2013,2014,2016,2017,2018]):#np.asarray(within_points_Ys.year):
        #Define the yearly Ys point
        Ys_point=np.transpose(np.asarray([np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['X']),np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['Y'])]))   
        
        if (len(Ys_point)==0):
            continue
        
        #Extract elevation of Ys
        for val in GrIS_DEM.sample(Ys_point): 
            #Calculate the corresponding elevation
            Ys_point_elevation=val[0]
        
        if (indiv_year == 2011):
            #Select ice slabs data from 2010 and 2011
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,within_points_ice.year==2011)]
        elif(indiv_year == 2012):
            #Select ice slabs data from 2010, 2011, 2012
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,(within_points_ice.year==2011)|(within_points_ice.year==2012))]
        elif(indiv_year == 2013):
            #Select ice slabs data from 2011, 2012, 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2011,(within_points_ice.year==2012)|(within_points_ice.year==2013))]
        elif(indiv_year == 2014):
            #Select ice slabs data from 2012, 2013, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,(within_points_ice.year==2013)|(within_points_ice.year==2014))]
        elif (indiv_year == 2015):
            #Select ice slabs data of the closest indiv_year, i.e. 2014 and the 2 previous ones
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,(within_points_ice.year==2013)|(within_points_ice.year==2014))]
        elif (indiv_year == 2016):
            #Select ice slabs data of the closest indiv_year, i.e. 2014 and the 2 previous ones
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,(within_points_ice.year==2013)|(within_points_ice.year==2014))]
        elif (indiv_year == 2017):
            #Select ice slabs data from 2017, 2014, 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2013,(within_points_ice.year==2014)|(within_points_ice.year==2017))]
        elif (indiv_year == 2018):
            #Select ice slabs data from 2018, 2017, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,(within_points_ice.year==2017)|(within_points_ice.year==2018))]
        elif (indiv_year == 2019):
            #Select ice slabs data from 2018, 2017, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,(within_points_ice.year==2017)|(within_points_ice.year==2018))]
        else:
            #Select ice slabs data of the current indiv_year
            subset_iceslabs=within_points_ice[within_points_ice.year==indiv_year]
        
        if (len(subset_iceslabs)==0):
            #No slab for this particular year, continue
            continue
                
        #Display the tracks of the current year within the polygon
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],color='purple',s=10)
        
        '''
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],c=subset_iceslabs['20m_ice_content_m'],s=10)
        ax2.scatter(within_points_Emax[within_points_Emax.year==indiv_year]['x'],within_points_Emax[within_points_Emax.year==indiv_year]['y'],color='blue',s=10)
        #Display the whole track
        ax2.scatter(df_2010_2018_csv[df_2010_2018_csv.Track_name==indiv_track]['lon_3413'],df_2010_2018_csv[df_2010_2018_csv.Track_name==indiv_track]['lat_3413'],color='black',s=10)
        '''
        
        #Keep only data where elevation is within elevation+/-buffer
        subset_iceslabs_buffered=subset_iceslabs[np.logical_and(subset_iceslabs['elevation']<=(Ys_point_elevation+buffer),subset_iceslabs['elevation']>=(Ys_point_elevation-buffer))]
        
        #Display the ice slabs points that are inside this buffer
        ax2.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
        
        #Display the Ys of the current indiv_year
        ax2.scatter(Ys_point[0][0],Ys_point[0][1],color='black',s=10,zorder=1)
        
        #Display the slab thickness distribution
        ax_distrib.hist(subset_iceslabs_buffered['20m_ice_content_m'],density=True,color=my_pal[str(indiv_year)],alpha=0.5)
        
        #Display the IQR on the distribution
        if (len(subset_iceslabs_buffered)>0):
            ax_distrib.axvline(x=np.quantile(subset_iceslabs_buffered['20m_ice_content_m'],0.25),linestyle='--',color='k')
            ax_distrib.axvline(x=np.quantile(subset_iceslabs_buffered['20m_ice_content_m'],0.5),color='red')
            ax_distrib.axvline(x=np.quantile(subset_iceslabs_buffered['20m_ice_content_m'],0.75),linestyle='--',color='k')
            ax_distrib.text(0.05, 0.5,str(np.round(np.quantile(subset_iceslabs_buffered['20m_ice_content_m'],0.75),1))+'m',ha='center', va='center', transform=ax_distrib.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

        ax_distrib.set_xlim(0,16)
        
        #Store subset_iceslabs_buffered 
        subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,subset_iceslabs_buffered])
        print(indiv_year)
        fig.suptitle(str(indiv_year)+' - 3 years running slabs')
        
        plt.show()
        
        #Display IQR and median on plots
        #Display in shades of grey older iceslabs if any
        #Display 2002-2003 ice slabs!
        

    #Save the figure
    #plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Ys_VS_IceSlabs/Ys_VS_IceSlabs'+str(indiv_year)+'_3YearsRunSlabs.png',dpi=500)

    #Display the polygone number
    #ax_distrib.set_title(str(indiv_index))
    
    # - do yearly maps!
    
    #Create a dataset of iceslabs, Emax and Ys for this stripe

#sns.displot(data=subset_iceslabs_buffered_summary, x="20m_ice_content_m", col="year_Ys", kde=True)



#1. Select flowlines


#3. Extract Emax and Ys for each year in each polygone

#4. Extract corresponding ice slab thickness