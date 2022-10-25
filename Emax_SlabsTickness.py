# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:00:42 2022

@author: JullienN
"""


def plot_histo(ax_plot,iceslabs_above,iceslabs_within,iceslabs_inbetween,region):
    
    if (region == 'GrIS'):
        ax_plot.hist(iceslabs_above['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_inbetween['20m_ice_content_m'],color='yellow',label='In Between',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.quantile(iceslabs_above['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.quantile(iceslabs_above['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.quantile(iceslabs_within['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.quantile(iceslabs_within['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.quantile(iceslabs_inbetween['20m_ice_content_m'],0.5),linestyle='--',color='yellow')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.quantile(iceslabs_inbetween['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='yellow')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        
    else:
        ax_plot.hist(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_inbetween[iceslabs_inbetween['key_shp']==region]['20m_ice_content_m'],color='yellow',label='In Between',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.quantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.quantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.quantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.quantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.quantile(iceslabs_inbetween[iceslabs_inbetween['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='yellow')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.quantile(iceslabs_inbetween[iceslabs_inbetween['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='yellow')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    
    #Set x lims
    ax_plot.set_xlim(-0.5,20)

    if (region == 'NW'):
        ax_plot.legend()
    
    if (region in list(['NO','NE','GrIS'])):
        ax_plot.yaxis.tick_right()#This is from Fig4andS6andS7.py from paper 'Greenland Ice Slabs Expansion and Thickening'

    return

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
from scipy import spatial
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys
from shapely.geometry import LineString
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.geometry import CAP_STYLE, JOIN_STYLE
import rioxarray as rxr

#Type of slabs product
type_slabs='high' #can be high or low

#Define which year to plot
desired_year=2019

### -------------------------- Load GrIS DEM ----------------------------- ###
#This is from paper Greenland Ice Sheet Ice Slab Expansion and Thickening, function 'extract_elevation.py'
#https://towardsdatascience.com/reading-and-visualizing-geotiff-images-with-python-8dcca7a74510
import rasterio
from rasterio.plot import show
path_GrIS_DEM = r'C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/elevations/greenland_dem_mosaic_100m_v3.0.tif'
GrIS_DEM = rasterio.open(path_GrIS_DEM)
### -------------------------- Load GrIS DEM ----------------------------- ###

#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84

path_NDWI='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/NDWI/'
#Load NDWI data for display
NDWI_image = rxr.open_rasterio(path_NDWI+'NDWI_p10_'+str(desired_year)+'.vrt',
                              masked=True).squeeze() #No need to reproject satelite image
#Extract x and y coordinates of satellite image
x_coord_NDWI=np.asarray(NDWI_image.x)
y_coord_NDWI=np.asarray(NDWI_image.y)

#Define path flowlines
path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'InBetween': "#fee391"}

'''
#Open flowlignes
polygons_Machguth2022=gpd.read_file(path_data+'polygons_Machguth2022/Ys_polygons_v3.2b.shp')

#After manual identification on QGIS, we do not need 0-79, 82-84, 121-124, 146-154, 207-208, 212-217, 223-230, 232, 242-244, 258-267, 277-282, 289-303, 318-333
nogo_polygon=np.concatenate((np.arange(0,79+1),np.arange(82,84+1),np.arange(121,124+1),np.arange(146,154+1),np.arange(207,208+1),
                             np.arange(212,217+1),np.arange(223,230+1),np.arange(232,233),np.arange(242,244+1),np.arange(258,267+1),
                             np.arange(277,282+1),np.arange(289,303+1),np.arange(318,333+1)))
'''
#Open Boxes from Tedstone and Machguth (2022)
Boxes_Tedstone2022=gpd.read_file(path_data+'Boxes_Tedstone2022/boxes.shp')

#Sort Boxes_Tedstone2022 as a function of FID
Boxes_Tedstone2022=Boxes_Tedstone2022.sort_values(by=['FID'],ascending=True)#from https://sparkbyexamples.com/pandas/pandas-sort-dataframe-by-multiple-columns/

#After manual identification on QGIS, we do not need 1-4, 20, 27, 33, 39-40, 42, 44-53
nogo_polygon=np.concatenate((np.arange(1,4+1),np.arange(20,20+1),np.arange(27,27+1),
                             np.arange(33,33+1),np.arange(39,40+1),np.arange(42,42+1),
                             np.arange(44,53+1)))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

### ---------------------------- Load dataset ---------------------------- ###
#Dictionnaries have already been created, load them
path_df_with_elevation='C:/Users/jullienn/switchdrive/Private/research/RT3/export_RT1_for_RT3/' 

#Load 2010-2018
f_20102018 = open(path_df_with_elevation+'df_20102018_with_elevation_for_RT3_rignotetalregions', "rb")
df_2010_2018 = pickle.load(f_20102018)
f_20102018.close()

#Load 2002-2003 dataset
path_2002_2003='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/'
df_2002_2003=pd.read_csv(path_2002_2003+'2002_2003_green_excel.csv')
### ---------------------------- Load dataset ---------------------------- ###

#Load max Ys from Machguth et al., (2022)
table_complete_annual_max_Ys=pd.read_csv(path_data+'_table_complete_annual_max_Ys.csv',delimiter=';',decimal=',')

#Load Emax from Tedstone and Machguth (2022)
'''
Emax_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd.csv',delimiter=',',decimal='.')
'''
Emax_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_NDWI_cleaned_2012_16_19.csv',delimiter=',',decimal='.')

#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})
#Emax_TedMach.drop(columns=['Unnamed: 0'])

Emax_plus_Mad_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_plus_mad.csv',delimiter=',',decimal='.')
Emax_Ted_minus_Mad_Mach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_minus_mad.csv',delimiter=',',decimal='.')

'''
### -------------------------- Load shapefiles --------------------------- ###
#Load Rignot et al., 2016 Greenland drainage bassins
path_rignotetal2016_GrIS_drainage_bassins='C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp',rows=slice(51,57,1)) #the regions are the last rows of the shapefile

#Extract indiv regions and create related indiv shapefiles
NW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NW']
CW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='CW']
SW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SW']
NO_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NO']
NE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NE']
SE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SE']
### -------------------------- Load shapefiles --------------------------- ###

#Plot to check
fig = plt.figure(figsize=(10,5))
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)

#Display GrIS drainage bassins
NO_rignotetal.plot(ax=ax1,color='white', edgecolor='black',linewidth=0.5)
NE_rignotetal.plot(ax=ax1,color='white', edgecolor='black',linewidth=0.5) 
SE_rignotetal.plot(ax=ax1,color='white', edgecolor='black',linewidth=0.5) 
SW_rignotetal.plot(ax=ax1,color='white', edgecolor='black',linewidth=0.5) 
CW_rignotetal.plot(ax=ax1,color='white', edgecolor='black',linewidth=0.5) 
NW_rignotetal.plot(ax=ax1,color='white', edgecolor='black',linewidth=0.5)

#Display region name 
ax1.text(NO_rignotetal.centroid.x,NO_rignotetal.centroid.y-20000,np.asarray(NO_rignotetal.SUBREGION1)[0])
ax1.text(NE_rignotetal.centroid.x,NE_rignotetal.centroid.y+20000,np.asarray(NE_rignotetal.SUBREGION1)[0])
ax1.text(SE_rignotetal.centroid.x,SE_rignotetal.centroid.y+60000,np.asarray(SE_rignotetal.SUBREGION1)[0])
ax1.text(SW_rignotetal.centroid.x,SW_rignotetal.centroid.y-120000,np.asarray(SW_rignotetal.SUBREGION1)[0])
ax1.text(CW_rignotetal.centroid.x,CW_rignotetal.centroid.y+20000,np.asarray(CW_rignotetal.SUBREGION1)[0])
ax1.text(NW_rignotetal.centroid.x,NW_rignotetal.centroid.y+20000,np.asarray(NW_rignotetal.SUBREGION1)[0])

Boxes_Tedstone2022.plot(ax=ax1,color='red', edgecolor='black',linewidth=0.5)
ax1.scatter(df_2010_2018['lon_3413'],df_2010_2018['lat_3413'],c=df_2010_2018['20m_ice_content_m'],s=0.1)
ax1.scatter(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y'],c=table_complete_annual_max_Ys['year'],s=10,cmap='magma')
ax1.scatter(Emax_TedMach['x'],Emax_TedMach['y'],c=Emax_TedMach['year'],s=5,cmap='magma')

plt.show()

pdb.set_trace()
#Save context map figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/context_map_tedstone.png',dpi=500)
'''

#Define df_2002_2003, df_2010_2018, Emax_TedMach, table_complete_annual_max_Ys as being a geopandas dataframes
points_2002_2003 = gpd.GeoDataFrame(df_2002_2003, geometry = gpd.points_from_xy(df_2002_2003['lon'],df_2002_2003['lat']),crs="EPSG:3413")
points_ice = gpd.GeoDataFrame(df_2010_2018, geometry = gpd.points_from_xy(df_2010_2018['lon_3413'],df_2010_2018['lat_3413']),crs="EPSG:3413")
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")
points_Ys = gpd.GeoDataFrame(table_complete_annual_max_Ys, geometry = gpd.points_from_xy(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y']),crs="EPSG:3413")

#Add a unique_ID_Emax column in points_ice to flag data with its corresponding Emax point
points_ice['unique_ID_Emax']=[np.nan]*len(points_ice)
#Add a selected_data column in points_ice to flag data which has already been taken in the above category
points_ice['selected_data']=[0]*len(points_ice)

#Define empty dataframe
iceslabs_above_selected_overall=pd.DataFrame()
iceslabs_selected_overall=pd.DataFrame()
iceslabs_inbetween_overall=pd.DataFrame()

#Define radius
radius=250

for indiv_index in Boxes_Tedstone2022.FID:
    
    if (indiv_index in nogo_polygon):
        #Zone excluded form proicessing, continue
        print(indiv_index,' excluded, continue')
        continue
    '''
    if (indiv_index < 32):
        continue
    '''
    print(indiv_index)
        
    #Prepare plot
    fig = plt.figure()
    fig.set_size_inches(14, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 10)
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    ax2 = plt.subplot(gs[0:10, 0:6],projection=crs)
    ax3 = plt.subplot(gs[0:5, 7:10])
    ax4 = plt.subplot(gs[6:10, 7:10],projection=crs)
    
    '''
    #Maximize plot size - This is from Fig1.py from Grenland ice slabs expansion and thickening paper.
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    '''
    #Extract individual polygon
    indiv_polygon=Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_index]

    #Display polygon
    '''
    indiv_polygon.plot(ax=ax1,color='orange', edgecolor='black',linewidth=0.5)
    '''
    indiv_polygon.plot(ax=ax2,color='none', edgecolor='black',linewidth=0.5,zorder=1)
    
    indiv_polygon.plot(ax=ax4,color='#faf6c8', edgecolor='black',linewidth=0.5)
    ax4.set_xlim(-667470, 738665)
    ax4.set_ylim(-3365680, -666380)
    #Display coastlines
    ax4.coastlines(edgecolor='black',linewidth=0.75)
    
    #Intersection between 2002-2003 ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_20022003 = gpd.sjoin(points_2002_2003, indiv_polygon, op='within')
    #Intersection between ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_ice = gpd.sjoin(points_ice, indiv_polygon, op='within')
    #Intersection between Emax and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Emax = gpd.sjoin(points_Emax, indiv_polygon, op='within')
    #Intersection between Ys and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Ys = gpd.sjoin(points_Ys, indiv_polygon, op='within')
    
    #rename colnames from join procedure to allow joining with Emax polygons
    within_points_ice=within_points_ice.rename(columns={"index_right":"index_right_polygon"})
    
    '''
    #plot
    ax1.scatter(within_points_20022003['lon'],within_points_20022003['lat'],c='#bdbdbd',s=0.1)
    ax1.scatter(within_points_Emax['x'],within_points_Emax['y'],c=within_points_Emax['year'],s=5,cmap='magma')
    ax1.scatter(within_points_ice['lon_3413'],within_points_ice['lat_3413'],c=within_points_ice['20m_ice_content_m'],s=0.1)
    ax1.scatter(within_points_Ys['X'],within_points_Ys['Y'],c=within_points_Ys['year'],s=10,cmap='magma')
    '''
        
    #Display antecedent ice slabs
    ax2.scatter(within_points_20022003['lon'],within_points_20022003['lat'],color='#bdbdbd',s=1)
    
    for indiv_year in list([desired_year]):#,2012,2016,2019]): #list([2010,2011,2012,2013,2014,2016,2017,2018]):#np.asarray(within_points_Ys.year):
        
        #Define empty dataframe
        subset_iceslabs_selected=pd.DataFrame()
        subset_iceslabs_above_selected=pd.DataFrame(columns=list(points_ice.keys()))

        #Select data of the desired year
        Emax_points=within_points_Emax[within_points_Emax.year==indiv_year]
        
        #We need at least 2 points per polygon to create an Emax line
        if (len(Emax_points)<2):
            continue
        
        #Add a column to add the elevation pickup on WGS84
        Emax_points['elevation_WGS84']=[np.nan]*len(Emax_points)
                
        #Define bounds of Emaxs in this box
        x_min=np.min(Emax_points['x'])-5e4
        x_max=np.max(Emax_points['x'])+5e4
        y_min=np.min(Emax_points['y'])-5e4
        y_max=np.max(Emax_points['y'])+5e4

        #Extract coordinates of NDWI image within Emaxs bounds
        logical_x_coord_within_bounds=np.logical_and(x_coord_NDWI>=x_min,x_coord_NDWI<=x_max)
        x_coord_within_bounds=x_coord_NDWI[logical_x_coord_within_bounds]
        logical_y_coord_within_bounds=np.logical_and(y_coord_NDWI>=y_min,y_coord_NDWI<=y_max)
        y_coord_within_bounds=y_coord_NDWI[logical_y_coord_within_bounds]

        #Define extents based on the bounds
        extent_NDWI = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
        #Display NDWI image
        cbar=ax2.imshow(NDWI_image[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_NDWI, transform=crs, origin='upper', cmap='Blues',zorder=0,vmin=0,vmax=0.3) #NDWI
        
        #plot all the Emax points of the considered indiv_year
        ax2.scatter(Emax_points['x'],Emax_points['y'],color='black',s=1,zorder=6)
        
        #Define the yearly Ys point
        Ys_point=np.transpose(np.asarray([np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['X']),np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['Y'])]))   
        
        #Display the Ys of the current indiv_year
        if (len(Ys_point>0)):
            #There is an Ys of that year for this polygon, plot it
            ax2.scatter(Ys_point[0][0],Ys_point[0][1],color='magenta',s=10,zorder=10)
        
        #Select ice slabs thickness to display distribution
        if (indiv_year == 2002):
            #Select ice slabs data from 2002            
            subset_iceslabs=within_points_20022003[within_points_20022003.year==2002]
            #Rename lat and lon columns to match existing routine
            subset_iceslabs=subset_iceslabs.rename(columns={"lat": "lat_3413","lon": "lon_3413"})
        elif (indiv_year == 2003):
            #Select ice slabs data from 2002 and 2003
            subset_iceslabs=within_points_20022003
            #Rename lat and lon columns to match existing routine
            subset_iceslabs=subset_iceslabs.rename(columns={"lat": "lat_3413","lon": "lon_3413"})
        elif (indiv_year == 2011):
            #Select ice slabs data from 2010 and 2011
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,within_points_ice.year==2011)]
        elif(indiv_year == 2012):
            #Select ice slabs data from 2010, 2011, 2012
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,(within_points_ice.year==2011)|(within_points_ice.year==2012))]
            '''
            #Select ice slabs data from 2011, 2012
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,within_points_ice.year==2011)]
            '''
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
            '''
            #Select ice slabs data of the closest indiv_year, i.e. 2014
            subset_iceslabs=within_points_ice[within_points_ice.year==2014]
            '''
        elif (indiv_year == 2017):
            #Select ice slabs data from 2017, 2014, 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2013,(within_points_ice.year==2014)|(within_points_ice.year==2017))]
        elif (indiv_year == 2018):
            #Select ice slabs data from 2018, 2017, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,(within_points_ice.year==2017)|(within_points_ice.year==2018))]
        elif (indiv_year == 2019):
            #Select ice slabs data from 2018, 2017, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,(within_points_ice.year==2017)|(within_points_ice.year==2018))]
            '''
            #Select ice slabs data from 2018, 2017
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2017,within_points_ice.year==2018)]
            '''
        else:
            #Select ice slabs data of the current indiv_year
            subset_iceslabs=within_points_ice[within_points_ice.year==indiv_year]
                
        if (len(subset_iceslabs)==0):
            #No slab for this particular year, continue
            continue
                
        #Display antecedent ice slabs
        ax2.scatter(within_points_ice[within_points_ice.year<=indiv_year]['lon_3413'],within_points_ice[within_points_ice.year<=indiv_year]['lat_3413'],color='gray',s=1,zorder=1)
        '''
        #Display the tracks of the current year within the polygon
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],color='purple',s=40,zorder=4)
        '''
        ######################### Connect Emax points #########################
        
        #Keep only Emax points whose box_id is associated with the current box_id
        Emax_points=Emax_points[Emax_points.box_id==indiv_index]
        
        '''
        #Display start and end of Emax points for line definition
        ax2.scatter(Emax_points['x'].iloc[0],Emax_points['y'].iloc[0],color='green',s=40,zorder=7)
        ax2.scatter(Emax_points['x'].iloc[-1],Emax_points['y'].iloc[-1],color='red',s=40,zorder=7)
        '''
        
        #Emax as tuples
        Emax_tuple=[tuple(row[['x','y']]) for index, row in Emax_points.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        #Connect Emax points between them
        lineEmax= LineString(Emax_tuple) #from https://shapely.readthedocs.io/en/stable/manual.html
        #Display Emax line
        ax2.plot(lineEmax.xy[0],lineEmax.xy[1],zorder=5,color='#a50f15',linewidth=0.5) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ######################### Connect Emax points #########################

        ########################### Polygon within ############################
        #Create a buffer around this line
        buffer_within_Emax = lineEmax.buffer(radius, cap_style=1) #from https://shapely.readthedocs.io/en/stable/code/buffer.py
        #Create polygon patch from this buffer
        plot_buffer_within_Emax = PolygonPatch(buffer_within_Emax,zorder=2,color='red',alpha=0.2)
        #Display patch
        ax2.add_patch(plot_buffer_within_Emax)        
        #Convert polygon of Emax buffer around connected Emax line into a geopandas dataframe
        Emax_within_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[buffer_within_Emax]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between subset_iceslabs and Emax_polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
        Intersection_EmaxBuffer_slabs = gpd.sjoin(subset_iceslabs, Emax_within_polygon, op='within')
        #Plot the result of this selection
        ax2.scatter(Intersection_EmaxBuffer_slabs['lon_3413'],Intersection_EmaxBuffer_slabs['lat_3413'],color='red',s=1,zorder=8)
        ########################### Polygon within ############################
                
        ################################ Above ################################
        #Define a line for the above upper boundary 4000m away from Emax line        
        if ((indiv_index==7) & (indiv_year==2016)):
            lineEmax_upper_start_pre = lineEmax.parallel_offset(2000, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            lineEmax_upper_start = lineEmax_upper_start_pre.parallel_offset(2000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            #We choose 10km, should we choose another value??
            lineEmax_upper_end_b = lineEmax_upper_start.parallel_offset(6000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
        else:
            lineEmax_upper_start = lineEmax.parallel_offset(4000, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            #We choose 10km, should we choose another value??
            lineEmax_upper_end_a = lineEmax.parallel_offset(5000, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            lineEmax_upper_end_b = lineEmax_upper_end_a.parallel_offset(5000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            
        #Plot the above upper boundaries        
        ax2.plot(lineEmax_upper_start.xy[0],lineEmax_upper_start.xy[1],zorder=5,color='#045a8d') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        #ax2.plot(lineEmax_upper_end_a.xy[0],lineEmax_upper_end_a.xy[1],zorder=5,color='red') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ax2.plot(lineEmax_upper_end_b.xy[0],lineEmax_upper_end_b.xy[1],zorder=5,color='#045a8d') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        
        #Create a polygon with low end begin the Emax line and upper end being the Emax line + 20000
        polygon_above=Polygon([*list(lineEmax_upper_end_b.coords),*list(lineEmax_upper_start.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
        #Create polygon patch of the polygon above
        plot_buffer_above_Emax = PolygonPatch(polygon_above,zorder=2,color='blue',alpha=0.2)
        #Display patch of polygone above
        #ax2.add_patch(plot_buffer_above_Emax)        
        #Convert polygon of Emax buffer above into a geopandas dataframe
        Emax_above_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_above]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between subset_iceslabs and Emax_above_polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        Intersection_EmaxBufferAbove_slabs = gpd.sjoin(subset_iceslabs, Emax_above_polygon, op='within')
        #Plot the result of this selection
        ax2.scatter(Intersection_EmaxBufferAbove_slabs['lon_3413'],Intersection_EmaxBufferAbove_slabs['lat_3413'],color='blue',s=1,zorder=8)
        ################################ Above ################################

        #Plot ice slabs thickness that are above and within Emax polygons
        ax3.hist(Intersection_EmaxBufferAbove_slabs['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax3.hist(Intersection_EmaxBuffer_slabs['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax3.set_xlabel('Ice content [m]')
        ax3.set_ylabel('Density [ ]')
        ax3.set_xlim(0,20)

        fig.suptitle('Box '+str(indiv_index)+ ' - '+str(indiv_year)+' - 3 years running slabs - radius '+str(radius)+' m - cleanedxytpd')
        plt.show()
        
        ############################## In between ##############################
        lineEmax_radius = lineEmax.parallel_offset(radius, 'right', join_style=1)
        #ax2.plot(lineEmax_radius .xy[0],lineEmax_radius .xy[1],zorder=5,color='yellow')
        ax2.plot(lineEmax_upper_start .xy[0],lineEmax_upper_start .xy[1],zorder=5,color='yellow')
        
        polygon_radius_4000=Polygon([*list(lineEmax_upper_start.coords),*list(lineEmax_radius.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
        plot_buffer_radius_4000 = PolygonPatch(polygon_radius_4000,zorder=2,color='yellow',alpha=0.2)
        #ax2.add_patch(plot_buffer_radius_4000)        
             
        #Convert polygon of polygon_radius_4000 into a geopandas dataframe
        Emax_radius_4000_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_radius_4000]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between subset_iceslabs and Emax_radius_4000_polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        Intersection_Emaxradius4000_slabs = gpd.sjoin(subset_iceslabs, Emax_radius_4000_polygon, op='within')
        #Plot the result of this selection
        ax2.scatter(Intersection_Emaxradius4000_slabs['lon_3413'],Intersection_Emaxradius4000_slabs['lat_3413'],color='yellow',s=1,zorder=7)
        '''
        ax3.hist(Intersection_Emaxradius4000_slabs['20m_ice_content_m'],color='yellow',label='In-between',alpha=0.5,bins=np.arange(0,17),density=True)
        '''
        ############################## In between ##############################
        ax3.legend()
        
        #Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
        legend_elements = [Line2D([0], [0], color='#bdbdbd', lw=2, label='2002-03 ice slabs'),
                           Line2D([0], [0], color='gray', lw=2, label='2010-18 ice slabs'),
                           Line2D([0], [0], color='purple', lw=2, label='Considered ice slabs (3 years)'),
                           Line2D([0], [0], color='black', lw=2, label='Emax retrieval', marker='o',linestyle='None'),
                           Line2D([0], [0], color='#a50f15', lw=2, label='Connected Emax retrieval'),
                           Patch(facecolor='red',label='Buffer around Emax'),
                           Patch(facecolor='blue',label='Area above Emax buffer'),
                           Line2D([0], [0], color='red', lw=2, label='Ice slabs within Emax buffer'),
                           Line2D([0], [0], color='blue', lw=2, label='Ice slabs above Emax buffer'),
                           Line2D([0], [0], color='magenta', lw=2, label='Ys', marker='o',linestyle='None'),
                           Patch(facecolor='yellow',label='Area in-between'),
                           Line2D([0], [0], color='yellow', lw=2, label='Ice slabs in-between')]
        
        '''
        ax2.legend(handles=legend_elements)
        plt.legend()
        '''
        
        #Set limits
        if (len(Intersection_EmaxBufferAbove_slabs)>0):
            ax2.set_xlim(np.min(Emax_points['x'])-1e4,
                         np.max(Emax_points['x'])+1e4)
            ax2.set_ylim(np.min(Emax_points['y'])-1e4,
                         np.max(Emax_points['y'])+1e4)
        
        #Save the iceslabs within and above of that polygon into another dataframe for overall plot
        iceslabs_above_selected_overall=pd.concat([iceslabs_above_selected_overall,Intersection_EmaxBufferAbove_slabs])
        iceslabs_selected_overall=pd.concat([iceslabs_selected_overall,Intersection_EmaxBuffer_slabs])
        iceslabs_inbetween_overall=pd.concat([iceslabs_inbetween_overall,Intersection_Emaxradius4000_slabs])
        
        #Save the figure
        plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/'+str(indiv_year)+'/Emax_VS_IceSlabs_'+str(indiv_year)+'_Box'+str(indiv_index)+'_3YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpd_with0mslabs.png',dpi=500,bbox_inches='tight')
        #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        
        plt.close()


##### TRY DISPLAY THE DISTRIBUTION OF THE LOW END - TAKE INTO ACCOUNT THE LIKELIHOOD?
pdb.set_trace()

#Display ice slabs distributions as a function of the regions
#Prepare plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(15, 10)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
axNW = plt.subplot(gs[0:5, 0:5])
axCW = plt.subplot(gs[5:10, 0:5])
axSW = plt.subplot(gs[10:15, 0:5])
axNO = plt.subplot(gs[0:5, 5:10])
axNE = plt.subplot(gs[5:10, 5:10])
axGrIS = plt.subplot(gs[10:15, 5:10])

#Plot histograms
plot_histo(axNW,iceslabs_above_selected_overall,iceslabs_selected_overall,iceslabs_inbetween_overall,'NW')
plot_histo(axCW,iceslabs_above_selected_overall,iceslabs_selected_overall,iceslabs_inbetween_overall,'CW')
plot_histo(axSW,iceslabs_above_selected_overall,iceslabs_selected_overall,iceslabs_inbetween_overall,'SW')
plot_histo(axNO,iceslabs_above_selected_overall,iceslabs_selected_overall,iceslabs_inbetween_overall,'NO')
plot_histo(axNE,iceslabs_above_selected_overall,iceslabs_selected_overall,iceslabs_inbetween_overall,'NE')
plot_histo(axGrIS,iceslabs_above_selected_overall,iceslabs_selected_overall,iceslabs_inbetween_overall,'GrIS')

#Finalise plot
axSW.set_xlabel('Ice content [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('Overall - '+str(indiv_year)+' - 3 years running slabs')
plt.show()
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/'+str(indiv_year)+'/Histo_Emax_VS_IceSlabs_'+str(indiv_year)+'_Box_Tedstone_3YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpd_with0mslabs.png',dpi=500)


#Display ice slabs distributions as a function of the regions without 0m thick ice slabs
#Prepare plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(15, 10)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
axNW = plt.subplot(gs[0:5, 0:5])
axCW = plt.subplot(gs[5:10, 0:5])
axSW = plt.subplot(gs[10:15, 0:5])
axNO = plt.subplot(gs[0:5, 5:10])
axNE = plt.subplot(gs[5:10, 5:10])
axGrIS = plt.subplot(gs[10:15, 5:10])

#Plot histograms
plot_histo(axNW,
           iceslabs_above_selected_overall[iceslabs_above_selected_overall['20m_ice_content_m']>0],
           iceslabs_selected_overall[iceslabs_selected_overall['20m_ice_content_m']>0],
           iceslabs_inbetween_overall[iceslabs_inbetween_overall['20m_ice_content_m']>0],
           'NW')
plot_histo(axCW,
           iceslabs_above_selected_overall[iceslabs_above_selected_overall['20m_ice_content_m']>0],
           iceslabs_selected_overall[iceslabs_selected_overall['20m_ice_content_m']>0],
           iceslabs_inbetween_overall[iceslabs_inbetween_overall['20m_ice_content_m']>0],
           'CW')
plot_histo(axSW,
           iceslabs_above_selected_overall[iceslabs_above_selected_overall['20m_ice_content_m']>0],
           iceslabs_selected_overall[iceslabs_selected_overall['20m_ice_content_m']>0],
           iceslabs_inbetween_overall[iceslabs_inbetween_overall['20m_ice_content_m']>0],
           'SW')
plot_histo(axNO,
           iceslabs_above_selected_overall[iceslabs_above_selected_overall['20m_ice_content_m']>0],
           iceslabs_selected_overall[iceslabs_selected_overall['20m_ice_content_m']>0],
           iceslabs_inbetween_overall[iceslabs_inbetween_overall['20m_ice_content_m']>0],
           'NO')
plot_histo(axNE,
           iceslabs_above_selected_overall[iceslabs_above_selected_overall['20m_ice_content_m']>0],
           iceslabs_selected_overall[iceslabs_selected_overall['20m_ice_content_m']>0],
           iceslabs_inbetween_overall[iceslabs_inbetween_overall['20m_ice_content_m']>0],
           'NE')
plot_histo(axGrIS,
           iceslabs_above_selected_overall[iceslabs_above_selected_overall['20m_ice_content_m']>0],
           iceslabs_selected_overall[iceslabs_selected_overall['20m_ice_content_m']>0],
           iceslabs_inbetween_overall[iceslabs_inbetween_overall['20m_ice_content_m']>0],
           'GrIS')

#Finalise plot
axSW.set_xlabel('Ice content [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('Overall - '+str(indiv_year)+' - 3 years running slabs - 0m thick slabs excluded')
plt.show()
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/'+str(indiv_year)+'/HistoNonZeros_Emax_VS_IceSlabs_'+str(indiv_year)+'_Box_Tedstone_3YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpd_with0mslabs.png',dpi=500)

#Display as boxplots
#Aggregate data together
iceslabs_above_selected_overall['type']=['Above']*len(iceslabs_above_selected_overall)
iceslabs_selected_overall['type']=['Within']*len(iceslabs_selected_overall)
iceslabs_inbetween_overall['type']=['InBetween']*len(iceslabs_inbetween_overall)
iceslabs_boxplot=pd.concat([iceslabs_above_selected_overall,iceslabs_inbetween_overall,iceslabs_selected_overall])
#Get rid of out region
iceslabs_boxplot=iceslabs_boxplot[iceslabs_boxplot.key_shp!='Out']

iceslabs_boxplot_GrIS=iceslabs_boxplot.copy(deep=True)
iceslabs_boxplot_GrIS['key_shp']=['GrIS']*len(iceslabs_boxplot_GrIS)
iceslabs_boxplot_region_GrIS=pd.concat([iceslabs_boxplot,iceslabs_boxplot_GrIS])

#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.boxplot(data=iceslabs_boxplot_region_GrIS, x="20m_ice_content_m", y="key_shp",hue="type",orient="h",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_ylabel('')
ax_regions_GrIS.set_xlabel('Ice content [m]')
ax_regions_GrIS.set_xlim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle(str(indiv_year)+' - 3 years running slabs')

#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/'+str(indiv_year)+'/Boxplot_Emax_VS_IceSlabs_'+str(indiv_year)+'_Box_Tedstone_3YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpd_with0mslabs.png',dpi=500)



'''

if (indiv_year in list([2002,2003])):
    #Display the ice slabs points that are inside this buffer
    ax2.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
else:
    #Store an empty dataframe with the index so that index is displayed in plot even without data 
    if (len(subset_iceslabs_buffered)==0):
        #No slab for this particular year at these elevations
        subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,indiv_index, np.nan]]),columns=subset_iceslabs_buffered.columns.values)],ignore_index=True)# from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html and https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/
        #From https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean and https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work
        print(str(indiv_index)+' has no data')
        continue
    
    #Display the ice slabs points that are inside this buffer
    ax2.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
    
    #Store subset_iceslabs_buffered 
    subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,subset_iceslabs_buffered],ignore_index=True)
    #From https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean and https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work

plt.show()
'''
