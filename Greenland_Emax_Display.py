# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:47:05 2022

@author: JullienN

This code is a modified version of the code 'Emax_SlabsTickness.py',
copied on Novermber 3rd 2022 at 19h45
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
from scipy import spatial
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys
from shapely.geometry import LineString
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.geometry import CAP_STYLE, JOIN_STYLE
import rioxarray as rxr

#Define which year to plot
desired_year=2019
#Define which map to plot on the background: either NDWI or master_map
desired_map='NDWI'

#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84

if (desired_map=='NDWI'):
    path_NDWI='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/NDWI/'
    #Load NDWI data for display
    MapPlot = rxr.open_rasterio(path_NDWI+'NDWI_p10_'+str(desired_year)+'.vrt',
                                  masked=True).squeeze() #No need to reproject satelite image
    vlim_min=0
    vlim_max=0.3
    
elif (desired_map=='master_map'):
    #Load hydrological master map from Tedstone and Machuguth (2022)
    path_CumHydroMap='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/master_maps/'
    #Load master_maps data for display
    MapPlot = rxr.open_rasterio(path_CumHydroMap+'master_map_GrIS_mean.vrt',
                                  masked=True).squeeze() #No need to reproject satelite image
    vlim_min=0
    vlim_max=150
    
else:
    print('Enter a correct map name!')
    pdb.set_trace()

#Extract x and y coordinates of satellite image
x_coord_MapPlot=np.asarray(MapPlot.x)
y_coord_MapPlot=np.asarray(MapPlot.y)

#Define path Boxes
path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'InBetween': "#fee391"}

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

#Load Emax from Tedstone and Machguth (2022)
'''
Emax_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd.csv',delimiter=',',decimal='.')
'''
Emax_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_NDWI_cleaned_2012_16_19_v2.csv',delimiter=',',decimal='.')

#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})
#Emax_TedMach.drop(columns=['Unnamed: 0'])

### ------------------------- Load df_2010_2018 --------------------------- ###
path_df_with_elevation='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/' 

#Load 2010-2018 high estimate
f_20102018_high = open(path_df_with_elevation+'final_excel/high_estimate/df_20102018_with_elevation_high_estimate_rignotetalregions', "rb")
df_2010_2018_high = pickle.load(f_20102018_high)
f_20102018_high.close

### ------------------------- Load df_2010_2018 --------------------------- ###

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

#Define Emax_TedMach as being a geopandas dataframe
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")

#Keep data of the desired year
Emax_points_yearly=points_Emax[points_Emax.year==desired_year]

#Prepare plot
fig = plt.figure()
fig.set_size_inches(14, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
gs = gridspec.GridSpec(14, 10)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(gs[0:14, 0:10],projection=crs)
#Display coastlines
ax1.coastlines(edgecolor='black',linewidth=0.075)
#Display 2010-2018 ice slabs data
ax1.scatter(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413'],c='#9ecae1',s=0.1,zorder=2)

for indiv_index in Boxes_Tedstone2022.FID:
    '''
    if (indiv_index in nogo_polygon):
        #Zone excluded form processing, continue
        print(indiv_index,' excluded, continue')
        continue
    '''
    print(indiv_index)
    
    #Extract individual polygon
    indiv_polygon=Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_index]
    '''
    #Display polygon
    if (indiv_index==7):
        indiv_polygon.plot(ax=ax1,color='orange', edgecolor='black',linewidth=0.05,zorder=1)
    else:
        indiv_polygon.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.05,zorder=1)
    '''
    indiv_polygon.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.05,zorder=1)

    
    #Intersection between Emax and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    Emax_points = gpd.sjoin(Emax_points_yearly, indiv_polygon, op='within')
    
    #We need at least 2 points per polygon to create an Emax line
    if (len(Emax_points)<2):
        continue
    
    #Define bounds of Emaxs in this box
    x_min=np.min(Emax_points['x'])-5e4
    x_max=np.max(Emax_points['x'])+5e4
    y_min=np.min(Emax_points['y'])-5e4
    y_max=np.max(Emax_points['y'])+5e4

    #Extract coordinates of NDWI image within Emaxs bounds
    logical_x_coord_within_bounds=np.logical_and(x_coord_MapPlot>=x_min,x_coord_MapPlot<=x_max)
    x_coord_within_bounds=x_coord_MapPlot[logical_x_coord_within_bounds]
    logical_y_coord_within_bounds=np.logical_and(y_coord_MapPlot>=y_min,y_coord_MapPlot<=y_max)
    y_coord_within_bounds=y_coord_MapPlot[logical_y_coord_within_bounds]

    #Define extents based on the bounds
    extent_MapPlot = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
    #Display NDWI image
    #cbar=ax1.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap='Blues',zorder=0,vmin=vlim_min,vmax=vlim_max)
    
    
    #plot all the Emax points of the considered indiv_year
    ax1.scatter(Emax_points['x'],Emax_points['y'],color='black',s=0.1,zorder=3)
    
    
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
    #ax1.plot(lineEmax.xy[0],lineEmax.xy[1],zorder=5,color='#a50f15',linewidth=0.05) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ######################### Connect Emax points #########################
    
    #pdb.set_trace()

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
#legend_elements = [Line2D([0], [0], color='#a50f15', lw=2, label='Emax')]
legend_elements = [Patch(facecolor='none',edgecolor='black',label='100km-wide boxes'),
                   Line2D([0], [0], color='#9ecae1', lw=2, marker='o',linestyle='None', label='2010-2018 ice slabs'),
                   Line2D([0], [0], color='black', lw=2, marker='o',linestyle='None', label='Emax retrieval')]
ax1.legend(handles=legend_elements,loc='lower right')
plt.show()
#plt.legend()

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax1.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
#gl.ylabels_right = False
gl.xlabels_bottom = False
ax1.axis('off')
#ax8map.legend(loc='upper right')
###################### From Tedstone et al., 2022 #####################

fig.suptitle(str(desired_year)+' Emax retrievals - cleanedxytpd V2')
#fig.suptitle(str(desired_year)+' Emax and '+desired_map+' - cleanedxytpd V2')

plt.show()
pdb.set_trace()

#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/'+str(desired_year)+'_EmaxMap_'+desired_map+'cleanedxytpdV2.pdf',dpi=2000)
