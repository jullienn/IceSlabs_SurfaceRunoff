# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:11:08 2022

@author: JullienN

This code is a modified version of the code 'Greenland_Emax_Display.py',
Copied on November 10th 2022 at 17h10
Itself being the code from a modified version of the code 'Emax_SlabsTickness.py',
copied on November 3rd 2022 at 19h45
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
desired_map='master_map'

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

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'InBetween': "#fee391"}

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Define path Boxes
path_data='C:/Users/jullienn/switchdrive/Private/research/RT3/data/'
#Open Boxes from Tedstone and Machguth (2022)
Boxes_Tedstone2022=gpd.read_file(path_data+'Boxes_Tedstone2022/boxes.shp')
#Sort Boxes_Tedstone2022 as a function of FID
Boxes_Tedstone2022=Boxes_Tedstone2022.sort_values(by=['FID'],ascending=True)#from https://sparkbyexamples.com/pandas/pandas-sort-dataframe-by-multiple-columns/

#Load Emax from Tedstone and Machguth (2022)
Emax_TedMach=pd.read_csv(path_data+'Emax/xytpd_NDWI_cleaned_2012_16_19_v2.csv',delimiter=',',decimal='.')
#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})
#Define Emax_TedMach as being a geopandas dataframe
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")
#Keep data of the desired year
Emax_points_yearly=points_Emax[points_Emax.year==desired_year]

### ------------------------- Load df_2010_2018 --------------------------- ###
path_df_with_elevation='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/' 
#Load 2010-2018 high estimate
f_20102018_high = open(path_df_with_elevation+'final_excel/high_estimate/df_20102018_with_elevation_high_estimate_rignotetalregions', "rb")
df_2010_2018_high = pickle.load(f_20102018_high)
f_20102018_high.close

#Load corresponding shapefile
iceslabs_jullien_highend_20102018=gpd.read_file(path_df_with_elevation+'shapefiles/iceslabs_jullien_highend_20102018.shp') 
### ------------------------- Load df_2010_2018 --------------------------- ###

### ------------------------ Load 2002-2003 data -------------------------- ###
path_2002_2003='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/'
df_2002_2003=pd.read_csv(path_2002_2003+'2002_2003_green_excel.csv')
### ------------------------ Load 2002-2003 data -------------------------- ###

### ---------------- Load firn aquifers Miège et al., 2016 ---------------- ###
path_aquifers='C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/firn_aquifers_miege/'
df_firn_aquifer_all=pd.DataFrame()
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2010.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2011.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2012.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2013.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2014.csv',delimiter=',',decimal='.'))

#Transform miege coordinates from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
points=transformer.transform(np.asarray(df_firn_aquifer_all["LONG"]),np.asarray(df_firn_aquifer_all["LAT"]))
df_firn_aquifer_all['lon_3413']=points[0]
df_firn_aquifer_all['lat_3413']=points[1]
### ---------------- Load firn aquifers Miège et al., 2016 ---------------- ###

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

#Prepare plot
fig = plt.figure()
fig.set_size_inches(14, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)

#Display GrIS drainage bassins
NO_rignotetal.plot(ax=ax1,color='#0f4969', edgecolor='black',linewidth=0.075)
NE_rignotetal.plot(ax=ax1,color='#0f4969', edgecolor='black',linewidth=0.075) 
SE_rignotetal.plot(ax=ax1,color='#0f4969', edgecolor='black',linewidth=0.075) 
SW_rignotetal.plot(ax=ax1,color='#0f4969', edgecolor='black',linewidth=0.075) 
CW_rignotetal.plot(ax=ax1,color='#0f4969', edgecolor='black',linewidth=0.075) 
NW_rignotetal.plot(ax=ax1,color='#0f4969', edgecolor='black',linewidth=0.075)

#Open and plot runoff limit medians shapefiles
path_poly='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/runoff_limit_polys/'
poly_1985_1992=gpd.read_file(path_poly+'poly_1985_1992_median_edited.shp')
poly_1985_1992.plot(ax=ax1,color='#df778e', edgecolor='none',linewidth=0.075)
poly_2013_2020=gpd.read_file(path_poly+'poly_2013_2020_median_edited.shp')
poly_2013_2020.plot(ax=ax1,color='white', edgecolor='none',linewidth=0.075)

#Display high-end 2010-2018 shapefile
'''
iceslabs_jullien_highend_20102018.plot(ax=ax1,color='none', edgecolor='red',linewidth=0.5)
'''

#Display 2002-2003 ice slabs
ax1.scatter(df_2002_2003.lon,df_2002_2003.lat,c='#000000',s=0.001,zorder=2)

#Display coastlines
ax1.coastlines(edgecolor='black',linewidth=0.075)

#Display GrIS drainage bassins limits
NO_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075)
NE_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
SE_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
SW_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
CW_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
NW_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075)

#Display region name 
ax1.text(NO_rignotetal.centroid.x,NO_rignotetal.centroid.y-20000,np.asarray(NO_rignotetal.SUBREGION1)[0])
ax1.text(NE_rignotetal.centroid.x,NE_rignotetal.centroid.y+20000,np.asarray(NE_rignotetal.SUBREGION1)[0])
ax1.text(SE_rignotetal.centroid.x,SE_rignotetal.centroid.y+60000,np.asarray(SE_rignotetal.SUBREGION1)[0])
ax1.text(SW_rignotetal.centroid.x,SW_rignotetal.centroid.y-120000,np.asarray(SW_rignotetal.SUBREGION1)[0])
ax1.text(CW_rignotetal.centroid.x,CW_rignotetal.centroid.y+20000,np.asarray(CW_rignotetal.SUBREGION1)[0])
ax1.text(NW_rignotetal.centroid.x,NW_rignotetal.centroid.y+20000,np.asarray(NW_rignotetal.SUBREGION1)[0])

'''
#Display 2010-2018 high end ice slabs data
ax1.scatter(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413'],c='#9ecae1',s=0.1,zorder=2)
'''

#Display low-end 2010-2012 shapefile
iceslabs_jullien_lowend_20102012=gpd.read_file(path_df_with_elevation+'shapefiles/iceslabs_jullien_lowend_2010_11_12.shp') 

#Display low-end 2010-2018 shapefile
iceslabs_jullien_lowend_20102018=gpd.read_file(path_df_with_elevation+'shapefiles/iceslabs_jullien_lowend_20102018.shp') 

#Display low-end 2010-2018 shapefile
iceslabs_jullien_lowend_20102018.plot(ax=ax1,color='none', edgecolor='red',linewidth=2)

#Display low-end 2010-2012 shapefile
iceslabs_jullien_lowend_20102012.plot(ax=ax1,color='none', edgecolor='blue',linewidth=1)
'''
#Load 2010-2018 low estimate
f_20102018_low = open(path_df_with_elevation+'final_excel/low_estimate/df_20102018_with_elevation_low_estimate_rignotetalregions', "rb")
df_2010_2018_low = pickle.load(f_20102018_low)
f_20102018_low.close
'''
'''
#Display 2010-2012 ice slabs data
ax1.scatter(df_2010_2018_high[df_2010_2018_high.year<=2012]['lon_3413'],
            df_2010_2018_high[df_2010_2018_high.year<=2012]['lat_3413'],
            c='blue',s=1,zorder=2)
'''
'''
#Display 2010-2018 high end ice slabs data
ax1.scatter(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413'],c='red',s=0.1,zorder=2)
'''
#Load 2010-2018 high estimate for Fig3
path_RT3='C:/Users/jullienn/switchdrive/Private/research/RT3/data/export_RT1_for_RT3/'
f_20102018_RT3 = open(path_RT3+'df_20102018_with_elevation_for_RT3_masked_rignotetalregions', "rb")
df_2010_2018_RT3 = pickle.load(f_20102018_RT3)
f_20102018_RT3.close
'''
#Display 2010-2012 ice slabs data
ax1.scatter(df_2010_2018_RT3[df_2010_2018_RT3['year']<=2012]['lon_3413'],
            df_2010_2018_RT3[df_2010_2018_RT3['year']<=2012]['lat_3413'],
            c=df_2010_2018_RT3[df_2010_2018_RT3['year']<=2012]['20m_ice_content_m'],
            cmap='Blues',s=0.1,zorder=2)
'''

#Display firn aquifers
ax1.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#74c476',s=0.001,zorder=2)

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax1.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
#gl.ylabels_right = False
gl.xlabels_bottom = False
ax1.axis('off')
#ax8map.legend(loc='upper right')
###################### From Tedstone et al., 2022 #####################

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Patch(facecolor='#df778e',edgecolor='none',label='2013-2020 runoff'),
                   Patch(facecolor='#0f4969',edgecolor='none',label='1985-1993 runoff'),
                   Patch(facecolor='none',edgecolor='black',label='2010-2018 high end ice slabs'),
                   Line2D([0], [0], color='#74c476', lw=2, marker='o',linestyle='None', label='2010-2014 firn aquifers')]
ax1.legend(handles=legend_elements,loc='lower right')
plt.show()

ax1.set_xlim(-642397, 1105201)
ax1.set_ylim(-3366273, -784280)

pdb.set_trace()

'''
#Save the figure
plt.savefig('C:/Users/jullienn/switchdrive/Private/research/RT1/figures/Greenland_hydrology_summary.pdf',dpi=500)
'''

plt.close()

#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)
#Display coastlines
ax1.coastlines(edgecolor='black',linewidth=0.075)

for indiv_index in Boxes_Tedstone2022.FID:
    
    #Extract individual polygon
    indiv_polygon=Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_index]
    '''
    #Display polygon
    indiv_polygon.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.05,zorder=1)
    '''
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
    cbar=ax1.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap='Blues',vmin=vlim_min,vmax=vlim_max)
    

#Open and plot runoff limit medians shapefiles
poly_1985_1992.plot(ax=ax1,facecolor='none', edgecolor='#0f4969',linewidth=0.5)
poly_2013_2020.plot(ax=ax1,facecolor='none', edgecolor='#df778e',linewidth=0.5)

#Display high-end 2010-2018 shapefile
iceslabs_jullien_highend_20102018.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.5)

#Display GrIS drainage bassins limits
NO_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075)
NE_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
SE_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
SW_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
CW_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075) 
NW_rignotetal.plot(ax=ax1,color='none', edgecolor='black',linewidth=0.075)

#Display region name 
ax1.text(NO_rignotetal.centroid.x,NO_rignotetal.centroid.y-20000,np.asarray(NO_rignotetal.SUBREGION1)[0])
ax1.text(NE_rignotetal.centroid.x,NE_rignotetal.centroid.y+20000,np.asarray(NE_rignotetal.SUBREGION1)[0])
ax1.text(SE_rignotetal.centroid.x,SE_rignotetal.centroid.y+60000,np.asarray(SE_rignotetal.SUBREGION1)[0])
ax1.text(SW_rignotetal.centroid.x,SW_rignotetal.centroid.y-120000,np.asarray(SW_rignotetal.SUBREGION1)[0])
ax1.text(CW_rignotetal.centroid.x,CW_rignotetal.centroid.y+20000,np.asarray(CW_rignotetal.SUBREGION1)[0])
ax1.text(NW_rignotetal.centroid.x,NW_rignotetal.centroid.y+20000,np.asarray(NW_rignotetal.SUBREGION1)[0])

#Display firn aquifers
ax1.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#74c476',s=0.001,zorder=2)
#Display 2002-2003 ice slabs
ax1.scatter(df_2002_2003.lon,df_2002_2003.lat,c='#000000',s=0.001,zorder=2)

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax1.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
#gl.ylabels_right = False
gl.xlabels_bottom = False
ax1.axis('off')
#ax8map.legend(loc='upper right')
###################### From Tedstone et al., 2022 #####################

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Patch(facecolor='none',edgecolor='black',label='2010-2018 high end ice slabs'),
                   Line2D([0], [0], color='#df778e', lw=2, label='2013-2020 runoff'),
                   Line2D([0], [0], color='#0f4969', lw=2, label='1985-1993 runoff'),
                   Line2D([0], [0], color='#000000', lw=2, marker='o',linestyle='None', label='2002-2003 ice slabs'),
                   Line2D([0], [0], color='#74c476', lw=2, marker='o',linestyle='None', label='2010-2014 firn aquifers')]
ax1.legend(handles=legend_elements,loc='lower right')
plt.show()

ax1.set_xlim(-642397, 1105201)
ax1.set_ylim(-3366273, -784280)

fig.suptitle(str(desired_year)+' NDWI in background')


plt.show()
pdb.set_trace()

'''
#Save the figure
plt.savefig('C:/Users/jullienn/switchdrive/Private/research/RT1/figures/Greenland_hydrology_summary_NDWI.pdf',dpi=500)
'''