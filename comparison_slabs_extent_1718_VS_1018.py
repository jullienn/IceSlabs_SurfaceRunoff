# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:09:14 2024

@author: jullienn
"""

def plot_zoom(ax_plot,xlim_plot,ylim_plot):
    
    #Display Greenland coast shapefile
    GreenlandCoast.plot(ax=ax_plot,color='#CEB481', edgecolor='grey',linewidth=0.1)
    GrIS_mask.plot(ax=ax_plot,color='white', edgecolor='black',linewidth=0.075)
    #Display 2010-2018 high end ice slabs jullien et al., 2023
    iceslabs_20102018_jullienetal2023.plot(ax=ax_plot,facecolor='black',edgecolor='black')
    #Display 2017-2018 high end ice slabs
    iceslabs_20172018.plot(ax=ax_plot,facecolor='#ba2b2b',edgecolor='#ba2b2b')
    #Display flightlines
    ax_plot.scatter(gdf_flighlines_2017_GrIS.lon_3413,gdf_flighlines_2017_GrIS.lat_3413,s=0.5,color='#bdbdbd',edgecolor='none')
    ax_plot.scatter(gdf_flighlines_2018_GrIS.lon_3413,gdf_flighlines_2018_GrIS.lat_3413,s=0.5,color='#bdbdbd',edgecolor='none')
    #Display 2017-2018 ice slabs thickness
    ax_plot.scatter(ice_slabs_thickness_2017_2018.lon_3413,ice_slabs_thickness_2017_2018.lat_3413,s=1,color='#3182bd',edgecolor='none')
    #Custom limits
    #ax_plot.axis('off')
    
    ax_plot.set_xlim(xlim_plot[0], xlim_plot[1])
    ax_plot.set_ylim(ylim_plot[0], ylim_plot[1])
    # Display scalebar with GeoPandas
    ax_plot.add_artist(ScaleBar(1,location='lower left',box_alpha=0))
    
    return


def add_rectangle_zoom(axis_InsetMap,axis_plot):
    #Display rectangle around datalocation - this is from Fig1.py paper Greenland Ice Sheet Ice Slabs Expansion and Thickening  
    #This is from https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    # Create a Rectangle patch and add the patch to the Axes
    axis_InsetMap.add_patch(patches.Rectangle(([axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][0],[axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][1]),
                                              np.abs([axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][0]-[axis_plot.get_xlim()[1],axis_plot.get_ylim()[1]][0]),
                                              np.abs([axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][1]-[axis_plot.get_xlim()[1],axis_plot.get_ylim()[1]][1]),
                                              angle=0, linewidth=1, edgecolor='black', facecolor='none'))
    return

import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Transformer
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import seaborn as sns
import rioxarray as rxr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import geoutils as gu
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches

#Define paths
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'
path_rignotetal2016_GrIS=path_switchdrive+'backup_Aglaja/working_environment/greenland_topo_data/'
path_jullienetal2023=path_switchdrive+'RT1/final_dataset_2002_2018/'
path_data=path_switchdrive+'RT3/data/'
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################


#Define palette for time , this is From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
pal_year= {2012 : "#6baed6", 2019 : "#fcbba1"}

### -------------------------- Load shapefiles --------------------------- ###
#Load Rignot et al., 2016 Greenland drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS+'GRE_Basins_IMBIE2_v1.3/GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp',rows=slice(51,57,1)) #the regions are the last rows of the shapefile
#Extract indiv regions and create related indiv shapefiles
NW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NW']
CW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='CW']
SW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SW']
NO_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NO']
NE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NE']
SE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SE']
#Load Rignot et al., 2016 Greenland Ice Sheet mask
GrIS_mask=gpd.read_file(path_rignotetal2016_GrIS+'GRE_IceSheet_IMBIE2/GRE_IceSheet_IMBIE2/GRE_IceSheet_IMBIE2_v1_EPSG3413.shp',rows=slice(1,2,1)) 
#Load Greenland coast shapefile
GreenlandCoast=gpd.read_file('C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/Greenland_coast/Greenland_coast.shp') 
#load 2010-2018 ice slabs high end from Jullien et al., (2023)
iceslabs_20102018_jullienetal2023=gpd.read_file(path_jullienetal2023+'/shapefiles/iceslabs_jullien_highend_20102018.shp')
#load 2017-2018 ice slabs high end but name it as iceslabs_20102018_jullienetal2023 to avoid changes in variable names
iceslabs_20172018=gpd.read_file(path_switchdrive+'RT3/data/IceSlabsExtentHighEndJullien_20172018/iceslabs_jullien_highend_20172018.shp')
### -------------------------- Load shapefiles --------------------------- ###

### -------------------------- Load 2017-2018 ice slabs thickness --------------------------- ###
ice_slabs_thickness_2010_2018 = pd.read_csv(path_jullienetal2023+'final_excel/high_estimate/clipped/'+'Ice_Layer_Output_Thicknesses_2010_2018_jullienetal2023_high_estimate_cleaned.csv')
ice_slabs_thickness_2017_2018 = ice_slabs_thickness_2010_2018[ice_slabs_thickness_2010_2018.Track_name.str[:4].astype(int)>=2017].copy()
#Transform coordinates from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
points=transformer.transform(np.asarray(ice_slabs_thickness_2017_2018["lon"]),np.asarray(ice_slabs_thickness_2017_2018["lat"]))
ice_slabs_thickness_2017_2018['lon_3413']=points[0]
ice_slabs_thickness_2017_2018['lat_3413']=points[1]
### -------------------------- Load 2017-2018 ice slabs thickness --------------------------- ###


### -------------------------- Load 2017-2018 AR flightlines --------------------------- ###
flighlines_2017 = pd.read_csv(path_jullienetal2023+'flightlines/'+'2017_Greenland_P3.csv',usecols=['LAT','LON'])
flighlines_2018 = pd.read_csv(path_jullienetal2023+'flightlines/'+'2018_Greenland_P3.csv',usecols=['LAT','LON'])
#Transform coordinates from WGS84 to EPSG:3413
points=transformer.transform(np.asarray(flighlines_2017["LON"]),np.asarray(flighlines_2017["LAT"]))
flighlines_2017['lon_3413']=points[0]
flighlines_2017['lat_3413']=points[1]

points=transformer.transform(np.asarray(flighlines_2018["LON"]),np.asarray(flighlines_2018["LAT"]))
flighlines_2018['lon_3413']=points[0]
flighlines_2018['lat_3413']=points[1]

#Make flightlines geopandas dataframes
gdf_flighlines_2017 = gpd.GeoDataFrame(flighlines_2017, geometry=gpd.points_from_xy(flighlines_2017.lon_3413, flighlines_2017.lat_3413), crs="EPSG:3413")
gdf_flighlines_2018 = gpd.GeoDataFrame(flighlines_2018, geometry=gpd.points_from_xy(flighlines_2018.lon_3413, flighlines_2018.lat_3413), crs="EPSG:3413")

#Clip flightlines to Greenland mask
gdf_flighlines_2017_GrIS = gpd.sjoin(gdf_flighlines_2017, GrIS_mask, how='left', predicate='within')
gdf_flighlines_2018_GrIS = gpd.sjoin(gdf_flighlines_2018, GrIS_mask, how='left', predicate='within')
#Drop nans to keep only flightlines which intersect with GrIS mask
gdf_flighlines_2017_GrIS.dropna(inplace=True)
gdf_flighlines_2018_GrIS.dropna(inplace=True)
### -------------------------- Load 2017-2018 AR flightlines --------------------------- ###

#Prepare plot
#Set fontsize plot
plt.rcParams.update({'font.size': 8})
fig = plt.figure()
fig.set_size_inches(6.5, 10.15) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
gs = gridspec.GridSpec(60, 60)
ax_NW = plt.subplot(gs[0:20, 0:30],projection=crs)
ax_CW = plt.subplot(gs[20:40, 0:30],projection=crs)
ax_SW = plt.subplot(gs[40:60, 0:30],projection=crs)
ax_NO = plt.subplot(gs[0:20, 30:60],projection=crs)
ax_inset_map = plt.subplot(gs[20:57, 30:60],projection=crs)

#Plot zoom over regions
plot_zoom(ax_SW,[-188026, -47955],[-2755751, -2374229])
plot_zoom(ax_CW,[-193883, -48609],[-2374229, -1980264])
plot_zoom(ax_NW,[-344906, -175535],[-1962869, -1482518])
plot_zoom(ax_NO,[-604315,-206822],[-1389835,-972158])

#Display inset map
GreenlandCoast.plot(ax=ax_inset_map,color='#CEB481', edgecolor='grey',linewidth=0.1)
GrIS_mask.plot(ax=ax_inset_map,color='white', edgecolor='black',linewidth=0.075)
#Display 2010-2018 high end ice slabs jullien et al., 2023
iceslabs_20102018_jullienetal2023.plot(ax=ax_inset_map,facecolor='black',edgecolor='black')
#Display 2017-2018 high end ice slabs
iceslabs_20172018.plot(ax=ax_inset_map,facecolor='#ba2b2b',edgecolor='#ba2b2b')
ax_inset_map.axis('off')
ax_inset_map.set_xlim(-613438, 843988)
ax_inset_map.set_ylim(-3365447, -630663)
gl=ax_inset_map.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=6)
gl.right_labels = False
gl.top_labels = False

#Display boxes zoom
add_rectangle_zoom(ax_inset_map,ax_SW)
add_rectangle_zoom(ax_inset_map,ax_CW)
add_rectangle_zoom(ax_inset_map,ax_NW)
add_rectangle_zoom(ax_inset_map,ax_NO)

#Display panel label
ax_NW.text(0.1, 0.95,'a',ha='center', va='center', transform=ax_NW.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_CW.text(0.1, 0.95,'b',ha='center', va='center', transform=ax_CW.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_SW.text(0.1, 0.95,'c',ha='center', va='center', transform=ax_SW.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_NO.text(0.05, 0.95,'d',ha='center', va='center', transform=ax_NO.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(0.05, 0.95,'e',ha='center', va='center', transform=ax_inset_map.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#Display panel label on inset map
ax_inset_map.text(ax_NW.get_xlim()[1]+60000, (ax_NW.get_ylim()[0]+ax_NW.get_ylim()[1])/2,'a',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(ax_CW.get_xlim()[1]+60000, (ax_CW.get_ylim()[0]+ax_CW.get_ylim()[1])/2,'b',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(ax_SW.get_xlim()[1]+60000, (ax_SW.get_ylim()[0]+ax_SW.get_ylim()[1])/2,'c',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(ax_NO.get_xlim()[1]+60000, (ax_NO.get_ylim()[0]+ax_NO.get_ylim()[1])/2,'d',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Patch(facecolor='black',edgecolor='none',label='2010-2018 ice slabs extent (maximum)'),
                   Patch(facecolor='#ba2b2b',edgecolor='none',label='2017-2018 ice slabs extent'),
                   Line2D([0], [0], color='#bdbdbd', lw=2, label='2017-2018 OIB AR flight-lines'),
                   Line2D([0], [0], color='#3182bd', lw=2, label='2017-2018 ice slabs')]
ax_inset_map.legend(handles=legend_elements,loc='lower center',fontsize=8,framealpha=0.7).set_zorder(7)
plt.show()

pdb.set_trace()
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig_201718_adjustment/v1/Fig_201718_adjustment.png',dpi=1000,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen

