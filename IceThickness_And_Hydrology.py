# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:14:20 2023

@author: jullienn
"""


def compute_distances(eastings,northings):
    #This function is from plot_2002_2003.py, which was originally taken from MacFerrin et al., 2019
    '''Compute the distance (in m here, not km as written originally) of the traces in the file.'''
    # C = sqrt(A^2  + B^2)
    distances = np.power(np.power((eastings[1:] - eastings[:-1]),2) + np.power((northings[1:] - northings[:-1]),2), 0.5)

    #Calculate the cumsum of the distances
    cumsum_distances=np.nancumsum(distances)
    #Seeting the first value of the cumsum to be zero as it is the origin
    return_cumsum_distances=np.zeros(eastings.shape[0])
    return_cumsum_distances[1:eastings.shape[0]]=cumsum_distances

    return return_cumsum_distances

def load_and_diplay_raster(path_data,vlim_min,vlim_max,cmap_raster,ax_plot,axc_plot,type_map):
    '''
    #Define limits, here case study 2 limits
    x_min=-103459
    x_max=-89614
    y_min=-2454966
    y_max=-2447521
    '''
    x_min=-115968
    x_max=-79143
    y_min=-2454966
    y_max=-2447521
        
    ### ------------- This is from Greenland_Hydrology_Summary.py ------------- ###
    #Load raster data for display
    MapPlot = rxr.open_rasterio(path_data,masked=True).squeeze() #No need to reproject satelite image
    ### ------------- This is from Greenland_Hydrology_Summary.py ------------- ###
    
    #Extract x and y coordinates of image
    x_coord_MapPlot=np.asarray(MapPlot.x)
    y_coord_MapPlot=np.asarray(MapPlot.y)
    ### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###
    
    #Extract coordinates ofcumulative raster within Emaxs bounds
    logical_x_coord_within_bounds=np.logical_and(x_coord_MapPlot>=x_min,x_coord_MapPlot<=x_max)
    x_coord_within_bounds=x_coord_MapPlot[logical_x_coord_within_bounds]
    logical_y_coord_within_bounds=np.logical_and(y_coord_MapPlot>=y_min,y_coord_MapPlot<=y_max)
    y_coord_within_bounds=y_coord_MapPlot[logical_y_coord_within_bounds]

    #Define extents based on the bounds
    extent_MapPlot = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
    
    #Display map
    cbar_MapPlot=ax_plot.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap=cmap_raster,vmin=vlim_min,vmax=vlim_max,zorder=0)
    
    if (type_map=='DEM'):
        #Display DEM contours 
        ax_plot.contour(x_coord_within_bounds,
                        y_coord_within_bounds,
                        MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds],
                        levels=np.arange(1659,1895,5), transform=crs, origin='upper', cmap='gray_r',vmin=1650,vmax=1655,zorder=1,alpha=0.5)
        
    #Set lims
    ax_plot.set_xlim(x_min,x_max)
    ax_plot.set_ylim(y_min,y_max)
    
    #Display cbar
    cbar_MapPlot_label=fig1.colorbar(cbar_MapPlot, cax=axc_plot)
    
    if (type_map == 'DEM'):
        cbar_MapPlot_label.set_label('Elevation [m]',labelpad=15)#labelpad is from https://stackoverflow.com/questions/17475619/how-do-i-adjust-offset-colorbar-title-in-matplotlib
    elif (type_map == 'NDWI'):
        cbar_MapPlot_label.set_label('NDWI [-]',labelpad=22.5)
    else:
        print('Not known!')
        pdb.set_trace()
    


    return

import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Transformer
import cartopy.crs as ccrs
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
import os
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib_scalebar.scalebar import ScaleBar

### Set sizes ###
# this is from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes',linewidth = 0.75)  #https://stackoverflow.com/questions/1639463/how-to-change-border-width-of-a-matplotlib-graph
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
### Set sizes ###

path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceThickness/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

#Load IMBIE drainage bassins
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

###############################################################################
################## Load ice slabs with Cumulative hydro dataset ###############
###############################################################################

#Path to data
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_CumHydro_And_IceThickness=path_local+'CumHydro_and_IceThickness/csv/'
#List all the files in the folder
list_composite=os.listdir(path_CumHydro_And_IceThickness) #this is inspired from https://pynative.com/python-list-files-in-a-directory/

#Set the window distance for rolling window calculations (in meter) for ice slabs thickness spatial heterogeneity
window_distance=3000

#Define empty dataframes
upsampled_CumHydro_and_IceSlabs = pd.DataFrame()
Transects_2017_2018 = pd.DataFrame()

#Loop over all the files
for indiv_file in list_composite:
    
    if (indiv_file == 'NotClipped_With0mSlabs'):
        continue
    
    print(indiv_file)
    #Open the individual file
    indiv_csv=pd.read_csv(path_CumHydro_And_IceThickness+indiv_file)
    
    #Drop Unnamed: 0 columns
    indiv_csv.drop(columns=['Unnamed: 0'],inplace=True)
    
    ### -------------------------- Upsample data -------------------------- ###
    #Upsample data: where index_right is identical (i.e. for each CumHydro cell), keep a single value of CumHydro and average the ice content
    indiv_upsampled_CumHydro_and_IceSlabs=indiv_csv.groupby('index_right').mean()
    #Append the data to each other
    upsampled_CumHydro_and_IceSlabs=pd.concat([upsampled_CumHydro_and_IceSlabs,indiv_upsampled_CumHydro_and_IceSlabs])
    ### -------------------------- Upsample data -------------------------- ###
    
    ### -------------- Spatial heterogeneity in ice thickness ------------- ###
    #1. Calculate the distance between each sampling point
    indiv_csv['distances']=compute_distances(indiv_csv['lon_3413'].to_numpy(),indiv_csv['lat_3413'].to_numpy())

    #2. Calculate rolling window
    #Define the minimum number of data to have for average window calculation
    min_periods_for_rolling=int(window_distance/int(indiv_csv["distances"].diff().abs().median())/3)#Here, the min number of data required to compute the window is 1/3 of maximum sampling data point in the window
    
    #For adaptive window size (because sometimes data is missing along the transect), convert the distance into a timestamp, and calculate the rolling window on this timestamp. That works as expected!
    #The distance represent the seconds after and arbitrary time. Here, 1 meter = 1 second. The default is 01/01/1970
    #This idea was probably inspired while looking at this https://stackoverflow.com/questions/24337499/pandas-rolling-apply-with-variable-window-length
    indiv_csv['time_distance']=pd.to_datetime(indiv_csv['distances'].round(2),unit='s')
    #Set the time_distance to the index
    indiv_csv.set_index('time_distance',inplace=True)
    #Apply rolling window by translating the distance into second equivalent.
    indiv_csv['rolling_mean_ice_thickness'] = indiv_csv.rolling(str(window_distance)+'s',center=True,closed="both",min_periods=min_periods_for_rolling)["20m_ice_content_m"].mean()
    indiv_csv['rolling_std_ice_thickness'] = indiv_csv.rolling(str(window_distance)+'s',center=True,closed="both",min_periods=min_periods_for_rolling)["20m_ice_content_m"].std() 
    indiv_csv['rolling_CV_ice_thickness'] = indiv_csv['rolling_std_ice_thickness']/indiv_csv['rolling_mean_ice_thickness']
    indiv_csv['ice_thickness_MINUS_rolling_mean_ice_thickness'] = indiv_csv['20m_ice_content_m']-indiv_csv['rolling_mean_ice_thickness']
    
    #3. Concatenate data
    Transects_2017_2018=pd.concat([Transects_2017_2018,indiv_csv])
    ### -------------- Spatial heterogeneity in ice thickness ------------- ###

###############################################################################
################## Load ice slabs with Cumulative hydro dataset ###############
###############################################################################

###############################################################################
###                   Ice Thickness spatial heterogeneity                   ###
###############################################################################
#Reset a new index to the Transects_2017_2018 dataset
Transects_2017_2018["index"]=np.arange(0,len(Transects_2017_2018))
Transects_2017_2018.set_index('index',inplace=True)

#Display
fig = plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(2, 6)
gs.update(hspace=0.8)
gs.update(wspace=0.8)
ax1= plt.subplot(gs[0:2, 0:2])
ax2= plt.subplot(gs[0:2, 2:4])
ax3= plt.subplot(gs[0:2, 4:6])

sns.histplot(Transects_2017_2018, x="ice_thickness_MINUS_rolling_mean_ice_thickness",element="poly",bins=np.arange(-10,10,0.5),
             stat="density",ax=ax1)

sns.histplot(Transects_2017_2018, x="rolling_std_ice_thickness",element="poly",
             stat="density",ax=ax2)

sns.histplot(Transects_2017_2018, x="rolling_CV_ice_thickness",element="poly",
             stat="density",ax=ax3)
fig.suptitle('Rolling window: '+str(window_distance)+' m')

print('--- 2017-2018 dataset ---')
print(Transects_2017_2018.rolling_CV_ice_thickness.quantile([0.05,0.25,0.5,0.75]).round(2))

### --- Perform the same analysis for the 2018 transect displayed in Fig. 6 --- ###
#Open the 20180427_01_170_172 file holding 0 m thick and > 16 m thick ice slabs - inspire from the loop above
indiv_csv=pd.read_csv(path_local+'SAR_and_IceThickness/csv/NotClipped_With0mSlabs/20180427_01_170_172_NotUpsampled.csv')
#Drop Unnamed: 0 columns
indiv_csv.drop(columns=['Unnamed: 0'],inplace=True)
#Where ice thickness is lower than 1 m, set to nan
indiv_csv.loc[indiv_csv['20m_ice_content_m']<1,'20m_ice_content_m']=np.nan
#1. Calculate the distance between each sampling point
indiv_csv['distances']=compute_distances(indiv_csv['lon_3413'].to_numpy(),indiv_csv['lat_3413'].to_numpy())
#2. Calculate rolling window
#For adaptive window size (because sometimes data is missing along the transect), convert the distance into a timestamp, and calculate the rolling window on this timestamp. That works as expected!
#The distance represent the seconds after and arbitrary time. Here, 1 meter = 1 second. The default is 01/01/1970
#This idea was probably inspired while looking at this https://stackoverflow.com/questions/24337499/pandas-rolling-apply-with-variable-window-length
indiv_csv['time_distance']=pd.to_datetime(indiv_csv['distances'].round(2),unit='s')
#Set the time_distance to the index
indiv_csv.set_index('time_distance',inplace=True)

#Define the minimum number of data to have for average window calculation
min_periods_for_rolling=int(window_distance/int(indiv_csv["distances"].diff().abs().median())/3)#Here, the min number of data required to compute the window is 1/3 of maximum sampling data point in the window

#Apply rolling window by translating the distance into second equivalent.
indiv_csv['rolling_mean_ice_thickness'] = indiv_csv.rolling(str(window_distance)+'s',center=True,closed="both",min_periods=min_periods_for_rolling)["20m_ice_content_m"].mean()
indiv_csv['rolling_std_ice_thickness'] = indiv_csv.rolling(str(window_distance)+'s',center=True,closed="both",min_periods=min_periods_for_rolling)["20m_ice_content_m"].std() 
indiv_csv['rolling_CV_ice_thickness'] = indiv_csv['rolling_std_ice_thickness']/indiv_csv['rolling_mean_ice_thickness']
indiv_csv['ice_thickness_MINUS_rolling_mean_ice_thickness'] = indiv_csv['20m_ice_content_m']-indiv_csv['rolling_mean_ice_thickness']
#3. Finalise dataframe
TransectFig6=indiv_csv.copy()
#TransectFig6=Transects_2017_2018[Transects_2017_2018.Track_name=='20180427_01_170_172'].copy()

#Keep data within displayed bounds
TransectFig6_WithinBounds = TransectFig6[np.logical_and(TransectFig6.lon>=-47.70785561652585,TransectFig6.lon<=-46.85)].copy()
#Flip upside down the dataframe because the transect goes from high to low elevations
TransectFig6_WithinBounds["new_index"]=np.arange(0,len(TransectFig6_WithinBounds))
TransectFig6_WithinBounds.set_index('new_index',inplace=True)
TransectFig6_WithinBounds_reverted=TransectFig6_WithinBounds.reindex(index=TransectFig6_WithinBounds.index[::-1]).copy()#this is from https://stackoverflow.com/questions/20444087/right-way-to-reverse-a-pandas-dataframe
#Calculate new distances
TransectFig6_WithinBounds_reverted['distances_reverted']=compute_distances(TransectFig6_WithinBounds_reverted['lon_3413'].to_numpy(),TransectFig6_WithinBounds_reverted['lat_3413'].to_numpy())


#Display the transect in Fig.6
fig = plt.figure(figsize=(8.27,5.5))#Nature pdf size = (8.27,10.87)
gs = gridspec.GridSpec(8, 101)
ax_TransectFig6 = plt.subplot(gs[2:6, 0:99])

ax_TransectFig6.fill_between(TransectFig6_WithinBounds_reverted.distances_reverted,
                             TransectFig6_WithinBounds_reverted["rolling_mean_ice_thickness"]-
                             TransectFig6_WithinBounds_reverted["rolling_std_ice_thickness"],
                             TransectFig6_WithinBounds_reverted["rolling_mean_ice_thickness"]+
                             TransectFig6_WithinBounds_reverted["rolling_std_ice_thickness"],
                             alpha=0.5)
ax_TransectFig6.plot(TransectFig6_WithinBounds_reverted.distances_reverted,TransectFig6_WithinBounds_reverted["rolling_mean_ice_thickness"])
ax_TransectFig6_second=ax_TransectFig6.twinx()
ax_TransectFig6_second.plot(TransectFig6_WithinBounds_reverted.distances_reverted,TransectFig6_WithinBounds_reverted["rolling_CV_ice_thickness"],color='C1')
#Change color of axis to match color of line, this is from https://stackoverflow.com/questions/1982770/changing-the-color-of-an-axis
ax_TransectFig6.tick_params(axis='y', colors='C0')
#ax_TransectFig6_second.spines['right'].set_color('C1')
ax_TransectFig6_second.tick_params(axis='y', colors='C1')

#Coordinates of sectors to display
coord_sectors=[#(67.620575, -47.59745),
               #(67.622106, -47.566856),
               (67.626644, -47.414368),
               (67.628561, -47.33543),
               (67.629785, -47.299504),
               (67.632129, -47.232908),
               #(67.632711, -47.216256),
               #(67.633521, -47.183796),
               (67.635528, -47.14),
               (67.636072, -47.09873)]

#Display sections on the plot
for indiv_point in coord_sectors:
    #add vertical dashed lines
    ax_TransectFig6.axvline(TransectFig6_WithinBounds_reverted.loc[(TransectFig6_WithinBounds_reverted.lon-indiv_point[1]).abs().idxmin()].distances_reverted,linestyle='dashed',color='black',linewidth=1)

#Convert distance from m to km
ax_TransectFig6.set_xticks(ax_TransectFig6.get_xticks())
ax_TransectFig6.set_xticklabels((ax_TransectFig6.get_xticks()/1000).astype(int))
#add labels
ax_TransectFig6.set_xlabel('Distance [km]')
ax_TransectFig6.set_ylabel('Ice thickness [m]',color='C0')
ax_TransectFig6_second.set_ylabel('Coefficient of variation [ ]',color='C1')
ax_TransectFig6.set_xlim(0,36836)
ax_TransectFig6_second.set_ylim(-0.15,TransectFig6_WithinBounds_reverted.rolling_CV_ice_thickness.max()+0.025)
ax_TransectFig6_second.set_yticks(np.arange(0,ax_TransectFig6_second.get_ylim()[1],0.1))
#Identification of the max distance to match radargrams in CaseStudy_Emax_IceSlabs.py:
#dataframe[str(year_ticks)]['distances'][np.argmin(np.abs(dataframe[str(year_ticks)]['lon_appended']-end_transect))]/1000-dataframe[str(year_ticks)]['distances'][np.argmin(np.abs(dataframe[str(year_ticks)]['lon_appended']-start_transect))]/1000
#gives 36.836142729833256

#Add which transect is displayed
#ax_TransectFig6.text(0.97, 0.95,'2018',ha='center', va='center', transform=ax_TransectFig6.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_TransectFig6.text(0.010, 0.95,'e',ha='center', va='center', transform=ax_TransectFig6.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

### --- Perform the same analysis for the 2018 transect displayed in Fig. 6 --- ###

### --------- Perform the same analysis for an hypothetical ice slab --------- ###
#The hypothetical ice slab is based on the spacing in Fig. 6
Hypothetical_IceSlabs_Transect=pd.DataFrame()
#Compute distance
Hypothetical_IceSlabs_Transect['distances']=TransectFig6_WithinBounds_reverted.distances_reverted.copy().to_numpy()
Hypothetical_IceSlabs_Transect['time_distance']=pd.to_datetime(Hypothetical_IceSlabs_Transect['distances'].round(2),unit='s')

#The amount of noise we apply is the median of the absolute of the vector of difference spaced by the size of the applied averaging window (i.e. an approximation of the max difference) in the 2018 transect in Fig. 6.
noise_to_apply=TransectFig6_WithinBounds_reverted["20m_ice_content_m"].diff(int(window_distance/TransectFig6_WithinBounds_reverted.distances_reverted.diff().median())).abs().median()

#Create and ice slabs transect 36.8 km long (to match case study transect length), whose lowermost point's thickness is the max rolling mean thickness in Fig. 6's transect, and uppermost thickness is the min rolling mean thickness in Fig. 6's transect:
max_iceslab=TransectFig6_WithinBounds_reverted["rolling_mean_ice_thickness"].max()
min_iceslab=TransectFig6_WithinBounds_reverted["rolling_mean_ice_thickness"].min()
#Note that the transect ends at 36.8 km but the ice thickness ends at index 168, i.e. 34.9 km
ice_thickness_hypothetical = np.ones(len(Hypothetical_IceSlabs_Transect))*[np.nan]
#Define the index which will hold ice thickness > 1 m
index_to_fill=(TransectFig6_WithinBounds_reverted['20m_ice_content_m']>1)
#Create the ice thickness linear interpolation and adding a noise ranging from -noise_to_apply to +noise_to_apply to this hypothetical ice slab thickness transect.
ice_thickness_hypothetical[index_to_fill]=np.linspace(max_iceslab,min_iceslab,index_to_fill.astype(int).sum())+np.random.randint(-noise_to_apply,noise_to_apply+1,index_to_fill.astype(int).sum())
Hypothetical_IceSlabs_Transect["ice_thickness"]=ice_thickness_hypothetical
#Hypothetical_IceSlabs_Transect["ice_thickness"]=np.linspace(max_iceslab,min_iceslab,len(Hypothetical_IceSlabs_Transect))+np.random.randint(-noise_to_apply,noise_to_apply+1,len(Hypothetical_IceSlabs_Transect))

#Add the min of ice thickness to prevent having slabs thinner than 1 m which mess up the calculation of the coefficient of variation
offset=0#np.abs(Hypothetical_IceSlabs_Transect["ice_thickness"].min())
Hypothetical_IceSlabs_Transect["ice_thickness"]=Hypothetical_IceSlabs_Transect["ice_thickness"]+offset
print(' --- Ice thickness min should be > 1 m:')
print(str(np.round(Hypothetical_IceSlabs_Transect["ice_thickness"].min(),2)))

#Set the time_distance to the index
Hypothetical_IceSlabs_Transect.set_index('time_distance',inplace=True)

#Define the minimum number of data to have for average window calculation
min_periods_for_rolling=int(window_distance/int(Hypothetical_IceSlabs_Transect["distances"].diff().abs().median())/3)#Here, the min number of data required to compute the window is 1/3 of maximum sampling data point in the window

#Apply rolling window to this hypothetical ice slab by translating the distance into second equivalent.
Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness'] = Hypothetical_IceSlabs_Transect.rolling(str(window_distance)+'s',center=True,closed="both",min_periods=min_periods_for_rolling)["ice_thickness"].mean()
Hypothetical_IceSlabs_Transect['rolling_std_ice_thickness'] = Hypothetical_IceSlabs_Transect.rolling(str(window_distance)+'s',center=True,closed="both",min_periods=min_periods_for_rolling)["ice_thickness"].std()
Hypothetical_IceSlabs_Transect['rolling_CV_ice_thickness'] = Hypothetical_IceSlabs_Transect['rolling_std_ice_thickness']/Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness']
Hypothetical_IceSlabs_Transect['ice_thickness_MINUS_rolling_mean_ice_thickness'] = Hypothetical_IceSlabs_Transect['ice_thickness']-Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness']

#Remove the offset
Hypothetical_IceSlabs_Transect['ice_thickness']=Hypothetical_IceSlabs_Transect['ice_thickness']-offset
Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness']=Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness']-offset

#Reset index to integer
Hypothetical_IceSlabs_Transect['new_index']=TransectFig6_WithinBounds_reverted.index
Hypothetical_IceSlabs_Transect.set_index('new_index',inplace=True)

#Display the hypothetical ice slab
fig = plt.figure(figsize=(8.27,3.55))#Nature pdf size = (8.27,10.87)
gs = gridspec.GridSpec(8, 101)
ax_Hypothetical_radargram = plt.subplot(gs[0:4, 0:99])
ax_Hypothetical_plot = plt.subplot(gs[4:8, 0:99])

#Hypothetical radargram
ax_Hypothetical_radargram.fill_between(Hypothetical_IceSlabs_Transect["distances"],Hypothetical_IceSlabs_Transect["ice_thickness"],color='grey',label='Idealised ice slab')
ax_Hypothetical_radargram.set_ylim(19,0)
ax_Hypothetical_radargram.set_xlim(ax_TransectFig6.get_xlim())
ax_Hypothetical_radargram.set_ylabel('Depth [m]')
ax_Hypothetical_radargram.set_xticks([])
ax_Hypothetical_radargram.legend(loc='upper center')

#Average, stdev and cv
ax_Hypothetical_plot.fill_between(Hypothetical_IceSlabs_Transect.distances,
                                  Hypothetical_IceSlabs_Transect["rolling_mean_ice_thickness"]-
                                  Hypothetical_IceSlabs_Transect["rolling_std_ice_thickness"],
                                  Hypothetical_IceSlabs_Transect["rolling_mean_ice_thickness"]+
                                  Hypothetical_IceSlabs_Transect["rolling_std_ice_thickness"],
                                  alpha=0.5)
ax_Hypothetical_plot.plot(Hypothetical_IceSlabs_Transect.distances,Hypothetical_IceSlabs_Transect["rolling_mean_ice_thickness"])
ax_Hypothetical_plot_second=ax_Hypothetical_plot.twinx()
ax_Hypothetical_plot_second.plot(Hypothetical_IceSlabs_Transect.distances,Hypothetical_IceSlabs_Transect["rolling_CV_ice_thickness"],color='C1')
#Change color of axis to match color of line, this is from https://stackoverflow.com/questions/1982770/changing-the-color-of-an-axis
ax_Hypothetical_plot.tick_params(axis='y', colors='C0')
#ax_TransectFig6_second.spines['right'].set_color('C1')
ax_Hypothetical_plot_second.tick_params(axis='y', colors='C1')

#Custom labels
ax_Hypothetical_plot.set_xlabel('Distance [km]')
ax_Hypothetical_plot.set_ylabel('Ice thickness [m]',color='C0')
ax_Hypothetical_plot_second.set_ylabel('Coefficient of variation [ ]',color='C1',)
ax_Hypothetical_plot.set_ylim(ax_TransectFig6.get_ylim())
ax_Hypothetical_plot_second.set_ylim(0,TransectFig6_WithinBounds_reverted.rolling_CV_ice_thickness.max()+0.025)
ax_Hypothetical_plot.set_xticks(ax_Hypothetical_plot.get_xticks())
ax_Hypothetical_plot.set_xticklabels((ax_Hypothetical_plot.get_xticks()/1000).astype(int))
ax_Hypothetical_plot.set_xlim(ax_TransectFig6.get_xlim())

#Add panel labels
ax_Hypothetical_radargram.text(0.01, 0.925,'a',ha='center', va='center', transform=ax_Hypothetical_radargram.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_Hypothetical_plot.text(0.010, 0.9,'b',ha='center', va='center', transform=ax_Hypothetical_plot.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#Custom legend myself - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Line2D([0], [0], color='C0', label='Mean'),
                   Patch(facecolor='C0',alpha=0.5,label='Mean +/- standard deviation'),
                   Line2D([0], [0], color='C1', label='Coefficient of variation')]
ax_Hypothetical_plot.legend(handles=legend_elements,loc='upper center',fontsize=8,framealpha=0.6).set_zorder(7)
plt.show()

print('--- Hypothetical ice slab ---')
print(Hypothetical_IceSlabs_Transect.rolling_CV_ice_thickness.quantile([0.05,0.25,0.5,0.75]).round(2))
#The spatial variability of the 2017/2018 ice slabs dtatset is about twice as the one from the hypothetical ice slabs transect.
#Note that in the real world dataset, there are longitudinal transects! But these ones are expected to have an even lower variability if we consider no other influence that melting gradient due to elevation gradient.
#Note that quantile 0.95 is highly variable at each run, so we do not consider it anymore! On the other han, quantile 0.25, 0.5, 0.75 are stable between the different runs.

#pdb.set_trace()

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig6/v4/Fig_hypothetical_slab.png',dpi=300,bbox_inches='tight')
'''
plt.close()

Corresponding_Hypothetical_index_Hypothetical_IceSlabs_Transect=[]
Corresponding_Hypothetical_index_TransectFig6_WithinBounds_reverted=[]
#Loop along the transect in Fig, extract the index in the Hypothetical_IceSlabs_Transect dataframe where the difference between the ice thickness a the Fig.6 transect place is the closest to the ice thickness ion the hypothetical dataframe
for index, row in TransectFig6_WithinBounds_reverted.iterrows():
    Corresponding_Hypothetical_index_Hypothetical_IceSlabs_Transect=np.append(Corresponding_Hypothetical_index_Hypothetical_IceSlabs_Transect,
                                                                             (Hypothetical_IceSlabs_Transect.rolling_mean_ice_thickness-row['rolling_mean_ice_thickness']).abs().idxmin())
    Corresponding_Hypothetical_index_TransectFig6_WithinBounds_reverted=np.append(Corresponding_Hypothetical_index_TransectFig6_WithinBounds_reverted,
                                                                                 index)

#Construct a pd.dataframe
PickUp_df=pd.DataFrame(data={"to_pickup":Corresponding_Hypothetical_index_Hypothetical_IceSlabs_Transect},
                       index=Corresponding_Hypothetical_index_TransectFig6_WithinBounds_reverted.astype(int))
#Drop NaNs
PickUp_df.dropna(inplace=True)
#Store the corresponding hypothetical Cv
TransectFig6_WithinBounds_reverted.loc[PickUp_df.index,'Hypothetical_Cv']=Hypothetical_IceSlabs_Transect.loc[PickUp_df.to_pickup,'rolling_CV_ice_thickness'].to_numpy()
#Display in Fig. transect 6 the place where the coefficient of variation cannot be interpreted, i.e. where the Cv is larger than idealised Cv for a specific ice thickness
ax_TransectFig6_second.plot(TransectFig6_WithinBounds_reverted.distances_reverted,TransectFig6_WithinBounds_reverted["Hypothetical_Cv"],color='black')

#Custom legend myself - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Line2D([0], [0], color='C0', label='Mean'),
                   Patch(facecolor='C0',alpha=0.5,label='Mean +/- standard deviation'),
                   Line2D([0], [0], color='C1', label='Coefficient of variation'),
                   Line2D([0], [0], color='black', label='Idealised coefficient of variation')]
ax_TransectFig6_second.legend(handles=legend_elements,loc='lower left',fontsize=8,framealpha=0.8).set_zorder(7)
plt.show()

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig6/v4/Fig6_e.png',dpi=300)#,bbox_inches='tight')
'''
### --------- Perform the same analysis for an hypothetical ice slab --------- ###

#pdb.set_trace()

### --------------------------- Sector B focus --------------------------- ###
#Display NDWI - this is from CaseStudy_Emax_IceSlabs.py

#Define transformer for coordinates transform from "EPSG:4326" to "EPSG:3413"
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
import rioxarray as rxr

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Display image
plt.rcParams.update({'font.size': 8})
fig1 = plt.figure(figsize=(8.27,5.16))#Nature pdf size = (8.27,10.87)
gs = gridspec.GridSpec(52, 101)
gs.update(wspace=0.1)
gs.update(wspace=0.5)
ax_NDWI = plt.subplot(gs[1:18, 0:100], projection=crs)
axc_NDWI = plt.subplot(gs[1:18, 100:101])
ax_DEM = plt.subplot(gs[18:35, 0:100], projection=crs)
axc_DEM = plt.subplot(gs[18:35, 100:101])
ax_StrainRate = plt.subplot(gs[35:52, 0:100], projection=crs)
axc_StrainRate = plt.subplot(gs[35:52, 100:101])

axc_IceThickness = plt.subplot(gs[0:1, 35:65])

### NDWI ###
#path_NDWI='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/NDWI/'+'NDWI_p10_'+str(2019)+'.vrt'
path_NDWI='X:/RT3_jullien/NDWI/'+'NDWI_p10_'+str(2019)+'.vrt'
vlim_min_NDWI=0
vlim_max_NDWI=0.6
cmap_NDWI='Blues'
#Display
load_and_diplay_raster(path_NDWI,vlim_min_NDWI,vlim_max_NDWI,cmap_NDWI,ax_NDWI,axc_NDWI,'NDWI')
### NDWI ###


### Local Artic DEM ###
path_DEM='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/10m_arcticDEM/'+'16_3940_10m_v3.0_reg_dem_ClippedToTransectFig5.tif'
vlim_min_DEM=1663#1710
vlim_max_DEM=1890#1810
cmap_DEM='Spectral'
#Display
load_and_diplay_raster(path_DEM,vlim_min_DEM,vlim_max_DEM,cmap_DEM,ax_DEM,axc_DEM,'DEM')
### Local Artic DEM ###

#Display transect
cbar_IceThickness=ax_NDWI.scatter(TransectFig6_WithinBounds_reverted.lon_3413,TransectFig6_WithinBounds_reverted.lat_3413,c=TransectFig6_WithinBounds_reverted["20m_ice_content_m"],cmap='gray_r')
ax_DEM.scatter(TransectFig6_WithinBounds_reverted.lon_3413,TransectFig6_WithinBounds_reverted.lat_3413,c=TransectFig6_WithinBounds_reverted["20m_ice_content_m"],cmap='gray_r')
ax_StrainRate.scatter(TransectFig6_WithinBounds_reverted.lon_3413,TransectFig6_WithinBounds_reverted.lat_3413,c=TransectFig6_WithinBounds_reverted["20m_ice_content_m"],cmap='gray_r')

'''
#Display sector B
ax_NDWI.scatter(TransectFig6_WithinBounds_reverted[np.logical_and(TransectFig6_WithinBounds_reverted.lon>=-47.299504,TransectFig6_WithinBounds_reverted.lon<=-47.232908)]['lon_3413'],
                TransectFig6_WithinBounds_reverted[np.logical_and(TransectFig6_WithinBounds_reverted.lon>=-47.299504,TransectFig6_WithinBounds_reverted.lon<=-47.232908)]['lat_3413'])
'''
#Coordinates of sectors to display
coord_sectors=[#(67.620575, -47.59745),
               #(67.622106, -47.566856),
               (67.626644, -47.414368),
               (67.628561, -47.33543),
               (67.629785, -47.299504),
               (67.632129, -47.232908),
               #(67.632711, -47.216256),
               #(67.633521, -47.183796),
               (67.635528, -47.14),
               (67.636072, -47.09873)]
#Display sections on the map
for indiv_point in coord_sectors:
    #Transform the coordinates from EPSG:3413 to EPSG:4326
    #Example from: https://pyproj4.github.io/pyproj/stable/examples.html
    points=transformer.transform(indiv_point[1],indiv_point[0])
    ax_NDWI.axvline(points[0],zorder=3,color='black',linestyle='dashed',linewidth=1)
    ax_DEM.axvline(points[0],zorder=3,color='black',linestyle='dashed',linewidth=1)
    ax_StrainRate.axvline(points[0],zorder=3,color='black',linestyle='dashed',linewidth=1)

#Display strain rate, from https://ubir.buffalo.edu/xmlui/handle/10477/82127
import xarray
GrIS_StrainRate= xarray.open_dataset("C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/10m_arcticDEM/GrISWinterPrincipalStrainrate_0670.nc")
GrIS_StrainRate.rio.write_crs(3413,inplace=True)
#Make sure crs is set
print(GrIS_StrainRate.spatial_ref)

'''
#Define limits, here case study 2 limits
x_min=-103459
x_max=-89614
y_min=-2454966
y_max=-2447521
'''
x_min=-115968
x_max=-79143
y_min=-2454966
y_max=-2447521

#Extract x and y coordinates of image
x_coord_GrIS_StrainRate=GrIS_StrainRate.x.data
y_coord_GrIS_StrainRate=GrIS_StrainRate.y.data
### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###

#Extract coordinates ofcumulative raster within Emaxs bounds
logical_x_coord_within_bounds=np.logical_and(x_coord_GrIS_StrainRate>=x_min,x_coord_GrIS_StrainRate<=x_max)
x_coord_within_bounds=x_coord_GrIS_StrainRate[logical_x_coord_within_bounds]
logical_y_coord_within_bounds=np.logical_and(y_coord_GrIS_StrainRate>=y_min,y_coord_GrIS_StrainRate<=y_max)
y_coord_within_bounds=y_coord_GrIS_StrainRate[logical_y_coord_within_bounds]

#Define extents based on the bounds
extent_GrIS_StrainRate = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]

#Display map
cbar_StrainRate=ax_StrainRate.imshow(GrIS_StrainRate.ep[logical_y_coord_within_bounds,logical_x_coord_within_bounds],
                                     extent=extent_GrIS_StrainRate, transform=crs, origin='upper', cmap='RdBu_r',
                                     vmin=-0.0015, vmax=0.0015,zorder=0)
                                     #vmin=np.quantile(GrIS_StrainRate.ep[logical_y_coord_within_bounds,logical_x_coord_within_bounds].data,0.01),
                                     #vmax=np.quantile(GrIS_StrainRate.ep[logical_y_coord_within_bounds,logical_x_coord_within_bounds].data,0.99),zorder=0)

#Set lims
ax_StrainRate.set_xlim(x_min,x_max)
ax_StrainRate.set_ylim(y_min,y_max)

#Display cbar
cbar_StrainRate_label=fig1.colorbar(cbar_StrainRate, cax=axc_StrainRate)
cbar_StrainRate_label.set_label('Principal strain rate [$yr^{-1}$]')

#Display cbar IceThickness
cbar_IceThickness_label=fig1.colorbar(cbar_IceThickness, cax=axc_IceThickness,orientation='horizontal',ticklocation='top')#Inspired from https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
cbar_IceThickness_label.set_label('Ice slab thickness [m]')

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax_NDWI.gridlines(draw_labels=True, xlocs=[-46.8,-47.0,-47.2,-47.4,-47.6], ylocs=[67.60,67.625,67.65], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
gl.right_labels = False
gl.bottom_labels = False
gl.top_labels = False

gl=ax_DEM.gridlines(draw_labels=True, xlocs=[-46.8,-47.0,-47.2,-47.4,-47.6], ylocs=[67.60,67.625,67.65], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
gl.right_labels = False
gl.bottom_labels = False
gl.top_labels = False

gl=ax_StrainRate.gridlines(draw_labels=True, xlocs=[-46.8,-47.0,-47.2,-47.4,-47.6], ylocs=[67.60,67.625,67.65], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
gl.right_labels = False
gl.top_labels = False
###################### From Tedstone et al., 2022 #####################

# Display scalebar with GeoPandas
ax_NDWI.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))
ax_DEM.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))
ax_StrainRate.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))

#Display panel label
ax_NDWI.text(0.01, 0.9,'a',ha='center', va='center', transform=ax_NDWI.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_DEM.text(0.01, 0.9,'b',ha='center', va='center', transform=ax_DEM.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_StrainRate.text(0.01, 0.9,'c',ha='center', va='center', transform=ax_StrainRate.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig6/v4/FigS4_new.png',dpi=300,bbox_inches='tight')
'''
### --------------------------- Sector B focus --------------------------- ###

pdb.set_trace()

###############################################################################
###                   Ice Thickness spatial heterogeneity                   ###
###############################################################################

#Should apply a noise so that the resulting rolling_std_ice_thickness in the idealised transect equals that:
TransectFig6_WithinBounds_reverted["rolling_std_ice_thickness"].median()

pdb.set_trace()

#Display the hypothetical ice slab
fig = plt.figure(figsize=(8.27,3.55))#Nature pdf size = (8.27,10.87)
gs = gridspec.GridSpec(8, 101)
ax_Hypothetical_radargram = plt.subplot(gs[4:8, 0:99])
ax_radargram_Fig6 = plt.subplot(gs[0:4, 0:99])

#Radargram Fig. 6
ax_radargram_Fig6.fill_between(TransectFig6_WithinBounds_reverted.distances_reverted,
                               TransectFig6_WithinBounds_reverted["20m_ice_content_m"])
ax_radargram_Fig6.set_ylim(20,0)
ax_radargram_Fig6.set_xlim(0,35000)
TransectFig6_WithinBounds_reverted["rolling_std_ice_thickness"].mean()

#Hypothetical radargram
ax_Hypothetical_radargram.fill_between(Hypothetical_IceSlabs_Transect["distance"],Hypothetical_IceSlabs_Transect["underlying_data"]-offset,color='grey')
ax_Hypothetical_radargram.set_ylim(20,0)
ax_Hypothetical_radargram.set_xlim(0,35)
Hypothetical_IceSlabs_Transect["rolling_std_ice_thickness"].mean()




fig = plt.figure(figsize=(8.27,3.55))#Nature pdf size = (8.27,10.87)
gs = gridspec.GridSpec(8, 101)
ax1 = plt.subplot(gs[0:8, 0:99])
#Radargram Fig. 6
ax1.plot(TransectFig6_WithinBounds_reverted.distances_reverted,
         TransectFig6_WithinBounds_reverted["20m_ice_content_m"])
ax1.set_ylim(20,0)
ax1.set_xlim(0,35000)

###############################################################################
###                       CumHydro and Ice Thickness                        ###
###############################################################################

#Reset a new index to this upsampled dataset
upsampled_CumHydro_and_IceSlabs["index"]=np.arange(0,len(upsampled_CumHydro_and_IceSlabs))
upsampled_CumHydro_and_IceSlabs.set_index('index',inplace=True)
#Transform upsampled_CumHydro_and_IceSlabs as a geopandas dataframe
upsampled_CumHydro_and_IceSlabs_gdp = gpd.GeoDataFrame(upsampled_CumHydro_and_IceSlabs,
                                                       geometry=gpd.GeoSeries.from_xy(upsampled_CumHydro_and_IceSlabs['lon_3413'],
                                                                                      upsampled_CumHydro_and_IceSlabs['lat_3413'],
                                                                                      crs='EPSG:3413'))
#Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
upsampled_CumHydro_and_IceSlabs_gdp_with_regions = gpd.sjoin(upsampled_CumHydro_and_IceSlabs_gdp, GrIS_drainage_bassins, predicate='within')
#Drop data belonging to ice cap
upsampled_CumHydro_and_IceSlabs_gdp_with_regions=upsampled_CumHydro_and_IceSlabs_gdp_with_regions[~(upsampled_CumHydro_and_IceSlabs_gdp_with_regions.SUBREGION1=="ICE_CAP")]

#Full ice slabs dataset
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_CumHydro_Thickness = plt.subplot(gs[0:10, 0:6])
ax_CumHydro_Thickness.hist2d(upsampled_CumHydro_and_IceSlabs_gdp_with_regions.raster_values,
                             upsampled_CumHydro_and_IceSlabs_gdp_with_regions['20m_ice_content_m'],cmap='magma_r',
                             bins=[np.arange(0,300,5),np.arange(0,20,0.5)],norm=mpl.colors.LogNorm())#,cmax=upsampled_CumHydro_and_IceSlabs.raster_values.quantile(0.5))

#Prepare empty dataframe
upsampled_CumHydro_and_IceSlabs_ForAnalysis=pd.DataFrame()

#Display per region
fig_CumHydro_IceThickness = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(4, 6)
gs.update(hspace=0.8)
gs.update(wspace=0.8)
ax_SW = plt.subplot(gs[0:2, 0:2])
ax_CW = plt.subplot(gs[0:2, 2:4])
ax_NW = plt.subplot(gs[0:2, 4:6])
ax_NO = plt.subplot(gs[2:4, 0:2])
ax_NE = plt.subplot(gs[2:4, 2:4])
ax_hist = plt.subplot(gs[2:4, 4:6])

for indiv_region in list(['SW','CW','NW','NO','NE']):
    
    #Remove the floor(min) in each region - floor of the min to allow fo log axis display in distribution plot
    regional_df=upsampled_CumHydro_and_IceSlabs_gdp_with_regions[upsampled_CumHydro_and_IceSlabs_gdp_with_regions.SUBREGION1==indiv_region].copy()
    regional_df["raster_values_minus_min"]=regional_df.raster_values-np.floor(regional_df.raster_values.min())
    
    #Concatenate
    upsampled_CumHydro_and_IceSlabs_ForAnalysis=pd.concat([upsampled_CumHydro_and_IceSlabs_ForAnalysis,regional_df])
    
    #Display
    if (indiv_region == 'SW'):
        axis_plot=ax_SW
    elif (indiv_region == 'CW'):
        axis_plot=ax_CW
    elif (indiv_region == 'NW'):
        axis_plot=ax_NW
    elif (indiv_region == 'NO'):
        axis_plot=ax_NO
    elif (indiv_region == 'NE'):
        axis_plot=ax_NE
    else:
        pdb.set_trace()
        
    cbar_region = axis_plot.hist2d(data=regional_df,
                                   x="raster_values_minus_min",
                                   y="20m_ice_content_m",cmap='magma_r',
                                   bins=[np.arange(0,100,5),np.arange(0,17,1)],norm=mpl.colors.LogNorm())
    
    #Display colorbar
    if (indiv_region == 'NO'):
        fig_CumHydro_IceThickness.colorbar(cbar_region[3], ax=axis_plot,label='Occurrence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot
    else:
        fig_CumHydro_IceThickness.colorbar(cbar_region[3], ax=axis_plot,label='') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot

    #Display region name
    axis_plot.set_title(indiv_region)

#Display regional CumHydro distributions
distribs=sns.histplot(upsampled_CumHydro_and_IceSlabs_ForAnalysis, x="raster_values_minus_min", hue="SUBREGION1",element="poly",
                      stat="count",log_scale=[False,True],bins=np.arange(0,100,5),ax=ax_hist)
sns.move_legend(distribs,"upper right",title="")
ax_hist.set_xlim(0,95)
ax_hist.set_xlabel('Hydrological occurrence')


### Finalise plot ###
#Set labels
ax_NO.set_xlabel('Hydrological occurrence')
ax_NO.set_ylabel('Ice thickness [m]')

#Add backgound to display panel label
ax_SW.text(0.045, 0.935,' ',ha='center', va='center', transform=ax_SW.transAxes,weight='bold',fontsize=10,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),zorder=10)
ax_CW.text(0.045, 0.935,' ',ha='center', va='center', transform=ax_CW.transAxes,weight='bold',fontsize=10,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),zorder=10)
ax_NW.text(0.045, 0.935,' ',ha='center', va='center', transform=ax_NW.transAxes,weight='bold',fontsize=10,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),zorder=10)
ax_NO.text(0.045, 0.935,' ',ha='center', va='center', transform=ax_NO.transAxes,weight='bold',fontsize=10,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),zorder=10)
ax_NE.text(0.045, 0.935,' ',ha='center', va='center', transform=ax_NE.transAxes,weight='bold',fontsize=10,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),zorder=10)
#Add panel labels
ax_SW.text(0.04, 0.925,'a',ha='center', va='center', transform=ax_SW.transAxes,weight='bold',fontsize=15,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_CW.text(0.04, 0.925,'b',ha='center', va='center', transform=ax_CW.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
ax_NW.text(0.04, 0.925,'c',ha='center', va='center', transform=ax_NW.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
ax_NO.text(0.04, 0.925,'d',ha='center', va='center', transform=ax_NO.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
ax_NE.text(0.04, 0.925,'e',ha='center', va='center', transform=ax_NE.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
ax_hist.text(0.04, 0.925,'f',ha='center', va='center', transform=ax_hist.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
### Finalise plot ###

pdb.set_trace()

#SAve figure
plt.savefig(path_switchdrive+'RT3/figures/Fig5/v1/Fig5.png',dpi=300,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen


###############################################################################
###                       CumHydro and Ice Thickness                        ###
###############################################################################


