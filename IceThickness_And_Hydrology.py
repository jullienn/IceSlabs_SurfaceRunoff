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
    
    #For adaptive window size (because sometimes data is missing along the transect), convert the distance into a timestamp, and calculate the rolling window on this timestamp. That works as expected!
    #The distance represent the seconds after and arbitrary time. Here, 1 meter = 1 second. The default is 01/01/1970
    #This idea was probably inspired while looking at this https://stackoverflow.com/questions/24337499/pandas-rolling-apply-with-variable-window-length
    indiv_csv['time_distance']=pd.to_datetime(indiv_csv['distances'].round(2),unit='s')
    #Set the time_distance to the index
    indiv_csv.set_index('time_distance',inplace=True)
    #Apply rolling window by translating the distance into second equivalent.
    indiv_csv['rolling_mean_ice_thickness'] = indiv_csv.rolling(str(window_distance)+'s',center=True,closed="both")["20m_ice_content_m"].mean()
    indiv_csv['rolling_std_ice_thickness'] = indiv_csv.rolling(str(window_distance)+'s',center=True,closed="both")["20m_ice_content_m"].std() 
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
print(Transects_2017_2018.rolling_CV_ice_thickness.quantile([0.05,0.25,0.5,0.75,0.95]).round(2))

#Compare the coefficient of variation with an hypothetical ice slabs transect:
#ice slabs transect 20 km long, whose lowermost point is 16 m thick and uppermost one is 1 m thick. Adding a noise ranging from -1 to +1 to this hypothetical ice slab thickness transect. 14 because ice slabs sampling points are roughly 14 m apart from each other
Hypothetical_IceSlabs_Transect=pd.DataFrame(data=np.linspace(16,1,int(20000/14+1))+np.random.randint(-10,11,len(np.linspace(16,1,int(20000/14+1))))/10,columns=["ice_thickness"])
Hypothetical_IceSlabs_Transect['time_distance']=pd.to_datetime(np.arange(0,20000,14),unit='s')
#Set the time_distance to the index
Hypothetical_IceSlabs_Transect.set_index('time_distance',inplace=True)

#Apply rolling window by translating the distance into second equivalent.
Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness'] = Hypothetical_IceSlabs_Transect.rolling(str(window_distance)+'s',center=True,closed="both")["ice_thickness"].mean()
Hypothetical_IceSlabs_Transect['rolling_std_ice_thickness'] = Hypothetical_IceSlabs_Transect.rolling(str(window_distance)+'s',center=True,closed="both")["ice_thickness"].std()
Hypothetical_IceSlabs_Transect['rolling_CV_ice_thickness'] = Hypothetical_IceSlabs_Transect['rolling_std_ice_thickness']/Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness']
Hypothetical_IceSlabs_Transect['ice_thickness_MINUS_rolling_mean_ice_thickness'] = Hypothetical_IceSlabs_Transect['ice_thickness']-Hypothetical_IceSlabs_Transect['rolling_mean_ice_thickness']

#Reset index to integer
Hypothetical_IceSlabs_Transect['new_index']=np.arange(0,len(Hypothetical_IceSlabs_Transect))
Hypothetical_IceSlabs_Transect.set_index('new_index',inplace=True)

print('--- Hypothetical ice slab ---')
print(Hypothetical_IceSlabs_Transect.rolling_CV_ice_thickness.quantile([0.05,0.25,0.5,0.75,0.95]).round(2))
#The spatial variability of the 2017/2018 ice slabs dtatset is about twice as the one from the hypothetical ice slabs transect.
#Note that in the real world dataset, there are longitudinal transects! But these ones are expected to have an even lower variability if we consider no other influence that melting gradient due to elevation gradient.

#Display the hypothetical ice slab
fig = plt.figure(figsize=(12,2))
gs = gridspec.GridSpec(2, 6)
ax1= plt.subplot(gs[0:2, 0:6])

ax1.fill_between(np.arange(0,len(Hypothetical_IceSlabs_Transect)),Hypothetical_IceSlabs_Transect["rolling_mean_ice_thickness"])
ax1.set_ylim(16,0)

###############################################################################
###                   Ice Thickness spatial heterogeneity                   ###
###############################################################################

pdb.set_trace()

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
                      stat="density",log_scale=[False,True],bins=np.arange(0,100,5),ax=ax_hist)
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


