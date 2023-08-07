# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:14:20 2023

@author: jullienn
"""

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

###############################################################################
###                       CumHydro and Ice Thickness                        ###
###############################################################################
################## Load ice slabs with Cumulative hydro dataset ################
#Path to data
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_CumHydro_And_IceThickness=path_local+'CumHydro_and_IceThickness/csv/'
#List all the files in the folder
list_composite=os.listdir(path_CumHydro_And_IceThickness) #this is inspired from https://pynative.com/python-list-files-in-a-directory/
#Define empty dataframes
upsampled_CumHydro_and_IceSlabs=pd.DataFrame()

#Loop over all the files
for indiv_file in list_composite:
    
    if (indiv_file == 'NotClipped_With0mSlabs'):
        continue
    
    print(indiv_file)
    #Open the individual file
    indiv_csv=pd.read_csv(path_CumHydro_And_IceThickness+indiv_file)
    ### ALL ###
    #Upsample data: where index_right is identical (i.e. for each CumHydro cell), keep a single value of CumHydro and average the ice content
    indiv_upsampled_CumHydro_and_IceSlabs=indiv_csv.groupby('index_right').mean()
    #Append the data to each other
    upsampled_CumHydro_and_IceSlabs=pd.concat([upsampled_CumHydro_and_IceSlabs,indiv_upsampled_CumHydro_and_IceSlabs])
    ### ALL ###
################## Load ice slabs with Cumulative hydro dataset ################

#Reset a new index to this upsampled dataset
upsampled_CumHydro_and_IceSlabs["index"]=np.arange(0,len(upsampled_CumHydro_and_IceSlabs))
upsampled_CumHydro_and_IceSlabs.set_index('index',inplace=True)
#Drop Unnamed: 0 columns
upsampled_CumHydro_and_IceSlabs.drop(columns=['Unnamed: 0'],inplace=True)
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
