# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:01:33 2022

@author: jullienn
"""

#The fuction plot_histo is from Emax_SlabsThickness.py
def plot_histo(ax_plot,iceslabs_above,iceslabs_within,iceslabs_inbetween,region):
    
    if (region == 'GrIS'):
        ax_plot.hist(iceslabs_above['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_inbetween['20m_ice_content_m'],color='yellow',label='In Between',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_inbetween['20m_ice_content_m'],0.5),linestyle='--',color='yellow')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.nanquantile(iceslabs_inbetween['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='yellow')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        
    else:
        ax_plot.hist(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_inbetween[iceslabs_inbetween['key_shp']==region]['20m_ice_content_m'],color='yellow',label='In Between',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_inbetween[iceslabs_inbetween['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='yellow')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.nanquantile(iceslabs_inbetween[iceslabs_inbetween['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='yellow')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    
    #Set x lims
    ax_plot.set_xlim(-0.5,20)

    if (region == 'NW'):
        ax_plot.legend()
    
    if (region in list(['NO','NE','GrIS'])):
        ax_plot.yaxis.tick_right()#This is from Fig4andS6andS7.py from paper 'Greenland Ice Slabs Expansion and Thickening'
    
    if (region in list(['NW','NO','CW','NE'])):
        ax_plot.set_xticklabels([])

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

radius=250

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'InBetween': "#fee391"}

#Generate boxplot and distributions using 2012, 2016 and 2019 as one population
path_load_data='C:/Users/jullienn/switchdrive/Private/research/RT3/data/'

#Define empty dataframe
iceslabs_within_121619=pd.DataFrame()
iceslabs_inbetween_121619=pd.DataFrame()
iceslabs_above_121619=pd.DataFrame()

#Loop over the years
for indiv_year in list([2012,2016,2019]):

    #Load data
    iceslabs_within_load=pd.read_csv(path_load_data+'iceslabs_masked_within_Emax_'+str(indiv_year)+'_cleanedxytpdV2_2years.csv')
    iceslabs_inbetween_load=pd.read_csv(path_load_data+'iceslabs_masked_inbetween_Emax_'+str(indiv_year)+'_cleanedxytpdV2_2years.csv')
    iceslabs_above_load=pd.read_csv(path_load_data+'iceslabs_masked_above_Emax_'+str(indiv_year)+'_cleanedxytpdV2_2years.csv')
    
    #Append data to each other
    iceslabs_within_121619=pd.concat([iceslabs_within_121619,iceslabs_within_load])
    iceslabs_inbetween_121619=pd.concat([iceslabs_inbetween_121619,iceslabs_inbetween_load])
    iceslabs_above_121619=pd.concat([iceslabs_above_121619,iceslabs_above_load])


######################## Plot with 0m thick ice slabs #########################
#Display ice slabs distributions as a function of the regions - This is fromEmax_SLabsThickness.py
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
plot_histo(axNW,iceslabs_above_121619,iceslabs_within_121619,iceslabs_inbetween_121619,'NW')
plot_histo(axCW,iceslabs_above_121619,iceslabs_within_121619,iceslabs_inbetween_121619,'CW')
plot_histo(axSW,iceslabs_above_121619,iceslabs_within_121619,iceslabs_inbetween_121619,'SW')
plot_histo(axNO,iceslabs_above_121619,iceslabs_within_121619,iceslabs_inbetween_121619,'NO')
plot_histo(axNE,iceslabs_above_121619,iceslabs_within_121619,iceslabs_inbetween_121619,'NE')
plot_histo(axGrIS,iceslabs_above_121619,iceslabs_within_121619,iceslabs_inbetween_121619,'GrIS')

#Finalise plot
axSW.set_xlabel('Ice content [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2012-16-19 - 2 years running slabs')
plt.show()

#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Histo_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)

#Display as boxplots
#Aggregate data together
iceslabs_above_121619['type']=['Above']*len(iceslabs_above_121619)
iceslabs_within_121619['type']=['Within']*len(iceslabs_within_121619)
iceslabs_inbetween_121619['type']=['InBetween']*len(iceslabs_inbetween_121619)
iceslabs_boxplot=pd.concat([iceslabs_above_121619,iceslabs_inbetween_121619,iceslabs_within_121619])

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
fig.suptitle('2012-16-19 - 2 years running slabs')


#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Boxplot_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)

######################## Plot with 0m thick ice slabs #########################

####################### Plot without 0m thick ice slabs #######################
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
           iceslabs_above_121619[iceslabs_above_121619['20m_ice_content_m']>0],
           iceslabs_within_121619[iceslabs_within_121619['20m_ice_content_m']>0],
           iceslabs_inbetween_121619[iceslabs_inbetween_121619['20m_ice_content_m']>0],
           'NW')
plot_histo(axCW,
           iceslabs_above_121619[iceslabs_above_121619['20m_ice_content_m']>0],
           iceslabs_within_121619[iceslabs_within_121619['20m_ice_content_m']>0],
           iceslabs_inbetween_121619[iceslabs_inbetween_121619['20m_ice_content_m']>0],
           'CW')
plot_histo(axSW,
           iceslabs_above_121619[iceslabs_above_121619['20m_ice_content_m']>0],
           iceslabs_within_121619[iceslabs_within_121619['20m_ice_content_m']>0],
           iceslabs_inbetween_121619[iceslabs_inbetween_121619['20m_ice_content_m']>0],
           'SW')
plot_histo(axNO,
           iceslabs_above_121619[iceslabs_above_121619['20m_ice_content_m']>0],
           iceslabs_within_121619[iceslabs_within_121619['20m_ice_content_m']>0],
           iceslabs_inbetween_121619[iceslabs_inbetween_121619['20m_ice_content_m']>0],
           'NO')
plot_histo(axNE,
           iceslabs_above_121619[iceslabs_above_121619['20m_ice_content_m']>0],
           iceslabs_within_121619[iceslabs_within_121619['20m_ice_content_m']>0],
           iceslabs_inbetween_121619[iceslabs_inbetween_121619['20m_ice_content_m']>0],
           'NE')
plot_histo(axGrIS,
           iceslabs_above_121619[iceslabs_above_121619['20m_ice_content_m']>0],
           iceslabs_within_121619[iceslabs_within_121619['20m_ice_content_m']>0],
           iceslabs_inbetween_121619[iceslabs_inbetween_121619['20m_ice_content_m']>0],
           'GrIS')

#Finalise plot
axSW.set_xlabel('Ice content [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2012-16-19 - 2 years running slabs - 0m thick slabs excluded')
plt.show()

#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/HistoNonZeros_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)


#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.boxplot(data=iceslabs_boxplot_region_GrIS[iceslabs_boxplot_region_GrIS['20m_ice_content_m']>0], x="20m_ice_content_m", y="key_shp",hue="type",orient="h",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_ylabel('')
ax_regions_GrIS.set_xlabel('Ice content [m]')
ax_regions_GrIS.set_xlim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle('2012-16-19 - 2 years running slabs - 0m thick slabs excluded')


#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/BoxplotNonZeros_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)


####################### Plot without 0m thick ice slabs #######################

