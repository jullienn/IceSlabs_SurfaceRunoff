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
        #Display sample size
        print(region,':\n')
        print('   Above: ',len(iceslabs_above),'\n')
        print('   Within: ',len(iceslabs_within),'\n')
        
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
        
        #Display sample size
        print(region,':\n')
        print('   Above: ',len(iceslabs_above[iceslabs_above['key_shp']==region]),'\n')
        print('   Within: ',len(iceslabs_within[iceslabs_within['key_shp']==region]),'\n')

    #Set x lims
    ax_plot.set_xlim(-0.5,20)

    if (region == 'NW'):
        ax_plot.legend()
    
    if (region in list(['NO','NE','GrIS'])):
        ax_plot.yaxis.tick_right()#This is from Fig4andS6andS7.py from paper 'Greenland Ice Slabs Expansion and Thickening'
    
    if (region in list(['NW','NO','CW','NE'])):
        ax_plot.set_xticklabels([])

    return


#The fuction plot_histo is from Emax_SlabsThickness.py
def plot_histo_EGU(ax_plot,iceslabs_above,iceslabs_within,region):
    
    if (region == 'GrIS'):
        ax_plot.hist(iceslabs_above['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Display sample size
        print(region,':\n')
        print('   Above: ',len(iceslabs_above),'\n')
        print('   Within: ',len(iceslabs_within),'\n')
        
    else:
        ax_plot.hist(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        
        #Display sample size
        print(region,':\n')
        print('   Above: ',len(iceslabs_above[iceslabs_above['key_shp']==region]),'\n')
        print('   Within: ',len(iceslabs_within[iceslabs_within['key_shp']==region]),'\n')

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

#Improve this code to display boxplots at VS above VS below MVRL but also SAR!!

#Extract relationship 

radius=250

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'InBetween': "#fee391"}

#Generate boxplot and distributions using 2012, 2016 and 2019 as one population
path_load_data='C:/Users/jullienn/switchdrive/Private/research/RT3/data/extracted_slabs/'

#Define empty dataframe
iceslabs_within_121619=pd.DataFrame()
iceslabs_inbetween_121619=pd.DataFrame()
iceslabs_above_121619=pd.DataFrame()

#Loop over the years
#for indiv_year in list([2012,2016,2019]):
for indiv_year in list([2019]):

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

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Histo_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''

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

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Boxplot_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''
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
'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/HistoNonZeros_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''

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

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/BoxplotNonZeros_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''

####################### Plot without 0m thick ice slabs #######################






####################### Plot 2019 above and within #######################
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
plot_histo_EGU(axNW,iceslabs_above_121619,iceslabs_within_121619,'NW')
plot_histo_EGU(axCW,iceslabs_above_121619,iceslabs_within_121619,'CW')
plot_histo_EGU(axSW,iceslabs_above_121619,iceslabs_within_121619,'SW')
plot_histo_EGU(axNO,iceslabs_above_121619,iceslabs_within_121619,'NO')
plot_histo_EGU(axNE,iceslabs_above_121619,iceslabs_within_121619,'NE')
plot_histo_EGU(axGrIS,iceslabs_above_121619,iceslabs_within_121619,'GrIS')

#Finalise plot
axSW.set_xlabel('Ice slabs thickness [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2019 MVRL - 2017-18 ice slabs thicknesses')
plt.show()

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Histo_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''

#Display as boxplots
#Aggregate data together
iceslabs_above_121619['type']=['Above']*len(iceslabs_above_121619)
iceslabs_within_121619['type']=['Within']*len(iceslabs_within_121619)
iceslabs_boxplot=pd.concat([iceslabs_above_121619,iceslabs_within_121619])

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
fig.suptitle('2019 MVRL - 2017-18 ice slabs thicknesses')


'''
#Save the figure
plt.savefig('C:/Users/jullienn/switchdrive/Private/research/conferences/EGU2023/figures/2019MVRL_above_within_EGU.png',dpi=500,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''
####################### Plot 2019 above and within #######################


############################# Plot 2019 SAR #############################

path_SAR='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/SAR_sectors/'

above_all=[]
within_all=[]
below_all=[]

for indiv_box in range(0,40):
    #open above
    try:
        above = np.asarray(pd.read_csv(path_SAR+'above/SAR_above_box_'+str(indiv_box)+'_2019.txt', header=None))
        #Append data
        above_all=np.append(above_all,above)
    except FileNotFoundError:
        print('No above')
    
    #open within
    try:
        within = np.asarray(pd.read_csv(path_SAR+'within/SAR_within_box_'+str(indiv_box)+'_2019.txt', header=None))
        #Append data
        within_all=np.append(within_all,within)
    except FileNotFoundError:
        print('No within')
    
    #open below
    try:
        below = np.asarray(pd.read_csv(path_SAR+'below/SAR_below_box_'+str(indiv_box)+'_2019.txt', header=None))
        #Append data
        below_all=np.append(below_all,below)
    except FileNotFoundError:
        print('No below')

#Display figure distribution
fig, (ax_distrib) = plt.subplots()      
ax_distrib.hist(below_all,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='green',label='Below')
#ax_distrib.hist(within_all,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='red',label='Within')
ax_distrib.hist(above_all,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='blue',label='Above')
ax_distrib.set_xlim(-20,-2)
ax_distrib.set_xlabel('Signal strength [dB]')
ax_distrib.set_ylabel('Density')
ax_distrib.legend()

#Display boxplot
df_below_all=pd.DataFrame(below_all,columns=['signal'])
df_below_all['cat']=['below']*len(df_below_all)
'''
df_within_all=pd.DataFrame(within_all,columns=['signal'])
df_within_all['cat']=['within']*len(df_within_all)
'''
df_above_all=pd.DataFrame(above_all,columns=['signal'])
df_above_all['cat']=['above']*len(df_above_all)
SAR_boxplot_GrIS=pd.concat([df_below_all,df_above_all])

#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_SAR = plt.subplot(gs[0:10, 0:6])
sns.boxplot(data=SAR_boxplot_GrIS, x="cat", y="signal",ax=ax_SAR)#, kde=True)
ax_SAR.set_ylabel('Signal strength [dB]')
ax_SAR.set_xlabel('Category')

pdb.set_trace()
############################# Plot 2019 SAR #############################


############################# 2019 binary map #############################
#Choose cutoff signal

#Apply binary cutoff

#Display map of efficient aquitard with 2019 MVRL


############################# 2019 binary map #############################




#Once all csv files of SAR extraction are performed, load them

#Define a list of data where the relationship could be ideal
list_ideal=['20170421_01_006_009.csv', '20170421_01_171_174.csv','20170502_01_171_173.csv',
              '20170505_02_008_010.csv', '20170505_02_181_183.csv', '20170506_01_010_012.csv',
              '20170508_02_011_013.csv', '20170508_02_165_171.csv',
              '20170511_01_010_025.csv', '20170511_01_176_178.csv', '20180421_01_004_007.csv',
              '20180423_01_180_182.csv', '20180425_01_005_008.csv',
              '20180425_01_166_169.csv', '20180427_01_004_006.csv',
              '20180427_01_170_172.csv', '20180429_01_008_014.csv']


if (load_data=='TRUE'):
    #Path to data
    path_csv_SAR_VS_IceContent='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/csv/'
    
    #List all the files in the folder
    list_composite=os.listdir(path_csv_SAR_VS_IceContent) #this is inspired from https://pynative.com/python-list-files-in-a-directory/
    '''
    list_composite=list_ideal
    '''
    #Loop over all the files
    for indiv_file in list_composite:
        
        #Open the individual file
        indiv_csv=pd.read_csv(path_csv_SAR_VS_IceContent+indiv_file)
        
        #If first file of the list, create the concatenated pandas dataframe. If not, fill it in
        if (indiv_file==list_composite[0]):
            appended_df=indiv_csv.copy(deep=True)
        else:
            #Append the data to each other
            appended_df=pd.concat([appended_df,indiv_csv])
    
    #Create a unique indey for each line, and set this new vector as the index in dataframe
    appended_df['index_unique']=np.arange(0,len(appended_df))
    appended_df=appended_df.set_index('index_unique')
    
    #Display some descriptive statistics
    appended_df.describe()['radar_signal']
    appended_df.describe()['20m_ice_content_m']

pdb.set_trace()
#Display the composite relationship using all the files
if (composite=='TRUE'):
    
    #7. Plot the overall relationship SAR VS Ice slabs thickness
    #Prepare plot
    fig = plt.figure()
    fig.set_size_inches(14, 8) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=2)
    gs.update(hspace=1)
    ax_SAR = plt.subplot(gs[0:2, 0:8])
    ax_scatter = plt.subplot(gs[2:10, 0:8])
    ax_IceContent = plt.subplot(gs[2:10, 8:10])
    
    #Display
    appended_df.plot.scatter(x='radar_signal',y='20m_ice_content_m',ax=ax_scatter)
    ax_scatter.set_xlim(-18,1)
    ax_scatter.set_ylim(-0.5,20.5)
    ax_scatter.set_xlabel('SAR [dB]')
    ax_scatter.set_ylabel('Ice content [m]')

    ax_IceContent.hist(appended_df['20m_ice_content_m'],
                       bins=np.arange(np.min(appended_df['20m_ice_content_m']),np.max(appended_df['20m_ice_content_m'])),
                       density=True,orientation='horizontal')
    ax_IceContent.set_xlabel('Density [ ]')
    ax_IceContent.set_ylim(-0.5,20.5)

    ax_SAR.hist(appended_df['radar_signal'],
                bins=np.arange(np.min(appended_df['radar_signal']),np.max(appended_df['radar_signal'])),
                density=True)
    ax_SAR.set_xlim(-18,1)
    ax_SAR.set_ylabel('Density [ ]')
    fig.suptitle('Ice content and SAR')
    
    ###########################################################################
    ###        Keep only where ice content is: 0 < ice content < 15m        ###
    ###########################################################################
    restricted_appended_df=appended_df.copy()
    restricted_appended_df=restricted_appended_df[np.logical_and(restricted_appended_df['20m_ice_content_m']>0,restricted_appended_df['20m_ice_content_m']<15)]
    
    #Prepare plot
    fig = plt.figure()
    fig.set_size_inches(14, 8) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=2)
    gs.update(hspace=1)
    ax_SAR = plt.subplot(gs[0:2, 0:8])
    ax_scatter = plt.subplot(gs[2:10, 0:8])
    ax_IceContent = plt.subplot(gs[2:10, 8:10])
    
    #Display
    restricted_appended_df.plot.scatter(x='radar_signal',y='20m_ice_content_m',ax=ax_scatter)
    ax_scatter.set_xlim(-18,1)
    ax_scatter.set_ylim(-0.5,20.5)
    ax_scatter.set_xlabel('SAR [dB]')
    ax_scatter.set_ylabel('Ice content [m]')

    ax_IceContent.hist(restricted_appended_df['20m_ice_content_m'],
                       bins=np.arange(np.min(restricted_appended_df['20m_ice_content_m']),np.max(restricted_appended_df['20m_ice_content_m'])),
                       density=True,orientation='horizontal')
    ax_IceContent.set_xlabel('Density [ ]')
    ax_IceContent.set_ylim(-0.5,20.5)

    ax_SAR.hist(restricted_appended_df['radar_signal'],
                bins=np.arange(np.min(restricted_appended_df['radar_signal']),np.max(restricted_appended_df['radar_signal'])),
                density=True)
    ax_SAR.set_xlim(-18,1)
    ax_SAR.set_ylabel('Density [ ]')
    fig.suptitle('0 < ice content < 15 m and SAR')
    ###########################################################################
    ###        Keep only where ice content is: 0 < ice content < 15m        ###
    ###########################################################################
    '''
    import seaborn as sns
    #Not working, probably because dataset too large
    sns.displot(data=appended_df, x="radar_signal", col="20m_ice_content_m", kde=True)
    sns.jointplot(data=appended_df, x="radar_signal", y="20m_ice_content_m")
    '''    
    
    #There are places where the radar signal is a NaN. Extract it
    appended_df_no_NaN=appended_df.copy()
    appended_df_no_NaN=appended_df_no_NaN[~pd.isna(appended_df_no_NaN['radar_signal'])]
    appended_df_no_NaN=appended_df_no_NaN[~pd.isna(appended_df_no_NaN['20m_ice_content_m'])]

    restricted_appended_df_no_NaN=restricted_appended_df.copy()
    restricted_appended_df_no_NaN=restricted_appended_df_no_NaN[~pd.isna(restricted_appended_df_no_NaN['radar_signal'])]
    
    #Display
    fig_heatmap = plt.figure()
    fig_heatmap.set_size_inches(14, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=3)
    gs.update(hspace=3)
    ax_hist2d = plt.subplot(gs[0:5, 0:5])
    ax_hist2d_restricted = plt.subplot(gs[0:5, 5:10])
    ax_hist2d_log = plt.subplot(gs[5:10, 0:5])
    ax_hist2d_restricted_log = plt.subplot(gs[5:10, 5:10])
    import matplotlib as mpl
    
    hist2d_cbar = ax_hist2d.hist2d(appended_df_no_NaN['radar_signal'],appended_df_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r')
    hist2d_log_cbar = ax_hist2d_log.hist2d(appended_df_no_NaN['radar_signal'],appended_df_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r',norm=mpl.colors.LogNorm())

    hist2d_restricted_cbar = ax_hist2d_restricted.hist2d(restricted_appended_df_no_NaN['radar_signal'],restricted_appended_df_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r')
    hist2d_restricted_log_cbar = ax_hist2d_restricted_log.hist2d(restricted_appended_df_no_NaN['radar_signal'],restricted_appended_df_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r',norm=mpl.colors.LogNorm())
                     #cmin=np.quantile(h[0].flatten(),0.5),cmax=np.max(h[0].flatten()))#density=True
    #ax_hist2d.set_xlim(-18,1)
    ax_hist2d_log.set_xlabel('SAR [dB]')
    ax_hist2d_log.set_ylabel('Ice content [m]')
    ax_hist2d.set_title('0 < ice content < 20 m')
    ax_hist2d_restricted.set_title('0 < ice content < 15 m')
    fig_heatmap.suptitle('Occurrence map')

    ax_hist2d.set_ylim(0,20)
    ax_hist2d_log.set_ylim(0,20)
    ax_hist2d_restricted.set_ylim(0,20)
    ax_hist2d_restricted_log.set_ylim(0,20)
    ax_hist2d_restricted.set_xlim(-16.5,-4)
    ax_hist2d_restricted_log.set_xlim(-16.5,-4)
    
    #Display colorbars    
    fig_heatmap.colorbar(hist2d_cbar[3], ax=ax_hist2d,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot
    fig_heatmap.colorbar(hist2d_log_cbar[3], ax=ax_hist2d_log,label='log(Occurence)')
    fig_heatmap.colorbar(hist2d_restricted_cbar[3], ax=ax_hist2d_restricted,label='Occurence')
    fig_heatmap.colorbar(hist2d_restricted_log_cbar[3], ax=ax_hist2d_restricted_log,label='log(Occurence)')
    
    '''
    #Fit a polynomial to this relationship using numpy.polyfit
    pol_fit=np.polynomial.polynomial.Polynomial.fit(restricted_appended_df['radar_signal'], restricted_appended_df['20m_ice_content_m'], 2,domain=[-12,0])
    
    x_to_fit=np.arange(pol_fit.domain[0],pol_fit.domain[1])
    fitted=np.polynomial.polynomial.polyval(x_to_fit,pol_fit.coef)
    
    #Display the pèolynomial fit
    ax_hist2d.plot(x_to_fit,fitted)
    '''
    #sort restricted_appended_df
    restricted_appended_df_no_NaN=restricted_appended_df_no_NaN.sort_values(by=['radar_signal'])
    
    #prepare data for fit        
    xdata = np.array(restricted_appended_df_no_NaN['radar_signal'])
    ydata = np.array(restricted_appended_df_no_NaN['20m_ice_content_m'])                           
                           
    #manual fit
    ax_hist2d_restricted.plot(xdata, func(xdata, 1,0.26,-2),'r',label='manual fit: y = 1*exp(-0.26*x) - 2')
        
    #automatic fit: try with scipy.curve_fit - following the example on the help page
    #popt, pcov = curve_fit(func, xdata, ydata,p0=[1,0.26,-2],bounds=([-2,0,-5],[5,2,2]))
    popt, pcov = curve_fit(func, xdata, ydata,p0=[1,0.26,-2],bounds=([0,-1,-4],[2,1,2]))
    ax_hist2d_restricted.plot(xdata, func(xdata, *popt),'b',label='automatic fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))#, 'r-'
    ax_hist2d_restricted.legend(loc='upper left',fontsize=6)
    
    pdb.set_trace()
    '''
    #Save figure
    plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/relationship/relationship_SAR_IceContent.png',dpi=300,bbox_inches='tight')
    '''
    #Define binning range
    range_binning=np.array([[restricted_appended_df_no_NaN['radar_signal'].min(),restricted_appended_df_no_NaN['radar_signal'].max()],
                            [restricted_appended_df_no_NaN['20m_ice_content_m'].min(),restricted_appended_df_no_NaN['20m_ice_content_m'].max()]])
    
    ###########################################################################
    ###           Get rid of data points where occurrence is low!           ###
    ###########################################################################

    #Define a dataset for regression calculation
    restricted_appended_df_no_NaN_regression=restricted_appended_df_no_NaN.copy()
    
    #Get rid of low occurence grid cells
    n=500#let's start by excluding drig cell whose occurence is lower than 100 individuals - should have a good reason behind this value!!    
    
    #Extract occurrence matrix and related x and y vectors
    occurence_matrix=hist2d_restricted_cbar[0]
    rows=hist2d_restricted_cbar[1]
    cols=hist2d_restricted_cbar[2]
    
    #Define empty vector for index store
    index_removal=[]
    
    #Loop to identify grid cell to delete
    for index_row in range(0,len(rows)-1):
        bounds_row=[rows[index_row],rows[index_row+1]] #row is SAR
        for index_col in range(0,len(cols)-1):
            bounds_col=[cols[index_col],cols[index_col+1]] #row is ice content
            #Does this SAR and ice content grid cell has less than n occurence? If yes, delete these data
            if (occurence_matrix[index_row,index_col]<n):
                #select data that respect both the SAR and ice content bounds
                logical_SAR=np.logical_and(restricted_appended_df_no_NaN['radar_signal']>=bounds_row[0],restricted_appended_df_no_NaN['radar_signal']<bounds_row[1])
                logical_ice_content=np.logical_and(restricted_appended_df_no_NaN['20m_ice_content_m']>=bounds_col[0],restricted_appended_df_no_NaN['20m_ice_content_m']<bounds_col[1])
                #combine logical vectors
                logical_combined=np.logical_and(logical_SAR,logical_ice_content)
                if (logical_combined.astype(int).sum()>0):
                    #store the index
                    index_removal=np.append(index_removal,np.array(logical_combined[logical_combined].index))
                    #display
                    '''
                    ax_hist2d_restricted.scatter(restricted_appended_df_no_NaN['radar_signal'].loc[np.array(logical_combined[logical_combined].index)],
                                                 restricted_appended_df_no_NaN['20m_ice_content_m'].loc[np.array(logical_combined[logical_combined].index)])
                    '''
    #get rid of corresponding data
    restricted_appended_df_no_NaN_regression=restricted_appended_df_no_NaN_regression.drop(index=np.unique(index_removal),axis=1)
    
    #Display resulting grid
    fig_regression = plt.figure()
    fig_regression.set_size_inches(7, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 5)
    gs.update(wspace=3)
    gs.update(hspace=3)
    ax_hist2d_regression = plt.subplot(gs[0:5, 0:5])
    ax_hist2d_regression_log = plt.subplot(gs[5:10,0:5])
    
    hist2d_regression_cbar = ax_hist2d_regression.hist2d(restricted_appended_df_no_NaN_regression['radar_signal'],restricted_appended_df_no_NaN_regression['20m_ice_content_m'],cmap='magma_r',bins=30,range=range_binning)
    hist2d_regression_log_cbar = ax_hist2d_regression_log.hist2d(restricted_appended_df_no_NaN_regression['radar_signal'],restricted_appended_df_no_NaN_regression['20m_ice_content_m'],cmap='magma_r',bins=30,norm=mpl.colors.LogNorm())
    
    ax_hist2d_regression_log.set_xlabel('SAR [dB]')
    ax_hist2d_regression_log.set_ylabel('Ice content [m]')
    ax_hist2d_regression.set_title('0 < ice content < 15 m')
    
    ax_hist2d_regression.set_ylim(0,20)
    ax_hist2d_regression_log.set_ylim(0,20)
    ax_hist2d_regression.set_xlim(-16.5,-4)
    ax_hist2d_regression_log.set_xlim(-16.5,-4)    

    #Calculate regression
    xdata = np.array(restricted_appended_df_no_NaN_regression['radar_signal'])
    ydata = np.array(restricted_appended_df_no_NaN_regression['20m_ice_content_m'])                           
    
    #manual fit
    ax_hist2d_regression.plot(xdata, func(xdata, 1,0.26,-2),'r',label='manual fit: y = 1*exp(-0.26*x) - 2')
    
    #automatic fit
    popt, pcov = curve_fit(func, xdata, ydata,p0=[1,0.26,-2],bounds=([-2,0,-5],[5,2,2]))
    ax_hist2d_regression.legend(loc='upper left',fontsize=8)

    ax_hist2d_regression.plot(xdata, func(xdata, *popt), label='fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))
    ax_hist2d_restricted.plot(xdata, func(xdata, *popt), label='%i filetered, fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(np.append(n,popt)))
    ax_hist2d_restricted.legend(loc='upper left',fontsize=8)
    
    #Display colorbars    
    fig_regression.colorbar(hist2d_regression_cbar[3], ax=ax_hist2d_regression,label="Occurrence")
    fig_regression.colorbar(hist2d_regression_log_cbar[3], ax=ax_hist2d_regression_log,label="log(Occurrence)")
    
    #"Compute one standard deviation errors on the parameters":
    perr = np.sqrt(np.diag(pcov))
    #popt gives: "Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized."
    
    pdb.set_trace()
    '''
    plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/relationship/relationship_SAR_IceContent_occurence='+str(n)+'.png',dpi=300,bbox_inches='tight')
    '''
    
    ###########################################################################
    ###           Get rid of data points where occurrence is low!           ###
    ###########################################################################


#Consider a test dataset, derive the relationship between SAR and ice content, predict ice content from SAR, evaluate the performance of the prediction
if (interpolation=='TRUE'):
    print('Performing the interpolation')
    
    #Get rid of data where NaNs
    appended_df_no_NaNs=appended_df[~appended_df['radar_signal'].isna()].copy()
    '''
    #Select only ice content lower than 7.5m
    appended_df_no_NaNs=appended_df_no_NaNs[appended_df_no_NaNs['20m_ice_content_m']<7.5].copy()
    '''
    #Prepare figure to display
    fig_selection = plt.figure()
    fig_selection.set_size_inches(14, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=3)
    gs.update(hspace=3)
    ax_whole_df = plt.subplot(gs[0:5, 0:5],projection=crs)
    ax_selected_df = plt.subplot(gs[0:5, 5:10],projection=crs)
    ax_whole_df_distrib = plt.subplot(gs[5:10, 0:5])
    ax_selected_df_distrib = plt.subplot(gs[5:10, 5:10])
    
    #Display coastlines
    ax_whole_df.coastlines(edgecolor='black',linewidth=0.075)
    ax_selected_df.coastlines(edgecolor='black',linewidth=0.075)
    #Display whole df
    ax_whole_df.scatter(appended_df_no_NaNs.lon_3413,appended_df_no_NaNs.lat_3413)
    ax_whole_df_distrib.hist(appended_df_no_NaNs['20m_ice_content_m'])
        
    #1. Select randomly x% of the dataset
    randomly_selected_df=appended_df_no_NaNs.sample(frac=0.5).copy()
    
    #Sort data for relationship computation
    randomly_selected_df=randomly_selected_df.sort_values(by=['radar_signal']).copy()
    
    #Display
    ax_selected_df.scatter(randomly_selected_df.lon_3413,randomly_selected_df.lat_3413)
    ax_selected_df_distrib.hist(randomly_selected_df['20m_ice_content_m'])
    
    #2. Extract the relationship between SAR and ice content
    popt, pcov = curve_fit(func, np.array(randomly_selected_df['radar_signal']),
                           np.array(randomly_selected_df['20m_ice_content_m']),
                           p0=[1,0.26,-2],bounds=([-2,0,-5],[5,2,2]))
    
    #Display relationship
    fig_relationship, (ax_relationship) = plt.subplots()
    hist2d_cbar = ax_relationship.hist2d(randomly_selected_df['radar_signal'],randomly_selected_df['20m_ice_content_m'],bins=30,cmap='magma_r')
    ax_relationship.plot(np.array(randomly_selected_df['radar_signal']),
                         func(np.array(randomly_selected_df['radar_signal']), *popt),
                         label='fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))
    
    #manual fit
    ax_relationship.plot(np.array(randomly_selected_df['radar_signal']), func(np.array(randomly_selected_df['radar_signal']), 1,0.26,-2),
                         'r',label='manual fit: y = 1*exp(-0.26*x) - 2')
    
    ax_relationship.legend(loc='upper left',fontsize=8)
    
    #Try other relationships

    #manual fit
    a=np.arange(-1,0,0.1)
    b=np.arange(0.1,0.3,0.01)
    c=np.arange(-10,0,1)
    
    error_best=1e20
    err_vect=[]
    
    for i in range (0,len(a)):
        print(a[i])
        for j in range (0,len(b)):
            for k in range (0,len(c)):
                '''
                ax_relationship.plot(np.array(randomly_selected_df['radar_signal']), func(np.array(randomly_selected_df['radar_signal']), a[i],b[j],c[k]),
                                     label='manual fit: y = %f*exp(-%f*x) +%f' % tuple([a[i],b[j],c[k]]))
                '''
                #get the smallest error
                error_now=np.sum(np.power(randomly_selected_df['20m_ice_content_m']-func(np.array(randomly_selected_df['radar_signal']), a[i],b[j],c[k]),2))
                err_vect=np.append(err_vect,error_now)
                if(error_now<error_best):
                    error_best=error_now
                    a_best=a[i]
                    b_best=b[j]
                    c_best=c[k]
    
    #Display where min
    ax_relationship.plot(np.array(randomly_selected_df['radar_signal']), func(np.array(randomly_selected_df['radar_signal']), a_best,b_best,c_best),
                         'k',label='fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple([a_best,b_best,c_best]))
    ax_relationship.legend(loc='upper left',fontsize=8)
    plt.show()
    
    fig_err, (ax_err) = plt.subplots()
    ax_err.plot(np.arange(0,len(err_vect)),err_vect)
    ax_err.scatter(np.where(err_vect==err_vect.min())[0][0],err_vect.min(),color='r')
    pdb.set_trace()
        
    
    #3. Apply relationship to SAR
    appended_df_no_NaNs['predicted_ice_content']=func(np.array(appended_df_no_NaNs['radar_signal']), 1,0.26,-2)
    #Determine the absolute difference between predicted ice content and original ice content
    appended_df_no_NaNs['abs_diff_ice_content']=np.abs(appended_df_no_NaNs['predicted_ice_content']-appended_df_no_NaNs['20m_ice_content_m'])
    
    #4. Compare the results with the remaining dataset
    fig_err, (ax_err) = plt.subplots()
    ax_err.hist(appended_df_no_NaNs['abs_diff_ice_content'],bins=np.arange(0,60,1),cumulative=True)
    
    #Prepare figure to display
    fig_selection = plt.figure()
    ax_diff = plt.subplot(projection=crs)
    
    #Display coastlines
    ax_diff.coastlines(edgecolor='black',linewidth=0.075)
    ax_diff.scatter(appended_df_no_NaNs.lon_3413,appended_df_no_NaNs.lat_3413,c=appended_df_no_NaNs['abs_diff_ice_content'])
        

#Potential improvement: determine and use the best likelihood for each transect. !Might require quite some work!



