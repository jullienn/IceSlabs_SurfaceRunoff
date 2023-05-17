# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:01:33 2022

@author: jullienn
"""

#The fuction plot_histo is from Emax_SlabsThickness.py
def plot_histo(ax_plot,iceslabs_above,iceslabs_within,iceslabs_below,region):
    if (region == 'GrIS'):
        ax_plot.hist(iceslabs_above['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_below['20m_ice_content_m'],color='green',label='Below',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_below['20m_ice_content_m'],0.5),linestyle='--',color='green')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.nanquantile(iceslabs_below['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='green')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Display sample size
        print(region,':\n')
        print('   Above: ',len(iceslabs_above),'\n')
        print('   Within: ',len(iceslabs_within),'\n')
        print('   Below: ',len(iceslabs_below),'\n')
        
    else:
        ax_plot.hist(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],color='green',label='Below',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='green')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.nanquantile(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='green')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        
        #Display sample size
        print(region,':\n')
        print('   Above: ',len(iceslabs_above[iceslabs_above['key_shp']==region]),'\n')
        print('   Within: ',len(iceslabs_within[iceslabs_within['key_shp']==region]),'\n')
        print('   Below: ',len(iceslabs_below[iceslabs_below['key_shp']==region]),'\n')
    
    #Set x lims
    ax_plot.set_xlim(-0.5,20)

    if (region == 'NW'):
        ax_plot.legend()
    
    if (region in list(['NO','NE','GrIS'])):
        ax_plot.yaxis.tick_right()#This is from Fig4andS6andS7.py from paper 'Greenland Ice Slabs Expansion and Thickening'
    
    if (region in list(['NW','NO','CW','NE'])):
        ax_plot.set_xticklabels([])

    return


def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def deg_n(x, a, n):
    return a * np.power(x,n)


def keep_sectorial(df_input,indiv_trackname_tokeep):    
    df_output=df_input[df_input.Track_name==indiv_trackname_tokeep].copy()
    return df_output

def sector_association(indiv_df_SAR_IceThickness,indiv_df_sectors,sector):
    #Perform match between indiv_df_SAR_IceThickness and indiv_IceThickness_sectors 
    indiv_df_SAR_IceThickness_sector = indiv_df_SAR_IceThickness.merge(indiv_df_sectors, how="left",on=['lat','lon'],suffixes=('_gauche','_droite'))
    
    #drop useless columns
    indiv_df_SAR_IceThickness_sector=indiv_df_SAR_IceThickness_sector.drop(columns=['Unnamed: 0_droite', 'Track_name_droite', 'Tracenumber',
                                                          'alongtrack_distance_m', '20m_ice_content_m_droite',
                                                          'likelihood_droite', 'lat_3413_droite', 'lon_3413_droite',
                                                          'key_shp_droite', 'elevation_droite', 'year_droite', 'geometry',
                                                          'index_right_polygon', 'FID', 'rev_subs', 'index_right_droite'])
    
    #Get rid of data points which do not intersect with sector
    indiv_df_SAR_IceThickness_sector_noNaN=indiv_df_SAR_IceThickness_sector[~indiv_df_SAR_IceThickness_sector.type.isna()]
    
    if (len(indiv_df_SAR_IceThickness_sector_noNaN)>0):
        #Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
        indiv_upsampled_SAR_and_IceSlabs_sector=indiv_df_SAR_IceThickness_sector_noNaN.groupby('index_right_gauche').mean()
        #Add column of sector
        indiv_upsampled_SAR_and_IceSlabs_sector['sector']=[sector]*len(indiv_upsampled_SAR_and_IceSlabs_sector)
    else:
        indiv_upsampled_SAR_and_IceSlabs_sector=indiv_df_SAR_IceThickness_sector_noNaN
    
    return indiv_upsampled_SAR_and_IceSlabs_sector


def hist_regions(df_to_plot,region_to_plot,ax_region):
    ax_region.hist(df_to_plot[df_to_plot.sector=='Below']['SAR'],density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='green',label='Below')
    ax_region.hist(df_to_plot[df_to_plot.sector=='Above']['SAR'],density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='blue',label='Above')
    ax_region.set_xlim(-20,-2)
    ax_region.set_xlabel('Signal strength [dB]')
    ax_region.set_ylabel('Density')
    ax_region.text(-19.8,0.2,region_to_plot)

    return



def display_2d_histogram(df_to_display,FS_display):
    
    #Display 2D histogram
    fig_heatmap, ((ax_SW, ax_CW, ax_NW), (ax_NO, ax_NE, ax_GrIS)) = plt.subplots(2, 3)
    fig_heatmap.set_size_inches(14, 7) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen


    cbar_SW=ax_SW.hist2d(df_to_display[df_to_display.SUBREGION1=='SW']['radar_signal'],
                         df_to_display[df_to_display.SUBREGION1=='SW']['20m_ice_content_m_gauche'],bins=30,cmap='magma_r',cmin=1)
    ax_SW.set_ylim(0,16)
    ax_SW.set_xlim(-16,-3.5)
    ax_SW.set_title('SW')
    fig_heatmap.colorbar(cbar_SW[3], ax=ax_SW,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot

    
    cbar_CW=ax_CW.hist2d(df_to_display[df_to_display.SUBREGION1=='CW']['radar_signal'],
                         df_to_display[df_to_display.SUBREGION1=='CW']['20m_ice_content_m_gauche'],bins=30,cmap='magma_r',cmin=1)
    ax_CW.set_ylim(0,16)
    ax_CW.set_xlim(-16,-3.5)
    ax_CW.set_title('CW')
    fig_heatmap.colorbar(cbar_CW[3], ax=ax_CW,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot


    cbar_NW=ax_NW.hist2d(df_to_display[df_to_display.SUBREGION1=='NW']['radar_signal'],
                         df_to_display[df_to_display.SUBREGION1=='NW']['20m_ice_content_m_gauche'],bins=30,cmap='magma_r',cmin=1)
    ax_NW.set_ylim(0,16)
    ax_NW.set_xlim(-16,-3.5)
    ax_NW.set_title('NW')
    fig_heatmap.colorbar(cbar_NW[3], ax=ax_NW,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot


    cbar_NO=ax_NO.hist2d(df_to_display[df_to_display.SUBREGION1=='NO']['radar_signal'],
                         df_to_display[df_to_display.SUBREGION1=='NO']['20m_ice_content_m_gauche'],bins=30,cmap='magma_r',cmin=1)
    ax_NO.set_ylim(0,16)
    ax_NO.set_xlim(-16,-3.5)
    ax_NO.set_title('NO')
    fig_heatmap.colorbar(cbar_NO[3], ax=ax_NO,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot


    cbar_NE=ax_NE.hist2d(df_to_display[df_to_display.SUBREGION1=='NE']['radar_signal'],
                         df_to_display[df_to_display.SUBREGION1=='NE']['20m_ice_content_m_gauche'],bins=30,cmap='magma_r',cmin=1)
    ax_NE.set_ylim(0,16)
    ax_NE.set_xlim(-16,-3.5)
    ax_NE.set_title('NE')
    fig_heatmap.colorbar(cbar_NE[3], ax=ax_NE,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot


    cbar_GrIS=ax_GrIS.hist2d(df_to_display['radar_signal'],
                             df_to_display['20m_ice_content_m_gauche'],bins=30,cmap='magma_r',cmin=1)
    ax_GrIS.set_ylim(0,16)
    ax_GrIS.set_xlim(-16,-3.5)
    ax_GrIS.set_title('GrIS')
    fig_heatmap.colorbar(cbar_GrIS[3], ax=ax_GrIS,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot

    #Display firn cores ice content and SAR
    ax_SW.scatter(FS_display['SAR'],FS_display['10m_ice_content_%']/10,c='blue')
    
    #Set labels
    ax_NO.set_xlabel('SAR [dB]')
    ax_NO.set_ylabel('Ice thickness [m]')
    fig_heatmap.suptitle('Occurrence map')
    
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
import os
import matplotlib as mpl
from scipy.optimize import curve_fit

composite='TRUE'

#Define projection
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################


#Define palette for time , this is From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'Below': "green"}

#Generate boxplot and distributions using 2012, 2016 and 2019 as one population
path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceThickness/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

#Load IMBIE drainage bassins
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

###############################################################################
###         Load Ice Slabs Thickness data in the different sectors          ###
###############################################################################
#Define empty dataframe
IceThickness_above=pd.DataFrame()
IceThickness_in_between=pd.DataFrame()
IceThickness_within=pd.DataFrame()
IceThickness_below=pd.DataFrame()

for indiv_box in range(4,32):
    print(indiv_box)
    #open above
    try:
        above = pd.read_csv(path_data+'SAR_sectors/above/IceSlabs_above_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(above)>0):
            #Append data
            IceThickness_above=pd.concat([IceThickness_above,above])
    except FileNotFoundError:
        print('No above')
    
    #open InBetween
    try:
        in_between = pd.read_csv(path_data+'SAR_sectors/in_between/IceSlabs_in_between_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(in_between)>0):
            #Append data
            IceThickness_in_between=pd.concat([IceThickness_in_between,in_between])
    except FileNotFoundError:
        print('No in_between')
    
    #open within
    try:
        within = pd.read_csv(path_data+'SAR_sectors/within/IceSlabs_within_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(within)>0):
            #Append data
            IceThickness_within=pd.concat([IceThickness_within,within])
    except FileNotFoundError:
        print('No within')
    
    #open below
    try:
        below = pd.read_csv(path_data+'SAR_sectors/below/IceSlabs_below_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(below)>0):
            #Append data
            IceThickness_below=pd.concat([IceThickness_below,below])
    except FileNotFoundError:
        print('No below')

###############################################################################
###         Load Ice Slabs Thickness data in the different sectors          ###
###############################################################################

###############################################################################
###         Plot Ice Slabs Thickness data in the different sectors          ###
###############################################################################

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
plot_histo(axNW,IceThickness_above,IceThickness_within,IceThickness_below,'NW')
plot_histo(axCW,IceThickness_above,IceThickness_within,IceThickness_below,'CW')
plot_histo(axSW,IceThickness_above,IceThickness_within,IceThickness_below,'SW')
plot_histo(axNO,IceThickness_above,IceThickness_within,IceThickness_below,'NO')
plot_histo(axNE,IceThickness_above,IceThickness_within,IceThickness_below,'NE')
plot_histo(axGrIS,IceThickness_above,IceThickness_within,IceThickness_below,'GrIS')

#Finalise plot
axSW.set_xlabel('Ice Thickness [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2019 - 2 years running slabs')
plt.show()

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Histo_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''

#Display as boxplots
#Aggregate data together
IceThickness_above['type']=['Above']*len(IceThickness_above)
IceThickness_in_between['type']=['In_Between']*len(IceThickness_in_between)
IceThickness_within['type']=['Within']*len(IceThickness_within)
IceThickness_below['type']=['Below']*len(IceThickness_below)
IceThickness_all_sectors=pd.concat([IceThickness_above,IceThickness_within,IceThickness_below])

IceThickness_all_sectors_GrIS=IceThickness_all_sectors.copy(deep=True)
IceThickness_all_sectors_GrIS['key_shp']=['GrIS']*len(IceThickness_all_sectors)
IceThickness_all_sectors_region_GrIS=pd.concat([IceThickness_all_sectors,IceThickness_all_sectors_GrIS])

#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.boxplot(data=IceThickness_all_sectors_region_GrIS, x="20m_ice_content_m", y="key_shp",hue="type",orient="h",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_ylabel('')
ax_regions_GrIS.set_xlabel('Ice Thickness [m]')
ax_regions_GrIS.set_xlim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle('2019 - 2 years running slabs')


'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Boxplot_Emax_VS_IceSlabs_Masked_2019_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_with0mslabs.png',dpi=500)
'''

#Try violin plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.violinplot(data=IceThickness_all_sectors_region_GrIS, y="20m_ice_content_m", x="key_shp",hue="type",orient="v",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_xlabel('')
ax_regions_GrIS.set_ylabel('Ice Thickness [m]')
ax_regions_GrIS.set_ylim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle('2019 - 2 years running slabs')
######################## Plot with 0m thick ice slabs #########################

pdb.set_trace()

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
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'NW')
plot_histo(axCW,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'CW')
plot_histo(axSW,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'SW')
plot_histo(axNO,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'NO')
plot_histo(axNE,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'NE')
plot_histo(axGrIS,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'GrIS')

#Finalise plot
axSW.set_xlabel('Ice content [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2019 - 2 years running slabs - 0m thick slabs excluded')
plt.show()
'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/HistoNonZeros_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''

#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.boxplot(data=IceThickness_all_sectors_region_GrIS[IceThickness_all_sectors_region_GrIS['20m_ice_content_m']>0], x="20m_ice_content_m", y="key_shp",hue="type",orient="h",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_ylabel('')
ax_regions_GrIS.set_xlabel('Ice content [m]')
ax_regions_GrIS.set_xlim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle('2019 - 2 years running slabs - 0m thick slabs excluded')

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/BoxplotNonZeros_Emax_VS_IceSlabs_Masked_20121619_Box_Tedstone_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs.png',dpi=500)
'''
####################### Plot without 0m thick ice slabs #######################

###############################################################################
###         Plot Ice Slabs Thickness data in the different sectors          ###
###############################################################################

#pdb.set_trace()

###############################################################################
###                                   SAR                                   ###
###############################################################################

############################# Sectors - 2019 MVRL #############################

#Create dataframe to associate each box with its region
box_and_region=pd.DataFrame(data={'box_nb': np.arange(1,32),'region': np.nan})
box_and_region['region'].iloc[0:9]='SW'
box_and_region['region'].iloc[9]='shared'
box_and_region['region'].iloc[10:14]='CW'
box_and_region['region'].iloc[14]='shared'
box_and_region['region'].iloc[15:21]='NW'
box_and_region['region'].iloc[21]='shared'
box_and_region['region'].iloc[22:28]='NO'
box_and_region['region'].iloc[28]='shared'
box_and_region['region'].iloc[29:32]='NE'

above_all=pd.DataFrame()
within_all=pd.DataFrame()
below_all=pd.DataFrame()
in_between_all=pd.DataFrame()

#Load SAR csv files
for indiv_box in range(4,32):
    #open above
    try:
        above = pd.read_csv(path_data+'SAR_sectors/above/SAR_above_box_'+str(indiv_box)+'_year_2019.csv')
        #drop index column
        above=above.drop(columns=["Unnamed: 0"])
        above['sector']=pd.Series(['Above']*len(above))
        above['box_nb']=pd.Series([indiv_box]*len(above))
        above['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(above))
        #Append data
        above_all=pd.concat([above_all,above])
    except FileNotFoundError:
        print('No above')
    
    #open in_between
    try:
        in_between = pd.read_csv(path_data+'SAR_sectors/in_between/SAR_in_between_box_'+str(indiv_box)+'_year_2019.csv')
        #drop index column
        in_between=in_between.drop(columns=["Unnamed: 0"])
        in_between['sector']=pd.Series(['InBetween']*len(in_between))
        in_between['box_nb']=pd.Series([indiv_box]*len(in_between))
        in_between['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(in_between))
        #Append data
        in_between_all=pd.concat([in_between_all,in_between])
    except FileNotFoundError:
        print('No in_between')
        
    #open within
    try:
        within = pd.read_csv(path_data+'SAR_sectors/within/SAR_within_box_'+str(indiv_box)+'_year_2019.csv')
        #drop index column
        within=within.drop(columns=["Unnamed: 0"])
        within['sector']=pd.Series(['Within']*len(within))
        within['box_nb']=pd.Series([indiv_box]*len(within))
        within['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(within))
        #Append data
        within_all=pd.concat([within_all,within])
    except FileNotFoundError:
        print('No within')
    
    #open below
    try:
        below = pd.read_csv(path_data+'SAR_sectors/below/SAR_below_box_'+str(indiv_box)+'_year_2019.csv')
        #drop index column
        below=below.drop(columns=["Unnamed: 0"])
        below['sector']=pd.Series(['Below']*len(below))
        below['box_nb']=pd.Series([indiv_box]*len(below))
        below['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(below))
        #Append data
        below_all=pd.concat([below_all,below])
    except FileNotFoundError:
        print('No below')

### For boxes which share different regions, perform intersection with GrIS drainage bassins ###
#Reunite all the sectors into one single dataframe
SAR_all_sectors=pd.concat([above_all,in_between_all,within_all,below_all])
#Reset index to have a single index per data point
SAR_all_sectors=SAR_all_sectors.reset_index(drop=True)
#Identify index which have the 'shared' region
index_shared=SAR_all_sectors[SAR_all_sectors.region=='shared'].index
#Select data in boxes sharing 2 regions
SAR_all_sectors_shared=SAR_all_sectors.loc[index_shared].copy()
#Transform SAR_all_sectors_shared into a geopandas dataframe
SAR_all_sectors_shared_gdp = gpd.GeoDataFrame(SAR_all_sectors_shared,
                                              geometry=gpd.GeoSeries.from_xy(SAR_all_sectors_shared['x_coord_SAR'],
                                                                             SAR_all_sectors_shared['y_coord_SAR'],
                                                                             crs='EPSG:3413'))
#Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
SAR_all_sectors_shared_gdp_with_regions = gpd.sjoin(SAR_all_sectors_shared_gdp, GrIS_drainage_bassins, predicate='within')
#Drop index of drainage bassins and the 'region' colum storing 'shared'
SAR_all_sectors_shared_gdp_with_regions=SAR_all_sectors_shared_gdp_with_regions.drop(columns=["region","index_right"])
SAR_all_sectors_shared_gdp_with_regions=SAR_all_sectors_shared_gdp_with_regions.rename(columns={"SUBREGION1":"region"})
#fill in the SAR_all_sectors dataframe the identified regions
SAR_all_sectors.loc[index_shared,'region']=SAR_all_sectors_shared_gdp_with_regions.region
'''
#Make sure identifcation went well - yes, it performs perfectly!
fig = plt.figure()
gs = gridspec.GridSpec(10, 6)
ax_region_check = plt.subplot(gs[0:10, 0:6],projection=crs)

GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SW'].plot(ax=ax_region_check,color='red',alpha=0.2)
GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='CW'].plot(ax=ax_region_check,color='magenta',alpha=0.2)
GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NW'].plot(ax=ax_region_check,color='green',alpha=0.2)
GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NO'].plot(ax=ax_region_check,color='blue',alpha=0.2)
GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NE'].plot(ax=ax_region_check,color='cyan',alpha=0.2)

ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='SW'].x_coord_SAR,
                        SAR_all_sectors[SAR_all_sectors.region=='SW'].y_coord_SAR,
                        color='red') #Perfect!
ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='CW'].x_coord_SAR,
                        SAR_all_sectors[SAR_all_sectors.region=='CW'].y_coord_SAR,
                        color='magenta') #Perfect!
ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='NW'].x_coord_SAR,
                        SAR_all_sectors[SAR_all_sectors.region=='NW'].y_coord_SAR,
                        color='green')
ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='NO'].x_coord_SAR,
                        SAR_all_sectors[SAR_all_sectors.region=='NO'].y_coord_SAR,
                        color='blue') #Perfect!
ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='NE'].x_coord_SAR,
                        SAR_all_sectors[SAR_all_sectors.region=='NE'].y_coord_SAR,
                        color='cyan')   
'''
### For boxes which share different regions, perform intersection with GrIS drainage bassins ###

#Display figure distribution
fig, (ax_distrib) = plt.subplots()      
ax_distrib.hist(SAR_all_sectors[SAR_all_sectors.sector=='Below'].SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='green',label='Below')
#ax_distrib.hist(in_between_all.SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='yellow',label='In Between')
#ax_distrib.hist(within_all.SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='red',label='Within')
ax_distrib.hist(SAR_all_sectors[SAR_all_sectors.sector=='Above'].SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='blue',label='Above')
ax_distrib.set_xlim(-20,-2)
ax_distrib.set_xlabel('Signal strength [dB]')
ax_distrib.set_ylabel('Density')
ax_distrib.legend()
ax_distrib.set_title('GrIS-wide')

#Display figure distribution in different regions
fig, (ax_distrib_SW,ax_distrib_CW,ax_distrib_NW,ax_distrib_NO,ax_distrib_NE) = plt.subplots(5,1)  
hist_regions(SAR_all_sectors[SAR_all_sectors.region=='SW'],'SW',ax_distrib_SW)
hist_regions(SAR_all_sectors[SAR_all_sectors.region=='CW'],'CW',ax_distrib_CW)
hist_regions(SAR_all_sectors[SAR_all_sectors.region=='NW'],'NW',ax_distrib_NW)
hist_regions(SAR_all_sectors[SAR_all_sectors.region=='NO'],'NO',ax_distrib_NO)
hist_regions(SAR_all_sectors[SAR_all_sectors.region=='NE'],'NE',ax_distrib_NE)
ax_distrib_SW.legend()
fig.suptitle('Regional separation')

#Display boxplot
#GrIS-wide
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_SAR = plt.subplot(gs[0:10, 0:6])
sns.boxplot(data=SAR_all_sectors[np.logical_or((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.sector=='Below'))], y="sector", x="SAR",ax=ax_SAR)#, kde=True)
ax_SAR.set_xlabel('Signal strength [dB]')
ax_SAR.set_ylabel('Category')
ax_SAR.set_title('GrIS-wide')

#Regions
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_SAR = plt.subplot(gs[0:10, 0:6])
sns.boxplot(data=SAR_all_sectors[np.logical_or((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.sector=='Below'))], y="region", x="SAR",hue="sector",ax=ax_SAR)#, kde=True)
ax_SAR.set_xlabel('Signal strength [dB]')
ax_SAR.set_ylabel('Category')
ax_SAR.set_title('GrIS-wide')

#Violin plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_SAR = plt.subplot(gs[0:10, 0:6])
sns.violinplot(data=pd.DataFrame(SAR_all_sectors[np.logical_or((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.sector=='Below'))].to_dict()),
               y="region", x="SAR",hue="sector",ax=ax_SAR,palette=my_pal)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
ax_SAR.set_xlabel('Signal strength [dB]')
ax_SAR.set_ylabel('Region')
ax_SAR.set_title('GrIS-wide')


'''
from scipy import stats
df_for_ttest=SAR_all_sectors[SAR_all_sectors.region=='SW'].copy()
stats.ttest_ind(df_for_ttest[df_for_ttest.sector=='Above'].SAR, df_for_ttest[df_for_ttest.sector=='Below'].SAR, axis=0, equal_var=True, nan_policy='raise')
'''

#Display above, within and below violin plot!
df_except_InBetween=SAR_all_sectors.drop(SAR_all_sectors[SAR_all_sectors.sector=='InBetween'].index.to_numpy()).copy()

#Violin plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_SAR = plt.subplot(gs[0:10, 0:6])
sns.violinplot(data=pd.DataFrame(df_except_InBetween.to_dict()),
               y="region", x="SAR",hue="sector",ax=ax_SAR,palette=my_pal)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
ax_SAR.set_xlabel('Signal strength [dB]')
ax_SAR.set_ylabel('Region')
ax_SAR.set_title('GrIS-wide')

############################# Sectors - 2019 MVRL #############################

###############################################################################
###                                   SAR                                   ###
###############################################################################

pdb.set_trace()


### ABOVE
## --- quantiles 0.25, 0.5, 0.75
#SW=-8.100298;-7.345824-6.67245
#CW=-6.198427;-5.79103;-5.45288
#NW=-7.914589;-6.623534;-5.72343
#NO=-6.453673;-5.319692;-4.507481
#NE=-5.779304;-5.08063;-4.402721

### BELOW
## --- quantiles 0.25, 0.5, 0.75
#SW=-10.201291;-9.623054;-9.077484
#CW=-10.353483;-8.994008;-7.742007
#NW=-11.245815;-10.080444;-8.926942
#NO=-10.186774;-8.533312;-7.128138
#NE-8.31114;-7.101851;-6.259047

### WITHIN
## --- quantiles 0.25, 0.5, 0.75
#SW=-9.739717;-9.198477;-8.653624
#CW=-9.199174;-8.057186;-7.060745
#NW=-10.485064;-9.141042;-8.22071
#NO=-8.973496;-7.322135;-6.249159
#NE=-7.455909;-6.663285;-5.716195
###############################################################################
###                          SAR and Ice Thickness                          ###
###############################################################################

########### Load ice slabs with SAR dataset and identify the sector ###########
#Path to data
path_SAR_And_IceThickness=path_data+'csv/'
#List all the files in the folder
list_composite=os.listdir(path_SAR_And_IceThickness) #this is inspired from https://pynative.com/python-list-files-in-a-directory/

#Define empty dataframe
upsampled_SAR_and_IceSlabs=pd.DataFrame()
upsampled_SAR_and_IceSlabs_above=pd.DataFrame()
upsampled_SAR_and_IceSlabs_in_between=pd.DataFrame()
upsampled_SAR_and_IceSlabs_within=pd.DataFrame()
upsampled_SAR_and_IceSlabs_below=pd.DataFrame()

#Loop over all the files
for indiv_file in list_composite:
    print(indiv_file)
    
    #Open the individual file
    indiv_csv=pd.read_csv(path_SAR_And_IceThickness+indiv_file)
    
    ### ALL ###
    #Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
    indiv_upsampled_SAR_and_IceSlabs=indiv_csv.groupby('index_right').mean()  
    #Append the data to each other
    upsampled_SAR_and_IceSlabs=pd.concat([upsampled_SAR_and_IceSlabs,indiv_upsampled_SAR_and_IceSlabs])
    ### ALL ###
    
    ### SECTORS ###
    '''
    #Transform this file into a geopandas dataframe
    indiv_csv_gdf = gpd.GeoDataFrame(indiv_csv, geometry=gpd.points_from_xy(indiv_csv.lon_3413, indiv_csv.lat_3413), crs="EPSG:3413")
    #Perform join between the combined ice slabs and SAR dataset with the sectorial dataframe - It could be a unique idnetifier to each data point, or spatial. Unique identifyer is not available, but could be done while generating data!
    indiv_csv_gdf_WithSectors = indiv_csv_gdf.sjoin_nearest(df_indiv_TrackName, how="left", max_distance=1,rsuffix='droite',distance_col='distance_match')
    '''
    #In the sectorial dataframes, keep only data corresponding to the current TrackName
    indiv_IceThickness_above=keep_sectorial(IceThickness_above,indiv_csv.Track_name.unique()[0])
    indiv_IceThickness_in_between=keep_sectorial(IceThickness_in_between,indiv_csv.Track_name.unique()[0])
    indiv_IceThickness_within=keep_sectorial(IceThickness_within,indiv_csv.Track_name.unique()[0])
    indiv_IceThickness_below=keep_sectorial(IceThickness_below,indiv_csv.Track_name.unique()[0])
    
    #Prepare figure to display
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 5)
    ax_check_csv_sectors = plt.subplot(gs[0:5, 0:5],projection=crs)
    ax_check_csv_sectors.scatter(indiv_csv.lon_3413,indiv_csv.lat_3413,s=3,color='black')
    ax_check_csv_sectors.scatter(indiv_IceThickness_above.lon_3413,indiv_IceThickness_above.lat_3413,s=1,color='blue')
    ax_check_csv_sectors.scatter(indiv_IceThickness_in_between.lon_3413,indiv_IceThickness_in_between.lat_3413,s=1,color='yellow')
    ax_check_csv_sectors.scatter(indiv_IceThickness_within.lon_3413,indiv_IceThickness_within.lat_3413,s=1,color='red')
    ax_check_csv_sectors.scatter(indiv_IceThickness_below.lon_3413,indiv_IceThickness_below.lat_3413,s=1,color='green')
        
    #Associate the sector to the dataframe where ice thickness and SAR data are present
    indiv_upsampled_SAR_and_IceSlabs_above=sector_association(indiv_csv,indiv_IceThickness_above,'above')
    indiv_upsampled_SAR_and_IceSlabs_in_between=sector_association(indiv_csv,indiv_IceThickness_in_between,'InBetween')
    indiv_upsampled_SAR_and_IceSlabs_within=sector_association(indiv_csv,indiv_IceThickness_within,'within')
    indiv_upsampled_SAR_and_IceSlabs_below=sector_association(indiv_csv,indiv_IceThickness_below,'below')
    
    #Append data to obtain one dataframe per sector
    if (len(indiv_upsampled_SAR_and_IceSlabs_above)>0):
        upsampled_SAR_and_IceSlabs_above=pd.concat([upsampled_SAR_and_IceSlabs_above,indiv_upsampled_SAR_and_IceSlabs_above])
    
    if (len(indiv_upsampled_SAR_and_IceSlabs_in_between)>0):
        upsampled_SAR_and_IceSlabs_in_between=pd.concat([upsampled_SAR_and_IceSlabs_in_between,indiv_upsampled_SAR_and_IceSlabs_in_between])
        
    if (len(indiv_upsampled_SAR_and_IceSlabs_within)>0):
        upsampled_SAR_and_IceSlabs_within=pd.concat([upsampled_SAR_and_IceSlabs_within,indiv_upsampled_SAR_and_IceSlabs_within])
        
    if (len(indiv_upsampled_SAR_and_IceSlabs_below)>0):
        upsampled_SAR_and_IceSlabs_below=pd.concat([upsampled_SAR_and_IceSlabs_below,indiv_upsampled_SAR_and_IceSlabs_below])
    plt.close()
    ### SECTORS ###
########### Load ice slabs with SAR dataset and identify the sector ###########

################### Relationship using data in sectors only ###################
#Prepare plot
fig = plt.figure()
fig.set_size_inches(14, 8) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
gs = gridspec.GridSpec(6, 6)
ax_scatter = plt.subplot(gs[0:6, 0:2])
ax_map = plt.subplot(gs[0:6, 2:4],projection=crs)
ax_above = plt.subplot(gs[0:6, 4:6])

ax_scatter.scatter(upsampled_SAR_and_IceSlabs_below.radar_signal,upsampled_SAR_and_IceSlabs_below['20m_ice_content_m_gauche'],color='green')
ax_scatter.scatter(upsampled_SAR_and_IceSlabs_in_between.radar_signal,upsampled_SAR_and_IceSlabs_in_between['20m_ice_content_m_gauche'],color='yellow')
ax_scatter.scatter(upsampled_SAR_and_IceSlabs_within.radar_signal,upsampled_SAR_and_IceSlabs_within['20m_ice_content_m_gauche'],color='red')
ax_scatter.scatter(upsampled_SAR_and_IceSlabs_above.radar_signal,upsampled_SAR_and_IceSlabs_above['20m_ice_content_m_gauche'],color='blue')

#Append data to each other
upsampled_SAR_and_IceSlabs_allsectors=pd.concat([upsampled_SAR_and_IceSlabs_above,upsampled_SAR_and_IceSlabs_in_between,upsampled_SAR_and_IceSlabs_within,upsampled_SAR_and_IceSlabs_below])

#Get rid of NaNs
upsampled_SAR_and_IceSlabs_allsectors_NoNaN=upsampled_SAR_and_IceSlabs_allsectors[~upsampled_SAR_and_IceSlabs_allsectors.radar_signal.isna()].copy()

#Transform upsampled_SAR_and_IceSlabs_allsectors_NoNaN as a geopandas dataframe
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp = gpd.GeoDataFrame(upsampled_SAR_and_IceSlabs_allsectors_NoNaN,
                                                                   geometry=gpd.GeoSeries.from_xy(upsampled_SAR_and_IceSlabs_allsectors_NoNaN['lon_3413_gauche'],
                                                                                                  upsampled_SAR_and_IceSlabs_allsectors_NoNaN['lat_3413_gauche'],
                                                                                                  crs='EPSG:3413'))

#Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions = gpd.sjoin(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp, GrIS_drainage_bassins, predicate='within')

#Display ice thickness and SAR at FS
FS_pd=pd.DataFrame(data={'Station': ['FS2', 'FS4', 'FS5'], '10m_ice_content_%': [95.06, 56.50, 38.44], 'SAR': [-10.44, -7.20, -6.09]})

#Display histograms
display_2d_histogram(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,FS_pd)
#Conclusion: I think this is hard to derive a relationship between SAR and ice thickness


################### Relationship using data in sectors only ###################


pdb.set_trace()

############### Relationship using the whole ice slabs dataset ###############
#Create a unique index for each line, and set this new vector as the index in dataframe
upsampled_SAR_and_IceSlabs['index_unique']=np.arange(0,len(upsampled_SAR_and_IceSlabs))
upsampled_SAR_and_IceSlabs=upsampled_SAR_and_IceSlabs.set_index('index_unique')

#Display some descriptive statistics
upsampled_SAR_and_IceSlabs.describe()['radar_signal']
upsampled_SAR_and_IceSlabs.describe()['20m_ice_content_m']

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
    upsampled_SAR_and_IceSlabs.plot.scatter(x='radar_signal',y='20m_ice_content_m',ax=ax_scatter)
    ax_scatter.set_xlim(-18,1)
    ax_scatter.set_ylim(-0.5,20.5)
    ax_scatter.set_xlabel('SAR [dB]')
    ax_scatter.set_ylabel('Ice content [m]')

    ax_IceContent.hist(upsampled_SAR_and_IceSlabs['20m_ice_content_m'],
                       bins=np.arange(np.min(upsampled_SAR_and_IceSlabs['20m_ice_content_m']),np.max(upsampled_SAR_and_IceSlabs['20m_ice_content_m'])),
                       density=True,orientation='horizontal')
    ax_IceContent.set_xlabel('Density [ ]')
    ax_IceContent.set_ylim(-0.5,20.5)

    ax_SAR.hist(upsampled_SAR_and_IceSlabs['radar_signal'],
                bins=np.arange(np.min(upsampled_SAR_and_IceSlabs['radar_signal']),np.max(upsampled_SAR_and_IceSlabs['radar_signal'])),
                density=True)
    ax_SAR.set_xlim(-18,1)
    ax_SAR.set_ylabel('Density [ ]')
    fig.suptitle('Ice content and SAR')
    
    ######### Keep only where ice content is: 0 < ice content < 12m ###########
    restricted_upsampled_SAR_and_IceSlabs=upsampled_SAR_and_IceSlabs.copy()
    restricted_upsampled_SAR_and_IceSlabs=restricted_upsampled_SAR_and_IceSlabs[np.logical_and(restricted_upsampled_SAR_and_IceSlabs['20m_ice_content_m']>0,restricted_upsampled_SAR_and_IceSlabs['20m_ice_content_m']<12)]
    ######### Keep only where ice content is: 0 < ice content < 12m ###########
    
    #There are places where the radar signal is a NaN. Extract it
    upsampled_SAR_and_IceSlabs_no_NaN=upsampled_SAR_and_IceSlabs.copy()
    upsampled_SAR_and_IceSlabs_no_NaN=upsampled_SAR_and_IceSlabs_no_NaN[~pd.isna(upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'])]

    restricted_upsampled_SAR_and_IceSlabs_no_NaN=restricted_upsampled_SAR_and_IceSlabs.copy()
    restricted_upsampled_SAR_and_IceSlabs_no_NaN=restricted_upsampled_SAR_and_IceSlabs_no_NaN[~pd.isna(restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'])]
    
    #Display heatmaps
    fig_heatmap = plt.figure()
    fig_heatmap.set_size_inches(14, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=3)
    gs.update(hspace=3)
    ax_hist2d = plt.subplot(gs[0:5, 0:5])
    ax_hist2d_restricted = plt.subplot(gs[0:5, 5:10])
    ax_hist2d_log = plt.subplot(gs[5:10, 0:5])
    ax_hist2d_restricted_log = plt.subplot(gs[5:10, 5:10])
    
    hist2d_cbar = ax_hist2d.hist2d(upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'],upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r')
    hist2d_log_cbar = ax_hist2d_log.hist2d(upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'],upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r',norm=mpl.colors.LogNorm())

    hist2d_restricted_cbar = ax_hist2d_restricted.hist2d(restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'],restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r')
    hist2d_restricted_log_cbar = ax_hist2d_restricted_log.hist2d(restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'],restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'],bins=30,cmap='magma_r',norm=mpl.colors.LogNorm())
                     #cmin=np.quantile(h[0].flatten(),0.5),cmax=np.max(h[0].flatten()))#density=True
    #ax_hist2d.set_xlim(-18,1)
    ax_hist2d_log.set_xlabel('SAR [dB]')
    ax_hist2d_log.set_ylabel('Ice content [m]')
    ax_hist2d.set_title('0 < ice content < 20 m')
    ax_hist2d_restricted.set_title('0 < ice content < 12 m')
    fig_heatmap.suptitle('Occurrence map')

    ax_hist2d.set_ylim(0,16)
    ax_hist2d_log.set_ylim(0,16)
    ax_hist2d_restricted.set_ylim(0,16)
    ax_hist2d_restricted_log.set_ylim(0,16)
    ax_hist2d.set_xlim(-15,-2)
    ax_hist2d_log.set_xlim(-15,-2)
    ax_hist2d_restricted.set_xlim(-15,-2)
    ax_hist2d_restricted_log.set_xlim(-15,-2)
    
    #Display colorbars    
    fig_heatmap.colorbar(hist2d_cbar[3], ax=ax_hist2d,label='Occurence') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot
    fig_heatmap.colorbar(hist2d_log_cbar[3], ax=ax_hist2d_log,label='log(Occurence)')
    fig_heatmap.colorbar(hist2d_restricted_cbar[3], ax=ax_hist2d_restricted,label='Occurence')
    fig_heatmap.colorbar(hist2d_restricted_log_cbar[3], ax=ax_hist2d_restricted_log,label='log(Occurence)')
    
    
    #sort restricted_appended_df
    restricted_upsampled_SAR_and_IceSlabs_no_NaN=restricted_upsampled_SAR_and_IceSlabs_no_NaN.sort_values(by=['radar_signal'])
    
    #prepare data for fit        
    xdata = np.array(restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'])
    ydata = np.array(restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'])                           
                           
    #manual fit
    ax_hist2d_restricted.plot(xdata, func(xdata, 1,0.26,-2),'r',label='manual fit: y = 1*exp(-0.26*x) - 2')
    ax_hist2d_restricted_log.plot(xdata, func(xdata, 1,0.26,-2),'r',label='manual fit: y = 1*exp(-0.26*x) - 2')

    #automatic fit: try with scipy.curve_fit - following the example on the help page
    #popt, pcov = curve_fit(func, xdata, ydata,p0=[1,0.26,-2],bounds=([-2,0,-5],[5,2,2]))
    popt, pcov = curve_fit(func, xdata, ydata,p0=[1,0.26,-2],bounds=([0,-1,-4],[2,1,2]))
    ax_hist2d_restricted.plot(xdata, func(xdata, *popt),'b',label='automatic fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))#, 'r-'
    ax_hist2d_restricted_log.plot(xdata, func(xdata, *popt),'b',label='automatic fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))#, 'r-'

    ax_hist2d_restricted.legend(loc='upper left',fontsize=6)
    ax_hist2d_restricted_log.legend(loc='upper left',fontsize=6)
    
    pdb.set_trace()
    
    '''
    #Save figure
    plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/relationship/relationship_SAR_IceContent.png',dpi=300,bbox_inches='tight')
    '''
    #Define binning range
    range_binning=np.array([[restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'].min(),restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal'].max()],
                            [restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'].min(),restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m'].max()]])
    
    ###########################################################################
    ###           Get rid of data points where occurrence is low!           ###
    ###########################################################################

    #Define a dataset for regression calculation
    restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression=restricted_upsampled_SAR_and_IceSlabs_no_NaN.copy()
    
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
                logical_SAR=np.logical_and(restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal']>=bounds_row[0],restricted_upsampled_SAR_and_IceSlabs_no_NaN['radar_signal']<bounds_row[1])
                logical_ice_content=np.logical_and(restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m']>=bounds_col[0],restricted_upsampled_SAR_and_IceSlabs_no_NaN['20m_ice_content_m']<bounds_col[1])
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
    restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression=restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression.drop(index=np.unique(index_removal),axis=1)
    
    #Display resulting grid
    fig_regression = plt.figure()
    fig_regression.set_size_inches(7, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 5)
    gs.update(wspace=3)
    gs.update(hspace=3)
    ax_hist2d_regression = plt.subplot(gs[0:5, 0:5])
    ax_hist2d_regression_log = plt.subplot(gs[5:10,0:5])
    
    hist2d_regression_cbar = ax_hist2d_regression.hist2d(restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression['radar_signal'],restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression['20m_ice_content_m'],cmap='magma_r',bins=30,range=range_binning)
    hist2d_regression_log_cbar = ax_hist2d_regression_log.hist2d(restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression['radar_signal'],restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression['20m_ice_content_m'],cmap='magma_r',bins=30,norm=mpl.colors.LogNorm())
    
    ax_hist2d_regression_log.set_xlabel('SAR [dB]')
    ax_hist2d_regression_log.set_ylabel('Ice content [m]')
    ax_hist2d_regression.set_title('0 < ice content < 15 m')
    
    ax_hist2d_regression.set_ylim(0,20)
    ax_hist2d_regression_log.set_ylim(0,20)
    ax_hist2d_regression.set_xlim(-16.5,-4)
    ax_hist2d_regression_log.set_xlim(-16.5,-4)    

    #Calculate regression
    xdata = np.array(restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression['radar_signal'])
    ydata = np.array(restricted_upsampled_SAR_and_IceSlabs_no_NaN_regression['20m_ice_content_m'])                           
    
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
    
############### Relationship using the whole ice slabs dataset ###############


###############################################################################
###                          SAR and Ice Thickness                          ###
###############################################################################



'''
###############################################################################
###                           Aquitard map                                  ###
###############################################################################

import rioxarray as rxr

#Define paths where data are stored
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_SAR=path_local+'data/SAR/HV_2017_2018/'

#Open SAR image
### --- This is from Fisg4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load SAR data
SAR_N_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_N_manual-0000000000-0000000000.tif',masked=True).squeeze()#No need to reproject satelite image
SAR_N_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_N_manual-0000000000-0000023296.tif',masked=True).squeeze()
SAR_NW_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_NW_manual-0000000000-0000000000.tif',masked=True).squeeze()
SAR_NW_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_NW_manual-0000000000-0000023296.tif',masked=True).squeeze()
SAR_SW_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000000000-0000000000.tif',masked=True).squeeze()
SAR_SW_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000023296-0000000000.tif',masked=True).squeeze()

#Choose cutoff signal
cutoff=-8
#Apply binary cutoff

#Display map of efficient aquitard with 2019 MVRL


###############################################################################
###                           Aquitard map                                  ###
###############################################################################
'''


### UNTIL THERE, CHECKED AND WORKING


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



