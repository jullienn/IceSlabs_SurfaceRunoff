# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:45:00 2023

@author: jullienn
"""

def calculate_ice_content(dfs_firn_core,firn_core,depth_start_firn):
    #Calculate firn ice content within the first 10 m.
    penetration_depth_Cband=1000

    #This corresponds to the index where the firn layer starts
    index_start_firn=np.where(dfs_firn_core[firn_core]['depth (cm)']==depth_start_firn)[0][0]
    index_end_consideration=np.where(dfs_firn_core[firn_core]['depth (cm)']==(depth_start_firn+penetration_depth_Cband))[0][0]

    #Select portion of firn dataframe exposed to Cband
    firn_dataframe=dfs_firn_core[firn_core].iloc[index_start_firn:index_end_consideration].copy()
    
    #Deal with ice layer in firn
    index_ice_in_firn=np.logical_and(firn_dataframe['material']=='firn',firn_dataframe['layer contents']=='ice')
    firn_dataframe.loc[index_ice_in_firn,'% ice']=firn_dataframe.loc[index_ice_in_firn,'layer thickness (cm)']*100

    #Deal with firn layer in ice
    index_firn_in_ice=np.logical_and(firn_dataframe['material']=='ice',firn_dataframe['layer contents']=='firn')
    index_firn_in_ice_with_NaN_perc_ice=np.logical_and(index_firn_in_ice,firn_dataframe['% ice'].isna())
    firn_dataframe.loc[index_firn_in_ice_with_NaN_perc_ice,'% ice']=(1-firn_dataframe.loc[index_firn_in_ice_with_NaN_perc_ice,'layer thickness (cm)'])*100

    #Where 'material' is 'ice' and '% ice' is nan, replace nan by 1. This is inspired from https://stackoverflow.com/questions/36909977/update-row-values-where-certain-condition-is-met-in-pandas
    firn_dataframe.loc[np.logical_and(firn_dataframe['material']=='ice',firn_dataframe['% ice'].isna()),'% ice']=100
    
    #Store ice content in dataframe
    firn_cores_pd['overview'].loc[firn_cores_pd['overview']['core']==firn_core,'ice content %']=np.sum(firn_dataframe['% ice']/100)/100/10*100

    print('Total ice content in the first 10 m firn core',firn_core,':',str(np.sum(firn_dataframe['% ice']/100)/100/10*100),'%')
    
    #Transform FS coordinates from WGS84 to EPSG:3413
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    points=transformer.transform(firn_cores_pd['overview'].loc[firn_cores_pd['overview']['core']==firn_core,'E'].to_numpy()[0],
                                 firn_cores_pd['overview'].loc[firn_cores_pd['overview']['core']==firn_core,'N'].to_numpy()[0])
    
    firn_cores_pd['overview'].loc[firn_cores_pd['overview']['core']==firn_core,'lon_3413']=points[0]
    firn_cores_pd['overview'].loc[firn_cores_pd['overview']['core']==firn_core,'lat_3413']=points[1]
    '''
    #Not necessary
    #Display density profile
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(5, 5)
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    ax_density_profile = plt.subplot(gs[0:5, 0:5])
    ax_density_profile.plot(dfs_firn_core[firn_core]['density'],dfs_firn_core[firn_core]['depth (cm)']/100)
    ax_density_profile.invert_yaxis()
    ax_density_profile.set_xlabel('Density [g/cm3]')
    ax_density_profile.set_ylabel('Depth [m]')
    '''
    return np.sum(firn_dataframe['% ice']/100)/100/10*100

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
import os
from pyproj import Transformer
import rasterio

#Define paths where data are stored
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

path_firn_cores=path_switchdrive+'RT3/data/firn_cores/'
path_SAR=path_local+'data/SAR/HV_2017_2018/'
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'

#Load IMBIE drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

#Open SAR image
SAR_SW_00_00 = rasterio.open(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000000000-0000000000.tif')

'''
### --- This is from Fisg4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load SAR data
SAR_SW_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000000000-0000000000.tif',masked=True).squeeze()
SAR_SW_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000023296-0000000000.tif',masked=True).squeeze()
'''

#Load firn cores data
firn_cores_pd=pd.read_excel(path_firn_cores+'firn_cores_2021.xlsx',sheet_name=['overview','FS2_12m','FS4_20m','FS4_5m','FS5_20m','FS5_5m'])
#Add a column to store ice content
firn_cores_pd['overview']['ice content %']=[np.nan]*len(firn_cores_pd['overview'])

#Note that we ignore ice layer in the snow layer on top of the firn
ice_content_percentage_FS5=calculate_ice_content(firn_cores_pd,'FS5_20m',123)
ice_content_percentage_FS4=calculate_ice_content(firn_cores_pd,'FS4_20m',140)
ice_content_percentage_FS2=calculate_ice_content(firn_cores_pd,'FS2_12m',114)

#Could be interesting to calculate in the first 5 meters in FS4 and FS5 to access signal penetration depths

#Drop FS where no ice content was computed
firn_cores_overview=firn_cores_pd['overview'][~firn_cores_pd['overview']['ice content %'].isna()].copy()

#Extract SAR data at place where firn core was drilled
tuple_list=np.array((firn_cores_overview.lon_3413.to_numpy(),firn_cores_overview.lat_3413.to_numpy())).T #from https://stackoverflow.com/questions/35091879/merge-two-arrays-vertically-to-array-of-tuples-using-numpy

#This is fropm https://gis.stackexchange.com/questions/190423/getting-pixel-values-at-single-point-using-rasterio
extracted_SAR=[]
for val in SAR_SW_00_00.sample(tuple_list): 
    extracted_SAR=np.append(extracted_SAR,val)

#Store extracted SAR in firn_cores_overview
firn_cores_overview['SAR']=extracted_SAR

print('End of code')









