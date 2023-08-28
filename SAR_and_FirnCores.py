# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:45:00 2023

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
import pickle
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd

#Define paths where data are stored
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

path_firn_cores=path_switchdrive+'RT3/data/firn_cores/'
path_SAR=path_local+'data/SAR/HV_2021/'
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'

#Load IMBIE drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

### -------------------------- Load Transect data ------------------------- ###
path_transect='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/IceSlabs_And_Coordinates/'
#Open 2013 transect
f_2013_Transect = open(path_transect+'20130409_01_010_012_IceSlabs.pickle', "rb")
Transect_2013 = pickle.load(f_2013_Transect)
f_2013_Transect.close()
#Open 2018 transect
f_2018_Transect = open(path_transect+'20180421_01_004_007_IceSlabs.pickle', "rb")
Transect_2018 = pickle.load(f_2018_Transect)
f_2018_Transect.close()
#Convert 0 in NaNs in ice slasb mask
Transect_2013["IceSlabs_Mask"][Transect_2013["IceSlabs_Mask"]==0]=np.nan
Transect_2018["IceSlabs_Mask"][Transect_2018["IceSlabs_Mask"]==0]=np.nan

#Transform radargram coordinates from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
#Transform 2013
points=transformer.transform(Transect_2013["longitude_EPSG_4326"],
                             Transect_2013["latitude_EPSG_4326"])
Transect_2013["longitude_EPSG_3413"]=points[0]
Transect_2013["latitude_EPSG_3413"]=points[1]
#Transform 2018
points=transformer.transform(Transect_2018["longitude_EPSG_4326"],
                             Transect_2018["latitude_EPSG_4326"])
Transect_2018["longitude_EPSG_3413"]=points[0]
Transect_2018["latitude_EPSG_3413"]=points[1]

#Compute distances
Transect_2013["distances"]= compute_distances(Transect_2013["longitude_EPSG_3413"],Transect_2013["latitude_EPSG_3413"])
Transect_2018["distances"]= compute_distances(Transect_2018["longitude_EPSG_3413"],Transect_2018["latitude_EPSG_3413"])
### -------------------------- Load Transect data ------------------------- ###

### ------------------------- Load firn cores data ------------------------ ###
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
### ------------------------- Load firn cores data ------------------------ ###

#Display ice slab transect
plt.rcParams.update({'font.size': 8})
fig1 = plt.figure(figsize=(19.09,  2.07))
gs = gridspec.GridSpec(5, 5)
ax_iceslab = plt.subplot(gs[0:5, 0:5])
#Display 2018 ice slab
cb=ax_iceslab.pcolor(Transect_2018["longitude_EPSG_4326"],Transect_2018["depth"],
                     Transect_2018["IceSlabs_Mask"],cmap=plt.get_cmap('gray_r'),vmin=0, vmax=0.0001)
#Display 2013 ice slab
cb=ax_iceslab.pcolor(Transect_2013["longitude_EPSG_4326"],Transect_2013["depth"],
                     Transect_2013["IceSlabs_Mask"],cmap=plt.get_cmap('autumn_r'),vmin=0, vmax=0.0001,alpha=0.5,edgecolor='None')
ax_iceslab.invert_yaxis() #Invert the y axis = avoid using flipud.
#Display FS location
ax_iceslab.scatter(firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS2_12m"].E,1)
ax_iceslab.scatter(firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS4_20m"].E,1)
ax_iceslab.scatter(firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS5_20m"].E,1)

#Display map to evaluate distance
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################
plt.rcParams.update({'font.size': 8})
fig_map = plt.figure(figsize=(5,5))
gs = gridspec.GridSpec(5, 5)
ax_map = plt.subplot(gs[0:5, 0:5],projection=crs)
ax_map.scatter(Transect_2013["longitude_EPSG_3413"],Transect_2013["latitude_EPSG_3413"],c='red')
ax_map.scatter(Transect_2018["longitude_EPSG_3413"],Transect_2018["latitude_EPSG_3413"],c='black',alpha=0.5)
ax_map.scatter(firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS2_12m"].lon_3413,
               firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS2_12m"].lat_3413)

ax_map.scatter(firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS4_20m"].lon_3413,
               firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS4_20m"].lat_3413)

ax_map.scatter(firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS5_20m"].lon_3413,
               firn_cores_pd["overview"][firn_cores_pd["overview"].core =="FS5_20m"].lat_3413)
ax_map.set_xlim(-139755,-56158)
ax_map.set_ylim(-2532672,-2517589)

# Display scalebar with GeoPandas
ax_map.add_artist(ScaleBar(1,location='upper right',box_alpha=0,box_color=None))
#Extract ice thickness in the vicinity of the FS location, in the first 10 m firn.
#The depth of the top of the firn layer in each firn core was already identified

#Set the depth at which starts the firn layer
firn_cores_overview.loc[0,"depth_firn"]=140#FS4_20m
firn_cores_overview.loc[2,"depth_firn"]=123#FS5_20m
firn_cores_overview.loc[4,"depth_firn"]=114#FS2_12m

#Transform firn_cores_overview into a geopandas df for distance calculation
firn_cores_overview = gpd.GeoDataFrame(firn_cores_overview,
                                       geometry=gpd.GeoSeries.from_xy(firn_cores_overview['lon_3413'],
                                                                       firn_cores_overview['lat_3413'],
                                                                       crs='EPSG:3413'))

#Extract the average horizontal resolution of the 2013 and 2018 radargrams

#perform average ice content over 1 km, i.e. +/- 500 m
for index, row in firn_cores_overview.iterrows():
    #print(row)
    
    # -------------------------------- 2013 --------------------------------- #
    #Locate the vicinity of the FS longitude in 2013 transect
    index_within_bounds_2013=np.logical_and(Transect_2013["longitude_EPSG_3413"]>=(row.lon_3413-500),Transect_2013["longitude_EPSG_3413"]<=(row.lon_3413+500))
    
    #Locate the index of the top firn layer and top firn layer + 10 m
    index_top_firn_2013=np.where(np.abs(Transect_2013["depth"]-row.depth_firn/100)==np.min(np.abs(Transect_2013["depth"]-row.depth_firn/100)))[0][0]
    index_10m_firn_2013=np.where(np.abs(Transect_2013["depth"]-(10+row.depth_firn/100))==np.min(np.abs(Transect_2013["depth"]-(10+row.depth_firn/100))))[0][0]
    
    #Display extracted sector on the map
    ax_iceslab.axvline(Transect_2013["longitude_EPSG_4326"][index_within_bounds_2013][0],c='orange')
    ax_iceslab.axvline(Transect_2013["longitude_EPSG_4326"][index_within_bounds_2013][-1],c='orange')
    
    #Extract the ice thickness in the top 10 m of firn in the vicinity of the transect 
    For_IceContent_Extraction_2013 = Transect_2013["IceSlabs_Mask"][index_top_firn_2013:index_10m_firn_2013,index_within_bounds_2013]
        
    Ice_Thickess_10m_2013=[]
    for column in range(0,For_IceContent_Extraction_2013.shape[1]):
        Ice_Thickess_10m_2013=np.append(Ice_Thickess_10m_2013,(For_IceContent_Extraction_2013[:,column]>0).astype(int).sum()*np.mean(np.diff(Transect_2013["depth"])))
    
    #Display 2013 mean ice thickness
    print('Total ice thickness in the first 10 m of 2013 radargram',row.core,':',np.round(np.mean(Ice_Thickess_10m_2013),1),'m')
    print('Total ice content in the first 10 m of 2013 radargram',row.core,':',np.round(np.mean(Ice_Thickess_10m_2013)/10*100,1),'%')
        
    #Calculate average distance between 2013 radargram buffer and FS location - use GeoPandas!
    Transect_2013_within_bounds = pd.DataFrame(data={'longitude_EPSG_3413':Transect_2013["longitude_EPSG_3413"][index_within_bounds_2013],
                                                     'latitude_EPSG_3413':Transect_2013["latitude_EPSG_3413"][index_within_bounds_2013]})
                                                       
    Transect_2013_within_bounds_gpd = gpd.GeoDataFrame(Transect_2013_within_bounds,
                                                       geometry=gpd.GeoSeries.from_xy(Transect_2013_within_bounds['longitude_EPSG_3413'],
                                                                                      Transect_2013_within_bounds['latitude_EPSG_3413'],
                                                                                      crs='EPSG:3413'))
    print('2013 radargram average distance with',row.core,':',np.round(Transect_2013_within_bounds_gpd.distance(row.geometry).mean()),'m')
    # -------------------------------- 2013 --------------------------------- #

    # -------------------------------- 2018 --------------------------------- #
    #Locate the vicinity of the FS longitude in 2018 transect
    index_within_bounds_2018=np.logical_and(Transect_2018["longitude_EPSG_3413"]>=(row.lon_3413-500),Transect_2018["longitude_EPSG_3413"]<=(row.lon_3413+500))
    
    #Locate the index of the top firn layer and top firn layer + 10 m
    index_top_firn_2018=np.where(np.abs(Transect_2018["depth"]-row.depth_firn/100)==np.min(np.abs(Transect_2018["depth"]-row.depth_firn/100)))[0][0]
    index_10m_firn_2018=np.where(np.abs(Transect_2018["depth"]-(10+row.depth_firn/100))==np.min(np.abs(Transect_2018["depth"]-(10+row.depth_firn/100))))[0][0]
    
    #Display extracted sector on the map
    ax_iceslab.axvline(Transect_2018["longitude_EPSG_4326"][index_within_bounds_2018][0],c='grey')
    ax_iceslab.axvline(Transect_2018["longitude_EPSG_4326"][index_within_bounds_2018][-1],c='grey')
    
    #Extract the ice thickness in the top 10 m of firn in the vicinity of the transect 
    For_IceContent_Extraction_2018 = Transect_2018["IceSlabs_Mask"][index_top_firn_2018:index_10m_firn_2018,index_within_bounds_2018]
        
    Ice_Thickess_10m_2018=[]
    for column in range(0,For_IceContent_Extraction_2018.shape[1]):
        Ice_Thickess_10m_2018=np.append(Ice_Thickess_10m_2018,(For_IceContent_Extraction_2018[:,column]>0).astype(int).sum()*np.mean(np.diff(Transect_2018["depth"])))
    
    #Display 2018 mean ice thickness
    print('Total ice thickness in the first 10 m of 2018 radargram',row.core,':',np.round(np.mean(Ice_Thickess_10m_2018),1),'m')
    print('Total ice content in the first 10 m of 2018 radargram',row.core,':',np.round(np.mean(Ice_Thickess_10m_2018)/10*100,1),'%')
    
    #Calculate average distance between 2018 radargram buffer and FS location - use GeoPandas!
    Transect_2018_within_bounds = pd.DataFrame(data={'longitude_EPSG_3413':Transect_2018["longitude_EPSG_3413"][index_within_bounds_2018],
                                                     'latitude_EPSG_3413':Transect_2018["latitude_EPSG_3413"][index_within_bounds_2018]})
                                                       
    Transect_2018_within_bounds_gpd = gpd.GeoDataFrame(Transect_2018_within_bounds,
                                                       geometry=gpd.GeoSeries.from_xy(Transect_2018_within_bounds['longitude_EPSG_3413'],
                                                                                      Transect_2018_within_bounds['latitude_EPSG_3413'],
                                                                                      crs='EPSG:3413'))
    print('2018 radargram average distance with',row.core,':',np.round(Transect_2018_within_bounds_gpd.distance(row.geometry).mean()),'m')
    # -------------------------------- 2018 --------------------------------- #
    
pdb.set_trace()

#Open SAR image
SAR_SW_00_00 = rasterio.open(path_SAR+'ref_IW_HV_2021_2021_32_106_40m_ASCDESC_SW_minnscenes50-0000000000-0000000000.tif')
'''
### --- This is from Fisg4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load SAR data
SAR_SW_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000000000-0000000000.tif',masked=True).squeeze()
SAR_SW_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000023296-0000000000.tif',masked=True).squeeze()
'''
#Extract SAR data at place where firn core was drilled
tuple_list=np.array((firn_cores_overview.lon_3413.to_numpy(),firn_cores_overview.lat_3413.to_numpy())).T #from https://stackoverflow.com/questions/35091879/merge-two-arrays-vertically-to-array-of-tuples-using-numpy

#This is fropm https://gis.stackexchange.com/questions/190423/getting-pixel-values-at-single-point-using-rasterio
extracted_SAR=[]
for val in SAR_SW_00_00.sample(tuple_list): 
    extracted_SAR=np.append(extracted_SAR,val)

#Store extracted SAR in firn_cores_overview
firn_cores_overview['SAR']=extracted_SAR

print('End of code')       

