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
import xarray
import matplotlib.patches as patches

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
#Open 2012 transect
f_2012_Transect = open(path_transect+'20120423_01_137_138_IceSlabs.pickle', "rb")
Transect_2012 = pickle.load(f_2012_Transect)
f_2012_Transect.close()
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
#Transform 2012
points=transformer.transform(Transect_2012["longitude_EPSG_4326"],
                             Transect_2012["latitude_EPSG_4326"])
Transect_2012["longitude_EPSG_3413"]=points[0]
Transect_2012["latitude_EPSG_3413"]=points[1]
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

### Display strain rate close to FS1 for crevassing potential ###

### --------------------- Load 2012 and 2018 radargrams ------------------- ###
path_depth_corrected='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/i_out_from_IceBridgeGPR_Manager_v2.py/pickles_and_images/Depth_Corrected_Picklefiles/'
#Open 2012 transect
f_2012_DepthCorrected = open(path_depth_corrected+'20120423_01_137_138_DEPTH_CORRECTED.pickle', "rb")
DepthCorrected_2012 = pickle.load(f_2012_DepthCorrected)
f_2012_DepthCorrected.close()
#Open 2018 transect
f_2018_DepthCorrected = open(path_depth_corrected+'20180421_01_004_007_DEPTH_CORRECTED.pickle', "rb")
DepthCorrected_2018 = pickle.load(f_2018_DepthCorrected)
f_2018_DepthCorrected.close()
### --------------------- Load 2012 and 2018 radargrams ------------------- ###

### ----------------- Load Wintertine Principal Strain Rates -------------- ###
#Display strain rate, from https://ubir.buffalo.edu/xmlui/handle/10477/82127
GrIS_StrainRate= xarray.open_dataset("C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/10m_arcticDEM/GrISWinterPrincipalStrainrate_0670.nc")
GrIS_StrainRate.rio.write_crs(3413,inplace=True)
#Make sure crs is set
print(GrIS_StrainRate.spatial_ref)

#Set x and y lims of the plot, and to use for GrIS Strain Rate load and display
west_lim = -47.42
east_lim = -46.9

x_min=Transect_2012["longitude_EPSG_3413"][np.argmin(np.abs(Transect_2012["longitude_EPSG_4326"]-west_lim))]
x_max=Transect_2012["longitude_EPSG_3413"][np.argmin(np.abs(Transect_2012["longitude_EPSG_4326"]-east_lim))]
y_min=-2532672.0
y_max=-2517589.0

#Extract x and y coordinates of image
x_coord_GrIS_StrainRate=GrIS_StrainRate.x.data
y_coord_GrIS_StrainRate=GrIS_StrainRate.y.data

#Extract coordinates ofcumulative raster within Emaxs bounds
logical_x_coord_within_bounds=np.logical_and(x_coord_GrIS_StrainRate>=x_min,x_coord_GrIS_StrainRate<=x_max)
x_coord_within_bounds=x_coord_GrIS_StrainRate[logical_x_coord_within_bounds]
logical_y_coord_within_bounds=np.logical_and(y_coord_GrIS_StrainRate>=y_min,y_coord_GrIS_StrainRate<=y_max)
y_coord_within_bounds=y_coord_GrIS_StrainRate[logical_y_coord_within_bounds]

#Define extents based on the bounds
extent_GrIS_StrainRate = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
### ----------------- Load Wintertine Principal Strain Rates -------------- ###

### ----------------- Load Worldview image July 2023 ----------------- ###
path_WorldView="C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/WorldView/015914996010_01_P001_PAN/"
WorldView = rxr.open_rasterio(path_WorldView+'23AUG02150555-P2AS-015914996010_01_P001.tif',
                              masked=True).squeeze() #No need to reproject satelite image
#Define crs of the WorldView image
crs_WorldView=ccrs.UTM(32)

#Define focus limits
x_min_WorldView = 398220
x_max_WorldView = 398651
y_min_WorldView = 7431365
y_max_WorldView = 7431751

#Extract x and y coordinates of WorldView image
x_coord_GrIS_WorldView=WorldView.x.data
y_coord_GrIS_WorldView=WorldView.y.data

#Extract coordinates ofcumulative raster within Emaxs bounds
logical_x_coord_within_bounds_WorldView=np.logical_and(x_coord_GrIS_WorldView>=x_min_WorldView,x_coord_GrIS_WorldView<=x_max_WorldView)
x_coord_within_bounds_WorldView=x_coord_GrIS_WorldView[logical_x_coord_within_bounds_WorldView]
logical_y_coord_within_bounds_WorldView=np.logical_and(y_coord_GrIS_WorldView>=y_min_WorldView,y_coord_GrIS_WorldView<=y_max_WorldView)
y_coord_within_bounds_WorldView=y_coord_GrIS_WorldView[logical_y_coord_within_bounds_WorldView]

#Define extents based on the bounds
extent_WorldView = [np.min(x_coord_within_bounds_WorldView), np.max(x_coord_within_bounds_WorldView), np.min(y_coord_within_bounds_WorldView), np.max(y_coord_within_bounds_WorldView)]#[west limit, east limit., south limit, north limit]
### ----------------- Load Worldview image July 2023 ----------------- ###

### --------------------------- Load FS1 location ------------------------- ###
Spring2021_coordinates=pd.read_csv('C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/'+'actual_site_coordinates_spr21_modified.csv',sep=';')
FS1_loc=Spring2021_coordinates[Spring2021_coordinates.name=='FS1'].copy()
#Transform FS1 loc
points=transformer.transform(FS1_loc["X"],
                             FS1_loc["Y"])
FS1_loc["longitude_EPSG_3413"]=points[0]
FS1_loc["latitude_EPSG_3413"]=points[1]
### --------------------------- Load FS1 location ------------------------- ###

#Prepare figure
fig_FS_Transect = plt.figure(figsize=(10.12,18))
gs = gridspec.GridSpec(30, 101)
ax_DepthCorrected_2012 = plt.subplot(gs[0:2, 0:100])
ax_DepthCorrected_2018 = plt.subplot(gs[2:4, 0:100])
ax_StrainRate = plt.subplot(gs[4:15, 0:100], projection=crs)
axc_StrainRate = plt.subplot(gs[4:15, 100:101])
ax_WorldView = plt.subplot(gs[15:30, 0:100], projection=crs_WorldView)

#Display
cb_2012=ax_DepthCorrected_2012.pcolor(Transect_2012["longitude_EPSG_4326"],Transect_2012["depth"],
                                      DepthCorrected_2012[np.arange(0,len(Transect_2012["depth"])),:],cmap=plt.get_cmap('gray'))#,vmin=0, vmax=0.0001)
cb_2018=ax_DepthCorrected_2018.pcolor(Transect_2018["longitude_EPSG_4326"],Transect_2018["depth"],
                                      DepthCorrected_2018[np.arange(0,len(Transect_2018["depth"])),:],cmap=plt.get_cmap('gray'))#,vmin=0, vmax=0.0001)
ax_DepthCorrected_2012.invert_yaxis()
ax_DepthCorrected_2018.invert_yaxis()

#Display FS1 location on the 2012 radargram
ax_DepthCorrected_2012.scatter(FS1_loc.X,1)

#Set xlims
ax_DepthCorrected_2012.set_xlim(west_lim,east_lim)
ax_DepthCorrected_2018.set_xlim(west_lim,east_lim)

#Display GrIS map
cbar_StrainRate=ax_StrainRate.imshow(GrIS_StrainRate.ep[logical_y_coord_within_bounds,logical_x_coord_within_bounds],
                                     extent=extent_GrIS_StrainRate, transform=crs, origin='upper', cmap='RdBu_r',
                                     vmin=-0.0015,
                                     vmax=0.0015,zorder=0)

#Set lims
ax_StrainRate.set_xlim(x_min,x_max)
ax_StrainRate.set_ylim(y_min,y_max)

#Display cbar
cbar_StrainRate_label=fig_FS_Transect.colorbar(cbar_StrainRate, cax=axc_StrainRate)
cbar_StrainRate_label.set_label('Principal strain rate [$yr^{-1}$]')

###################### From Tedstone et al., 2022 #####################
gl=ax_StrainRate.gridlines(draw_labels=True, xlocs=[-47.4,-47.2,-47.0], ylocs=[66.90,66.95,67.00], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
#Customize lat labels
gl.right_labels = False
gl.top_labels = False
###################### From Tedstone et al., 2022 #####################

# Display scalebar with GeoPandas
ax_StrainRate.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))

# Display FS1 location on the GrIS strain rate map
ax_StrainRate.scatter(FS1_loc.longitude_EPSG_3413,FS1_loc.latitude_EPSG_3413)

#Display the ice slabs transect on the GrIS Strain rate map
ax_StrainRate.scatter(Transect_2012["longitude_EPSG_3413"][np.logical_and(Transect_2012["longitude_EPSG_4326"]>=west_lim,Transect_2012["longitude_EPSG_4326"]<=east_lim)],
                      Transect_2012["latitude_EPSG_3413"][np.logical_and(Transect_2012["longitude_EPSG_4326"]>=west_lim,Transect_2012["longitude_EPSG_4326"]<=east_lim)],color='black')

#Display Woldview
ax_WorldView.imshow(WorldView[logical_y_coord_within_bounds_WorldView,logical_x_coord_within_bounds_WorldView],extent=extent_WorldView, transform=crs_WorldView, origin='upper', cmap='Blues_r',zorder=0)

# Display scalebar with GeoPandas
ax_WorldView.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))

#Display WorldView image extent on the Strain map
#Transform WorldView extent coordinates UTM 32N into EPSG 3413
transformer_EPSG32623_EPSG3413 = Transformer.from_crs("EPSG:32623", "EPSG:3413", always_xy=True)
points=transformer_EPSG32623_EPSG3413.transform([x_min_WorldView,x_min_WorldView,x_max_WorldView,x_max_WorldView],
                                                [y_min_WorldView,y_max_WorldView,y_max_WorldView,y_min_WorldView])
ax_StrainRate.plot(np.append(points[0],points[0][0]),np.append(points[1],points[1][0]),color='black')

#Display WorldView image extent on the radargrams
#Transform WorldView extent coordinates UTM 32N into EPSG 4326
transformer_EPSG32623_EPSG4326 = Transformer.from_crs("EPSG:32623", "EPSG:4326", always_xy=True)
points=transformer_EPSG32623_EPSG4326.transform([x_min_WorldView,x_min_WorldView,x_max_WorldView,x_max_WorldView],
                                                [y_min_WorldView,y_max_WorldView,y_max_WorldView,y_min_WorldView])
ax_DepthCorrected_2012.axvline(points[0][0],color='black')
ax_DepthCorrected_2012.axvline(points[0][-1],color='black')
ax_DepthCorrected_2018.axvline(points[0][0],color='black')
ax_DepthCorrected_2018.axvline(points[0][-1],color='black')

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

