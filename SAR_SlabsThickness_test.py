# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:46:44 2023

@author: JullienN
"""

import pickle
import scipy.io
import numpy as np
import pdb
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib.gridspec as gridspec
import rioxarray as rxr
import cartopy.crs as ccrs
from pyproj import Transformer

#Define considered transect
IceSlabsTransect_list=['20180423_01_180_182']
'''
IceSlabsTransect_list=['20180421_01_004_007','20180425_01_166_169','20180426_01_004_006',
                       '20180423_01_180_182','20180427_01_004_006','20180425_01_005_008',
                       '20180427_01_170_172','20180421_01_174_177']
'''
#Define paths
path='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'
path_jullienetal2023='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/'
path_jullienetal2023_forRT3='C:/Users/jullienn/switchdrive/Private/research/RT3/data/export_RT1_for_RT3/'

#1. Load SAR data and a transect case study
'''
#Open file
f_IceSlabsTransect = open(path_jullienetal2023+'IceSlabs_And_Coordinates/'+IceSlabsTransect_date+'_IceSlabs.pickle', "rb")
IceSlabsTransect = pickle.load(f_IceSlabsTransect)
f_IceSlabsTransect.close()

#Transform coordinates from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
points=transformer.transform(np.asarray(IceSlabsTransect["longitude_EPSG_4326"]),np.asarray(IceSlabsTransect["latitude_EPSG_4326"]))
IceSlabsTransect['longitude_EPSG_3413']=points[0]
IceSlabsTransect['latitude_EPSG_3413']=points[1]
'''
### ---------- --This is from 'Greenland_Hydrology_Summary.py' ------------ ###

### ------------------------- Load df_2010_2018 --------------------------- ###
#Load 2010-2018 high estimate
f_20102018_high = open(path_jullienetal2023+'/final_excel/high_estimate/clipped/df_20102018_with_elevation_high_estimate_rignotetalregions_cleaned', "rb")
df_2010_2018_high = pickle.load(f_20102018_high)
f_20102018_high.close

#Load 2010-2018 high estimate for RT3 masked
f_20102018_high_RT3_masked = open(path_jullienetal2023_forRT3+'df_20102018_with_elevation_for_RT3_masked_rignotetalregions', "rb")
df_2010_2018_high_RT3_masked = pickle.load(f_20102018_high_RT3_masked)
f_20102018_high_RT3_masked.close
### ------------------------- Load df_2010_2018 --------------------------- ###

#Open SAR image - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84

#Load SAR data
SAR = rxr.open_rasterio(path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif',
                              masked=True).squeeze() #No need to reproject satelite image

#It seems that SAR data are oversampled, i.e. 5x5 matrix can be resample to 1x1. Shall I take the center of the matrix, i.e. at loc[2:2] ??
offsetx=2
offsety=4
SAR_upsampled=SAR[np.arange(offsety,SAR.shape[0]-5,5),np.arange(offsetx,SAR.shape[1]-2,5)]

#Extract x and y coordinates of SAR image
x_coord_SAR=np.asarray(SAR.x)
y_coord_SAR=np.asarray(SAR.y)

'''
#Define bounds wwhere to zoom
x_min=np.min(Emax_points['x'])-5e4
x_max=np.max(Emax_points['x'])+5e4
y_min=np.min(Emax_points['y'])-5e4
y_max=np.max(Emax_points['y'])+5e4

#Extract coordinates of NDWI image within Emaxs bounds
logical_x_coord_within_bounds=np.logical_and(x_coord_MapPlot>=x_min,x_coord_MapPlot<=x_max)
x_coord_within_bounds=x_coord_MapPlot[logical_x_coord_within_bounds]
logical_y_coord_within_bounds=np.logical_and(y_coord_MapPlot>=y_min,y_coord_MapPlot<=y_max)
y_coord_within_bounds=y_coord_MapPlot[logical_y_coord_within_bounds]

#Define extents based on the bounds
extent_MapPlot = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
#Display NDWI image
cbar=ax1.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap='Blues',vmin=vlim_min,vmax=vlim_max)
'''
#Define extents
extent_SAR = [np.min(x_coord_SAR), np.max(x_coord_SAR), np.min(y_coord_SAR), np.max(y_coord_SAR)]#[west limit, east limit., south limit, north limit]

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)
#Display coastlines
ax1.coastlines(edgecolor='black',linewidth=0.075)
### ---------- --This is from 'Greenland_Hydrology_Summary.py' ------------ ###

#2. Display SAR and transect
#Display SAR image
cbar=ax1.imshow(SAR, extent=extent_SAR, transform=crs, origin='upper', cmap='Blues')
#Display Ice Slabs transect
'''
ax1.scatter(IceSlabsTransect['longitude_EPSG_3413'],IceSlabsTransect['latitude_EPSG_3413'],c=IceSlabsTransect['latitude_EPSG_3413'])
'''

#Select data belonging to the list
Extraction_SAR_transect=df_2010_2018_high_RT3_masked[df_2010_2018_high_RT3_masked.Track_name.isin(IceSlabsTransect_list)]

### !!! Keep in mind the df_2010_2018_high_RT3_masked dataset should be clipped
### to the Jullien et al., 2023 ice slabs extent extent shapefile before the overall regression to be done !!!
ax1.scatter(Extraction_SAR_transect['lon_3413'],
            Extraction_SAR_transect['lat_3413'],
            c=Extraction_SAR_transect['20m_ice_content_m'])

#Display Rignot and Mouginot regions edges

#3. Extract the SAR signal of the transect

### --- This is inspired from 'extract_elevation.py' from paper 'Greenland Ice Slabs Expansion and Thickening --- ###
#https://towardsdatascience.com/reading-and-visualizing-geotiff-images-with-python-8dcca7a74510
import rasterio
#Load SAR data
path_SAR = path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif'
SAR_RIO = rasterio.open(path_SAR)

SAR_values=[]
for index, row in Extraction_SAR_transect.iterrows():
    #This is from https://gis.stackexchange.com/questions/190423/getting-pixel-values-at-single-point-using-rasterio
    for val in SAR_RIO.sample([(row.lon_3413,row.lat_3413)]): 
        #Calculate the corresponding SAR value
        SAR_values=np.append(SAR_values,val)
### --- This is inspired from 'extract_elevation.py' from paper 'Greenland Ice Slabs Expansion and Thickening --- ###

#Store the SAR values in the dataframe
Extraction_SAR_transect['SAR']=SAR_values

'''
[print(val) for val in SAR_RIO.sample([(row.lon_3413,row.lat_3413)]) for index, row in Extraction_SAR_transect.iterrows()]

[print(row.lon_3413,row.lat_3413) for index, row in Extraction_SAR_transect.iterrows()]

df2_t = Extraction_SAR_transect.DataFrame({for val in SAR_RIO.sample([(row.lon_3413,row.lat_3413)]) for index, row in Extraction_SAR_transect.iterrows()})

Extraction_SAR_transect.apply(SAR_RIO.sample([(Extraction_SAR_transect['lon_3413'], Extraction_SAR_transect['lat_3413'])]), axis=1)
'''

#4. Plot relationship SAR VS Ice slabs thickness
Extraction_SAR_transect.plot.scatter(x='SAR',y='20m_ice_content_m')

### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###
from shapely.geometry import LineString
from descartes import PolygonPatch

### Extract SAR values in the vicinity of the transect to reduce computation
#1. Transform transect into a line
Extraction_SAR_transect_tuple=[tuple(row[['lon_3413','lat_3413']]) for index, row in Extraction_SAR_transect.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
transect_line=LineString(Extraction_SAR_transect_tuple)

#2. Create a buffer around this line
buffered_transect_polygon=transect_line.buffer(200,cap_style=3)
#Convert polygon of buffered_transect_polygon into a geopandas dataframe
buffered_transect_polygon_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[buffered_transect_polygon]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe

''' Visualize
#Create polygon patch of the polygon
plot_buffered_transect_polygon = PolygonPatch(buffered_transect_polygon,edgecolor='red',facecolor='none')
#Display plot_buffered_transect_polygon
ax1.add_patch(plot_buffered_transect_polygon)
ax1.scatter(Extraction_SAR_transect['lon_3413'],Extraction_SAR_transect['lat_3413'],color='black')
'''
### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###

#3. Extract SAR values within the buffer
from shapely.geometry import mapping

#Perform clip
SAR_clipped = SAR.rio.clip(buffered_transect_polygon_gpd.geometry.apply(mapping),
                                      # This is needed if your GDF is in a diff CRS than the raster data
                                      buffered_transect_polygon_gpd.crs) #This is from https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/crop-raster-data-with-shapefile-in-python/

SAR_upsampled_clipped = SAR_upsampled.rio.clip(buffered_transect_polygon_gpd.geometry.apply(mapping),
                                      # This is needed if your GDF is in a diff CRS than the raster data
                                      buffered_transect_polygon_gpd.crs) #This is from https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/crop-raster-data-with-shapefile-in-python/


#Display quickly that it worked
SAR_upsampled_clipped.plot(ax=ax1)

#Check the grid overlap
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax2 = plt.subplot(projection=crs)
SAR_clipped.plot(ax=ax2,edgecolor='black')
SAR_upsampled_clipped.plot(ax=ax2,edgecolor='red',alpha=0.2)
'''
ax2.set_xlim(-138232, -137363)
ax2.set_ylim(-2525790, -2524978)
'''

# Vectorise the SAR raster
######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
x, y, radar_signal = SAR_upsampled_clipped.x.values, SAR_upsampled_clipped.y.values, SAR_upsampled_clipped.values
x, y = np.meshgrid(x, y)
x, y, radar_signal = x.flatten(), y.flatten(), radar_signal.flatten()

print("Converting to GeoDataFrame...")
SAR_pd = pd.DataFrame.from_dict({'radar_signal': radar_signal, 'x': x, 'y': y})
SAR_vector = gpd.GeoDataFrame(SAR_pd, geometry=gpd.GeoSeries.from_xy(SAR_pd['x'], SAR_pd['y'], crs=SAR_upsampled_clipped.rio.crs))
SAR_grid = SAR_vector.buffer(50, cap_style=3)
#Convert SAR_grid into a geopandas dataframe
SAR_grid_gpd = gpd.GeoDataFrame(SAR_pd,geometry=gpd.GeoSeries(SAR_grid),crs='epsg:3413')#from https://gis.stackexchange.com/questions/266098/how-to-convert-a-geoseries-to-a-geodataframe-with-geopandas
#There is indeed one unique index fear each cell in SAR_grid_gpd

#Display centroid of each cell as well as the created polygons and make sure they are correct
ax2.scatter(SAR_pd.x,SAR_pd.y,color='blue')
SAR_grid.plot(ax=ax2,alpha=0.8,edgecolor='yellow')
######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########


# Perform the intersection between each cell of the SAR data and Ice Slabs transect
### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###
#Convert Extraction_SAR_transect into a geopandas dataframe
Extraction_SAR_transect_gpd = gpd.GeoDataFrame(Extraction_SAR_transect, geometry=gpd.points_from_xy(Extraction_SAR_transect.lon_3413, Extraction_SAR_transect.lat_3413), crs="EPSG:3413")
#Perform the join
pointInPolys= gpd.tools.sjoin(Extraction_SAR_transect_gpd, SAR_grid_gpd, predicate="within", how='left') #This is from https://www.matecdev.com/posts/point-in-polygon.html
### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###

#Drop the SAR column because it does not seems to be correct, the column 'radar_signal' is!
pointInPolys=pointInPolys.drop(labels=['SAR'],axis='columns')

#Where index is identical, keep a single value of radar signal and average the ice content
downsampled_df=pointInPolys.groupby('index_right').mean()


#Display the results
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax3 = plt.subplot(projection=crs)
SAR_grid_gpd.plot(ax=ax3,edgecolor='black')
ax3.scatter(downsampled_df.lon_3413,downsampled_df.lat_3413,c=downsampled_df['20m_ice_content_m'],s=500)
ax3.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['20m_ice_content_m'])


#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
ax4 = plt.subplot()
ax4.scatter(downsampled_df['radar_signal'],downsampled_df['20m_ice_content_m'])


'''
#export Extraction_SAR_transect as a shapefile.
Extraction_SAR_transect_gpd = gpd.GeoDataFrame(Extraction_SAR_transect, geometry=gpd.points_from_xy(Extraction_SAR_transect.lon_3413, Extraction_SAR_transect.lat_3413))
Extraction_SAR_transect_gpd.to_file(path+'SAR/Extraction_SAR_transect_single_gpd.shp')
from rasterstats import zonal_stats, point_query
stats = zonal_stats(path+'SAR/Extraction_SAR_transect_single_gpd.shp', path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif')
pts = point_query(path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif',path+'SAR/Extraction_SAR_transect_single_gpd.shp')

'''




#Display the ice slabs transect
