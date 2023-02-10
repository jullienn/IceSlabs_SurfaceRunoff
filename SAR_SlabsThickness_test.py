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

#Define projection
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Define considered transect
IceSlabsTransect_list=['20180421_01_004_007']
'''
IceSlabsTransect_list=['20180421_01_004_007','20180425_01_166_169','20180426_01_004_006',
                       '20180423_01_180_182','20180427_01_004_006','20180425_01_005_008',
                       '20180427_01_170_172','20180421_01_174_177']
'''
#Define paths where data are stored
path='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'
path_jullienetal2023='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/'
path_jullienetal2023_forRT3='C:/Users/jullienn/switchdrive/Private/research/RT3/data/export_RT1_for_RT3/'
path_rignotetal2016_GrIS_drainage_bassins='C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'

#1. Load SAR data and a transect case study
#Open transect file
f_IceSlabsTransect = open(path_jullienetal2023+'IceSlabs_And_Coordinates/'+IceSlabsTransect_list[0]+'_IceSlabs.pickle', "rb")
IceSlabsTransect = pickle.load(f_IceSlabsTransect)
f_IceSlabsTransect.close()

#Transform transect coordinates from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
points=transformer.transform(np.asarray(IceSlabsTransect["longitude_EPSG_4326"]),np.asarray(IceSlabsTransect["latitude_EPSG_4326"]))
IceSlabsTransect['longitude_EPSG_3413']=points[0]
IceSlabsTransect['latitude_EPSG_3413']=points[1]

#Display ice slabs transect
fig = plt.figure()
fig.set_size_inches(18, 2) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_radargram = plt.subplot()
ax_radargram.pcolor(IceSlabsTransect['longitude_EPSG_4326'],
                    IceSlabsTransect['depth'],
                    IceSlabsTransect['IceSlabs_Mask'],
                    cmap='gray_r')
ax_radargram.invert_yaxis() #Invert the y axis = avoid using flipud.
ax_radargram.set_aspect(0.0075)
ax_radargram.set_title('Ice slab transect')
### ------------ This is from 'Greenland_Hydrology_Summary.py' ------------ ###

### ------------------------- Load df_2010_2018 --------------------------- ###
#Load 2010-2018 high estimate for RT3 masked
f_20102018_high_RT3_masked = open(path_jullienetal2023_forRT3+'df_20102018_with_elevation_for_RT3_masked_rignotetalregions', "rb")
df_2010_2018_high_RT3_masked = pickle.load(f_20102018_high_RT3_masked)
f_20102018_high_RT3_masked.close
### ------------------------- Load df_2010_2018 --------------------------- ###

### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#Open SAR image
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load SAR data
SAR = rxr.open_rasterio(path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif',
                              masked=True).squeeze() #No need to reproject satelite image
### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###

#Extract x and y coordinates of SAR image
x_coord_SAR=np.asarray(SAR.x)
y_coord_SAR=np.asarray(SAR.y)

#Define extents of SAR image
extent_SAR = [np.min(x_coord_SAR), np.max(x_coord_SAR), np.min(y_coord_SAR), np.max(y_coord_SAR)]#[west limit, east limit., south limit, north limit]

### -------------- Proof that SAR data are oversampled, 5x5 -------------- ###
#Define bounds of Emaxs in this box
x_min=-130600
x_max=-128610
y_min=-2525301
y_max=-2524707
    
#Extract x and y coordinates of upsampled SAR image
x_coord_focus=np.asarray(SAR.x)
y_coord_focus=np.asarray(SAR.y)

#Extract coordinates of NDWI image within Emaxs bounds
logical_x_coord_within_bounds=np.logical_and(x_coord_focus>=x_min,x_coord_focus<=x_max)
x_coord_within_bounds=x_coord_focus[logical_x_coord_within_bounds]
logical_y_coord_within_bounds=np.logical_and(y_coord_focus>=y_min,y_coord_focus<=y_max)
y_coord_within_bounds=y_coord_focus[logical_y_coord_within_bounds]

#prepare figure
fig_focus = plt.figure()
fig_focus.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_focus = plt.subplot(projection=crs)
SAR[logical_y_coord_within_bounds,logical_x_coord_within_bounds].plot(ax=ax_focus,edgecolor='black')

#Set similar x and y limits
ax_focus.set_xlim(x_min,x_max)
ax_focus.set_ylim(y_min,y_max)
ax_focus.set_title('Proof of oversampling')

#From this by looking at logical_x_coord_within_bounds and logical_y_coord_within_bounds
#I conclude that the offset is 1 along x, and 2 along y
### -------------- Proof that SAR data are oversampled, 5x5 -------------- ###

'''
old way of display
ax_orig.imshow(SAR, extent=extent_SAR, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=-12,vmax=-4)
#Set similar x and y limits
ax_orig.set_xlim(-130600.50763509057, -128610.96548613739)
ax_orig.set_ylim(-2525301.458871396, -2524707.022497623)
'''

#The SAR data are oversampled, i.e. 5x5 matrix should be resample to 1x1. I must take the center of the matrix, i.e. at loc[2:2], as well as consider any offset
offset_row=2
offset_col=1
SAR_upsampled=SAR[np.arange(offset_row+2,SAR.shape[0]-5-offset_row,5),np.arange(offset_col+2,SAR.shape[1]-5-offset_col,5)]

#Extract x and y coordinates of upsampled SAR image
x_coord_SAR_upsampled=np.asarray(SAR_upsampled.x)
y_coord_SAR_upsampled=np.asarray(SAR_upsampled.y)

#Extract coordinates of NDWI image within Emaxs bounds
logical_x_coord_within_bounds_up=np.logical_and(x_coord_SAR_upsampled>=x_min,x_coord_SAR_upsampled<=x_max)
x_coord_within_bounds_up=x_coord_SAR_upsampled[logical_x_coord_within_bounds_up]
logical_y_coord_within_bounds_up=np.logical_and(y_coord_SAR_upsampled>=y_min,y_coord_SAR_upsampled<=y_max)
y_coord_within_bounds_up=y_coord_SAR_upsampled[logical_y_coord_within_bounds_up]

#Check upsampling worked - yes it does!
fig_upsampled = plt.figure()
fig_upsampled.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_upsampled= plt.subplot(projection=crs)

#Display upsampled SAR
SAR_upsampled[logical_y_coord_within_bounds_up,logical_x_coord_within_bounds_up].plot(ax=ax_upsampled,edgecolor='black')

#Set similar x and y limits
ax_upsampled.set_xlim(x_min,x_max)
ax_upsampled.set_ylim(y_min,y_max)
ax_upsampled.set_title('Upsampled - offset_row:'+str(offset_row)+', offset_col:'+str(offset_col))

'''
old way of display

#Define extents of upsampled SAR image
extent_SAR_upsampled = [np.min(x_coord_SAR_upsampled), np.max(x_coord_SAR_upsampled), np.min(y_coord_SAR_upsampled), np.max(y_coord_SAR_upsampled)]#[west limit, east limit., south limit, north limit]

ax_downsampled.imshow(SAR_upsampled, extent=extent_SAR_upsampled, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=-12,vmax=-4)
#Set similar x and y limits
ax_downsampled.set_xlim(-130600.50763509057, -128610.96548613739)
ax_downsampled.set_ylim(-2525301.458871396, -2524707.022497623)
'''
pdb.set_trace()

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
cbar=ax1.imshow(SAR, extent=extent_SAR, transform=crs, origin='upper', cmap='Blues',zorder=1)

#Select data belonging to the list
Extraction_SAR_transect=df_2010_2018_high_RT3_masked[df_2010_2018_high_RT3_masked.Track_name.isin(IceSlabsTransect_list)]

### !!! Keep in mind the df_2010_2018_high_RT3_masked dataset should be clipped
### to the Jullien et al., 2023 ice slabs extent extent shapefile before the overall regression to be done !!!
'''
ax1.scatter(Extraction_SAR_transect['lon_3413'],
            Extraction_SAR_transect['lat_3413'],
            c=Extraction_SAR_transect['20m_ice_content_m'],zorder=3)
'''
'''
#Display Ice Slabs transect
ax1.scatter(IceSlabsTransect['longitude_EPSG_3413'],IceSlabsTransect['latitude_EPSG_3413'],c=IceSlabsTransect['latitude_EPSG_3413'])
'''
#Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')
GrIS_drainage_bassins.plot(ax=ax1,facecolor='none',edgecolor='black')

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
'''
#4. Plot relationship SAR VS Ice slabs thickness
Extraction_SAR_transect.plot.scatter(x='SAR',y='20m_ice_content_m')
'''
#3. Deal with oversampling issue
### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###
from shapely.geometry import LineString
from descartes import PolygonPatch

### Extract SAR values in the vicinity of the transect to reduce computation
#3.a. Transform transect of interest into a line
Extraction_SAR_transect_tuple=[tuple(row[['lon_3413','lat_3413']]) for index, row in Extraction_SAR_transect.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
transect_line=LineString(Extraction_SAR_transect_tuple)

#3.b. Create a buffer around this line
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

#3.c. Extract SAR values within the buffer
from shapely.geometry import mapping

#Perform clip between the buffered polygon around the transect and SAR data
SAR_clipped = SAR.rio.clip(buffered_transect_polygon_gpd.geometry.apply(mapping),
                                      # This is needed if your GDF is in a diff CRS than the raster data
                                      buffered_transect_polygon_gpd.crs) #This is from https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/crop-raster-data-with-shapefile-in-python/

SAR_upsampled_clipped = SAR_upsampled.rio.clip(buffered_transect_polygon_gpd.geometry.apply(mapping),
                                      # This is needed if your GDF is in a diff CRS than the raster data
                                      buffered_transect_polygon_gpd.crs) #This is from https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/crop-raster-data-with-shapefile-in-python/

#Check that selecting SAR around transect worked - yes it does!
SAR_upsampled_clipped.plot(ax=ax1,cmap='viridis',zorder=2,edgecolor='red')

'''
#Zoom to make sure the up-sampling do match well with original SAR data
ax1.set_xlim(-130680.87785909555, -128424.44590967303)
ax1.set_ylim(-2525321.209587765, -2524661.0832195827)
'''

buffered_transect_polygon_gpd.plot(ax=ax1,facecolor='none',edgecolor='black',zorder=4)
#plt.close()
pdb.set_trace()

#Check the original grid and the upsampled grip overlap - looks all good to me
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_grid_overlap = plt.subplot(projection=crs)
SAR_clipped.plot(ax=ax_grid_overlap,edgecolor='black')
SAR_upsampled_clipped.plot(ax=ax_grid_overlap,edgecolor='red',alpha=0.5)
buffered_transect_polygon_gpd.plot(ax=ax_grid_overlap,facecolor='none',edgecolor='magenta',zorder=4)
#Display title
ax_grid_overlap.set_title('Original and upsampled grid check')

#Zoom at the start
ax_grid_overlap.set_xlim(np.min(SAR_clipped.x)-100, np.min(SAR_clipped.x)+1000)
ax_grid_overlap.set_ylim(np.min(SAR_clipped.y)-100,np.max(SAR_clipped.y)+100)

pdb.set_trace()
#Zoom at the end
ax_grid_overlap.set_xlim(np.max(SAR_clipped.x)-1000, np.max(SAR_clipped.x)+100)
ax_grid_overlap.set_ylim(np.min(SAR_clipped.y)-100,np.max(SAR_clipped.y)+100)
pdb.set_trace()

plt.close()

#4. Vectorise the upsampled SAR raster
######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
x, y, radar_signal = SAR_upsampled_clipped.x.values, SAR_upsampled_clipped.y.values, SAR_upsampled_clipped.values
x, y = np.meshgrid(x, y)
x, y, radar_signal = x.flatten(), y.flatten(), radar_signal.flatten()

#Convert to geodataframe
SAR_pd = pd.DataFrame.from_dict({'radar_signal': radar_signal, 'x': x, 'y': y})
#The SAR_vector is a geodataframe of points whose coordinates represent the centroid of each cell
SAR_vector = gpd.GeoDataFrame(SAR_pd, geometry=gpd.GeoSeries.from_xy(SAR_pd['x'], SAR_pd['y'], crs=SAR_upsampled_clipped.rio.crs))
#Create a square buffer around each centroid to reconsititute the raster but where each cell is an individual polygon
SAR_grid = SAR_vector.buffer(50, cap_style=3)
#Convert SAR_grid into a geopandas dataframe, where we keep the information of the centroids (i.e. the SAR signal)
SAR_grid_gpd = gpd.GeoDataFrame(SAR_pd,geometry=gpd.GeoSeries(SAR_grid),crs='epsg:3413')#from https://gis.stackexchange.com/questions/266098/how-to-convert-a-geoseries-to-a-geodataframe-with-geopandas
#There is indeed one unique index for each cell in SAR_grid_gpd - it worked!
######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########

#Display centroid of each cell as well as the created polygons and make sure they are correct - looks all good to me!
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_check_centroid = plt.subplot(projection=crs)
#Display the raster SAR upsampled
SAR_upsampled_clipped.plot(ax=ax_check_centroid,edgecolor='black')
#Display the polygons corresponding to SAR upsampled
SAR_grid_gpd.plot(ax=ax_check_centroid,alpha=0.2,edgecolor='red')
#Display the centroid of each polygon
ax_check_centroid.scatter(SAR_pd.x,SAR_pd.y,color='blue')
pdb.set_trace()

#5. Perform the intersection between each cell of the polygonized SAR data and Ice Slabs transect data
### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###
#Convert Extraction_SAR_transect into a geopandas dataframe
Extraction_SAR_transect_gpd = gpd.GeoDataFrame(Extraction_SAR_transect, geometry=gpd.points_from_xy(Extraction_SAR_transect.lon_3413, Extraction_SAR_transect.lat_3413), crs="EPSG:3413")

#Perform the join between ice slabs thickness and SAR data
pointInPolys= gpd.tools.sjoin(Extraction_SAR_transect_gpd, SAR_grid_gpd, predicate="within", how='left') #This is from https://www.matecdev.com/posts/point-in-polygon.html
### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###

#Check extraction is correct - That looks all good to me!
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_check_extraction = plt.subplot(projection=crs)
#Display the raster SAR upsampled
SAR_upsampled_clipped.plot(ax=ax_check_extraction,vmin=-12,vmax=-4)
#Display the extracted SAR signal
ax_check_extraction.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['radar_signal'],vmin=-12,vmax=-4)

pdb.set_trace()
#Drop the SAR column because it is related to the SAR extraction performed before. The column 'radar_signal' is the outcome of the join!
#Note that the values stores in 'radar_signal' and SAR are the same, which prooves that the upsampling is correct!
pointInPolys=pointInPolys.drop(labels=['SAR'],axis='columns')

#6. Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
upsampled_SAR_and_IceSlabs=pointInPolys.groupby('index_right').mean()

#Display the results
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_check_upsampling = plt.subplot(projection=crs)
SAR_grid_gpd.plot(ax=ax_check_upsampling,edgecolor='black')
ax_check_upsampling.scatter(upsampled_SAR_and_IceSlabs.lon_3413,upsampled_SAR_and_IceSlabs.lat_3413,c=upsampled_SAR_and_IceSlabs['20m_ice_content_m'],s=500,vmin=0,vmax=20)
ax_check_upsampling.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['20m_ice_content_m'],vmin=0,vmax=20)

#7. Display the upsampled radar signal strength with upsampled ice content
#Prepare plot
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
ax4 = plt.subplot()
ax4.scatter(upsampled_SAR_and_IceSlabs['radar_signal'],upsampled_SAR_and_IceSlabs['20m_ice_content_m'])


#Display the ice slabs transect
