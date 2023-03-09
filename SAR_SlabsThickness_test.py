# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:46:44 2023

@author: JullienN
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

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def deg_n(x, a, n):
    return a * np.power(x,n)

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
from shapely.geometry import LineString
from descartes import PolygonPatch
import rioxarray
import rasterio
import os
from scipy.optimize import curve_fit

generate_data='FALSE' #If true, generate the individual csv files and figures
load_data='TRUE'
composite='FALSE'
interpolation='TRUE'
fig_display='FALSE' #If TRUE, generate figures

#Define projection
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Define paths where data are stored
path='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'
path_jullienetal2023='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/'
path_rignotetal2016_GrIS_drainage_bassins='C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'

#1. Load SAR data and a transect case study
### ------------------------- Load df_2010_2018 --------------------------- ###
#Load 2010-2018 high estimate
f_20102018_high_cleaned = open(path_jullienetal2023+'final_excel/high_estimate/clipped/df_20102018_with_elevation_high_estimate_rignotetalregions_cleaned', "rb")
df_20102018_high_cleaned = pickle.load(f_20102018_high_cleaned)
f_20102018_high_cleaned.close
### ------------------------- Load df_2010_2018 --------------------------- ###

### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#Open SAR image
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
'''
#Load SAR data
SAR = rxr.open_rasterio(path+'SAR/HV_2017_2018/'+'ref_HV_2017_2018_32_106_100m_dynmask_blended-0000000000-0000000000.tif',
                              masked=True).squeeze() #No need to reproject satelite image
### almost full greenland: 0000-0000, south greenland: 00023296-00000
'''
#Load SAR data - 2019-2022
SAR = rxr.open_rasterio(path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif',
                              masked=True).squeeze() #No need to reproject satelite image

#Extract x and y coordinates of SAR image
x_coord_SAR=np.asarray(SAR.x)
y_coord_SAR=np.asarray(SAR.y)

#Define extents of SAR image
extent_SAR = [np.min(x_coord_SAR), np.max(x_coord_SAR), np.min(y_coord_SAR), np.max(y_coord_SAR)]#[west limit, east limit., south limit, north limit]

if (generate_data=='TRUE'):
    #Generate the csv files and figures of individual relationship
    if (fig_display=='TRUE'):
        ### -------------- Proof that SAR data are not oversampled anymore -------------- ###
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
        ax_focus.set_title('Check data are not oversampled anymore')
        ### -------------- Proof that SAR data are not oversampled anymore -------------- ###
    ### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
    
    #Loop over all the 2018 transects
    for IceSlabsTransect_name in list(df_20102018_high_cleaned[df_20102018_high_cleaned.year==2017].Track_name.unique()):
        print('Treating',IceSlabsTransect_name)
        '''
        if (IceSlabsTransect_name!='20180423_01_056_056'):
            continue
        '''    
        #Open transect file
        f_IceSlabsTransect = open(path_jullienetal2023+'IceSlabs_And_Coordinates/'+IceSlabsTransect_name+'_IceSlabs.pickle', "rb")
        IceSlabsTransect = pickle.load(f_IceSlabsTransect)
        f_IceSlabsTransect.close()
        
        #Transform transect coordinates from WGS84 to EPSG:3413
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
        points=transformer.transform(np.asarray(IceSlabsTransect["longitude_EPSG_4326"]),np.asarray(IceSlabsTransect["latitude_EPSG_4326"]))
        IceSlabsTransect['longitude_EPSG_3413']=points[0]
        IceSlabsTransect['latitude_EPSG_3413']=points[1]
        
        #Compute distances
        IceSlabsTransect['distances']=compute_distances(IceSlabsTransect['longitude_EPSG_3413'],IceSlabsTransect['latitude_EPSG_3413'])
        
        if (fig_display=='TRUE'):
            #Display ice slabs transect
            fig = plt.figure()
            fig.set_size_inches(18, 2) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
            #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
            ax_radargram = plt.subplot()
            ax_radargram.pcolor(IceSlabsTransect['distances']/1000,
                                IceSlabsTransect['depth'],
                                IceSlabsTransect['IceSlabs_Mask'],
                                cmap='gray_r')
            ax_radargram.invert_yaxis() #Invert the y axis = avoid using flipud.
            ax_radargram.set_aspect(0.1)
            ax_radargram.set_title('Ice slab transect')
        ### ------------ This is from 'Greenland_Hydrology_Summary.py' ------------ ###
        
        #2. Extract ice slabs transect of interest
        #Select ice slabs data belonging to the list of interest
        Extraction_SAR_transect=df_20102018_high_cleaned[df_20102018_high_cleaned.Track_name==IceSlabsTransect_name]
        ### Now using the clipped to polygons high estimate datatset
        
        if (fig_display=='TRUE'):
            #Prepare plot
            fig = plt.figure()
            fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
            #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
            ax1 = plt.subplot(projection=crs)
            #Display coastlines
            ax1.coastlines(edgecolor='black',linewidth=0.075)
            ### ---------- --This is from 'Greenland_Hydrology_Summary.py' ------------ ###
            #Display SAR image
            cbar=ax1.imshow(SAR, extent=extent_SAR, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=-4,vmax=0)
            
            ax1.scatter(Extraction_SAR_transect['lon_3413'],
                        Extraction_SAR_transect['lat_3413'],
                        c=Extraction_SAR_transect['20m_ice_content_m'],zorder=3)
            
            #Display Ice Slabs transect
            ax1.scatter(IceSlabsTransect['longitude_EPSG_3413'],IceSlabsTransect['latitude_EPSG_3413'],c=IceSlabsTransect['latitude_EPSG_3413'])
            
            #Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
            GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')
            GrIS_drainage_bassins.plot(ax=ax1,facecolor='none',edgecolor='black')
            #Zoom to make sure the up-sampling do match well with original SAR data
            ax1.set_xlim(-95179, -94133)
            ax1.set_ylim(-2526058, -2524661)
            #Display reference point
            ax1.scatter(-94600.0,-2525000.0)
            ax1.set_title('Original raster')
        
        #3. Extract SAR values in the vicinity of the transect to reduce computation
        ### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###
        #3.a. Transform transect of interest into a line
        Extraction_SAR_transect_tuple=[tuple(row[['lon_3413','lat_3413']]) for index, row in Extraction_SAR_transect.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        transect_line=LineString(Extraction_SAR_transect_tuple)
        
        #3.b. Create a buffer around this line
        buffered_transect_polygon=transect_line.buffer(200,cap_style=3)
        #Convert polygon of buffered_transect_polygon into a geopandas dataframe
        buffered_transect_polygon_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[buffered_transect_polygon]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        
        '''
        #Display buffered_transect_polygon_gpd
        buffered_transect_polygon_gpd.plot(ax=ax1,facecolor='none',edgecolor='red',zorder=4)
        '''
        '''
        #Visualize
        #Create polygon patch of the polygon
        plot_buffered_transect_polygon = PolygonPatch(buffered_transect_polygon,edgecolor='red',facecolor='none',zorder=10)
        #Display plot_buffered_transect_polygon
        ax1.add_patch(plot_buffered_transect_polygon)
        #ax1.scatter(Extraction_SAR_transect['lon_3413'],Extraction_SAR_transect['lat_3413'],color='black')
        '''
        #plt.savefig(path+'SAR/HV_2017_2018/original_raster.png',dpi=300,bbox_inches='tight')
        #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        ### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###
        
        #3.c. Extract SAR values within the buffer - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html
        #Clip SAR data to the buffered polygon
        
        try:#from https://docs.python.org/3/tutorial/errors.html
            SAR_clipped = SAR.rio.clip(buffered_transect_polygon_gpd.geometry.values, buffered_transect_polygon_gpd.crs, drop=True, invert=False)
            print("Transect intersection with SAR data worked well.")
        except rioxarray.exceptions.NoDataInBounds:#From https://corteva.github.io/rioxarray/html/_modules/rioxarray/exceptions.html#NoDataInBounds
            print("Transect do not intersect with SAR data. Interrupt and go to next transect ...")
            continue
            
        #Define extents of SAR_clipped image
        extent_SAR_clipped = [np.min(np.asarray(SAR_clipped.x)), np.max(np.asarray(SAR_clipped.x)),
                              np.min(np.asarray(SAR_clipped.y)), np.max(np.asarray(SAR_clipped.y))]#[west limit, east limit., south limit, north limit]
        
        #4. Vectorise the SAR raster
        ######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
        x, y, radar_signal = SAR_clipped.x.values, SAR_clipped.y.values, SAR_clipped.values
        x, y = np.meshgrid(x, y)
        x, y, radar_signal = x.flatten(), y.flatten(), radar_signal.flatten()
        
        #Convert to geodataframe
        SAR_pd = pd.DataFrame.from_dict({'radar_signal': radar_signal, 'x': x, 'y': y})
        #The SAR_vector is a geodataframe of points whose coordinates represent the centroid of each cell
        SAR_vector = gpd.GeoDataFrame(SAR_pd, geometry=gpd.GeoSeries.from_xy(SAR_pd['x'], SAR_pd['y'], crs=SAR_clipped.rio.crs))
        #Create a square buffer around each centroid to reconsititute the raster but where each cell is an individual polygon
        SAR_grid = SAR_vector.buffer(50, cap_style=3)
        #Convert SAR_grid into a geopandas dataframe, where we keep the information of the centroids (i.e. the SAR signal)
        SAR_grid_gpd = gpd.GeoDataFrame(SAR_pd,geometry=gpd.GeoSeries(SAR_grid),crs='epsg:3413')#from https://gis.stackexchange.com/questions/266098/how-to-convert-a-geoseries-to-a-geodataframe-with-geopandas
        #There is indeed one unique index for each cell in SAR_grid_gpd - it worked!
        ######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
        
        '''
        #Export the grid to check on QGIS
        SAR_grid_gpd.to_file(path+'SAR/HV_2017_2018/SAR_grid.shp')
        '''
        
        if (fig_display=='TRUE'):
            #Display centroid of each cell as well as the created polygons and make sure they are correct
            #Prepare plot
            fig = plt.figure()
            fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
            #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
            ax_check_centroid = plt.subplot(projection=crs)
            #Display the raster SAR upsampled
            ax_check_centroid.imshow(SAR_clipped, extent=extent_SAR_clipped, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=-4,vmax=0)
            #Display the polygons corresponding to SAR upsampled
            SAR_grid_gpd.plot(ax=ax_check_centroid,alpha=0.2,facecolor='none',edgecolor='red')
            #Display the centroid of each polygon
            ax_check_centroid.scatter(SAR_pd.x,SAR_pd.y,color='blue')
            #Display buffered_transect_polygon_gpd
            buffered_transect_polygon_gpd.plot(ax=ax_check_centroid,facecolor='none',edgecolor='red',zorder=4)
            ax_check_centroid.set_title('Clipped SAR data and corresponding vector grid')
            #This does not look correct, but I am convinced this is due to a matplotlib diplay
            #plt.savefig(path+'SAR/HV_2017_2018/clipped_without_NaNs_raster_and_grid.png',dpi=300,bbox_inches='tight')
        
        #Additional proof the extracted SAR signal with this method is correct using another way of extracting SAR signal
        ### --- This is inspired from 'extract_elevation.py' from paper 'Greenland Ice Slabs Expansion and Thickening --- ###
        #https://towardsdatascience.com/reading-and-visualizing-geotiff-images-with-python-8dcca7a74510
        '''
        #Load SAR data
        path_SAR = path+'SAR/HV_2017_2018/'+'ref_HV_2017_2018_32_106_100m_dynmask_blended-0000000000-0000000000.tif'
        SAR_RIO = rasterio.open(path_SAR)
        '''
        #Load SAR data - 2019-2022
        path_SAR=path+'SAR/'+'ref_2019_2022_61_106_SWGrIS_20m-0000023296-0000000000.tif'
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
        #pdb.set_trace()
        #Drop the SAR column because it is related to the SAR extraction performed before. The column 'radar_signal' is the outcome of the join!
        #Note that the values stores in 'radar_signal' and SAR are the same, which prooves that the upsampling is correct!
        pointInPolys=pointInPolys.drop(labels=['SAR'],axis='columns')
        '''
        
        #5. Perform the intersection between each cell of the polygonized SAR data and Ice Slabs transect data
        ### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###
        #Convert Extraction_SAR_transect into a geopandas dataframe
        Extraction_SAR_transect_gpd = gpd.GeoDataFrame(Extraction_SAR_transect, geometry=gpd.points_from_xy(Extraction_SAR_transect.lon_3413, Extraction_SAR_transect.lat_3413), crs="EPSG:3413")
                
        #Perform the join between ice slabs thickness and SAR data
        pointInPolys= gpd.tools.sjoin(Extraction_SAR_transect_gpd, SAR_grid_gpd, predicate="within", how='left') #This is from https://www.matecdev.com/posts/point-in-polygon.html
        ### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###
                
        if (fig_display=='TRUE'):
            #Make sure SAR extraction between the two methods is identical
            fig = plt.figure()
            ax_SAR_VS_radar_signal = plt.subplot()
            ax_SAR_VS_radar_signal.scatter(Extraction_SAR_transect['SAR'],pointInPolys['radar_signal'])
        
            #Check extraction is correct
            #Prepare plot
            fig = plt.figure()
            fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
            #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
            ax_check_extraction_clipped = plt.subplot(projection=crs)
            #Display the raster SAR upsampled
            ax_check_extraction_clipped.imshow(SAR_clipped, extent=extent_SAR_clipped, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=-4,vmax=0)
            #Display the extracted SAR signal
            ax_check_extraction_clipped.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['radar_signal'],cmap='Blues',vmin=-4,vmax=0,edgecolor='black')
            ax_check_extraction_clipped.set_title('Clipped SAR data and corresponding extracted SAR signal')
            #This does not look correct, but I am convinced this is due to a matplotlib diplay
            #plt.savefig(path+'SAR/HV_2017_2018/clipped_without_NaNs_raster_and_extract_SAR.png',dpi=300,bbox_inches='tight')
        
        #6. Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
        upsampled_SAR_and_IceSlabs=pointInPolys.groupby('index_right').mean()
        
        if (fig_display=='TRUE'):
            #Display upsampling worked
            fig = plt.figure()
            fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
            #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
            ax_check_upsampling = plt.subplot(projection=crs)
            SAR_grid_gpd.plot(ax=ax_check_upsampling,edgecolor='black')
            ax_check_upsampling.scatter(upsampled_SAR_and_IceSlabs.lon_3413,upsampled_SAR_and_IceSlabs.lat_3413,c=upsampled_SAR_and_IceSlabs['20m_ice_content_m'],s=500,vmin=0,vmax=20)
            ax_check_upsampling.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['20m_ice_content_m'],vmin=0,vmax=20)
        
        #7. Plot relationship SAR VS Ice slabs thickness: display the oversampled and upsampled radar signal strength with upsampled ice content
        #Prepare plot
        fig_indiv = plt.figure()
        fig_indiv.set_size_inches(14, 8) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        gs = gridspec.GridSpec(6, 10)
        gs.update(wspace=2)
        gs.update(hspace=1)
        ax_oversampled_SAR = plt.subplot(gs[0:4, 0:5])
        ax_upsampled_SAR = plt.subplot(gs[0:4, 5:10])
        ax_radargram = plt.subplot(gs[4:6, 0:10])
        
        #Display
        Extraction_SAR_transect.plot.scatter(x='SAR',y='20m_ice_content_m',ax=ax_oversampled_SAR)
        ax_upsampled_SAR.scatter(upsampled_SAR_and_IceSlabs['radar_signal'],upsampled_SAR_and_IceSlabs['20m_ice_content_m'])
        ax_radargram.pcolor(IceSlabsTransect['distances']/1000,
                            IceSlabsTransect['depth'],
                            IceSlabsTransect['IceSlabs_Mask'],
                            cmap='gray_r')
        #Improve axis
        ax_oversampled_SAR.set_xlim(-15,-4)
        ax_oversampled_SAR.set_ylim(-0.5,20.5)
        ax_upsampled_SAR.set_xlim(-15,-4)
        ax_upsampled_SAR.set_ylim(-0.5,20.5)
        ax_radargram.invert_yaxis() #Invert the y axis = avoid using flipud.
        ax_radargram.set_aspect(0.1)
        ax_radargram.set_title('Ice slab transect')
        #Display titles
        ax_oversampled_SAR.set_title('Original sampling')
        ax_upsampled_SAR.set_title('Upsampled dataset')
        ax_radargram.set_title('Radargram')
        ax_radargram.set_xlabel('Distance [km]')
        ax_radargram.set_ylabel('Depth [m]')
        
        #Save the figure
        fig_indiv.suptitle(IceSlabsTransect_name)
        plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/2019_2022/images/'+IceSlabsTransect_name+'.png',dpi=300,bbox_inches='tight')
            
        #Export the extracted values as csv
        pointInPolys.to_csv('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/2019_2022/csv/'+IceSlabsTransect_name+'.csv',
                            columns=['Track_name', 'lat', 'lon',
                                   '20m_ice_content_m', 'likelihood', 'lat_3413', 'lon_3413', 'key_shp',
                                   'elevation', 'year', 'index_right', 'radar_signal'])
    
    print('Done in generating 2018 data')

pdb.set_trace()
#Once all csv files of SAR extraction are performed, load them
if (load_data=='TRUE'):
    #Path to data
    path_csv_SAR_VS_IceContent='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/2019_2022/csv/'
        
    #List all the files in the folder
    list_composite=os.listdir(path_csv_SAR_VS_IceContent) #this is inspired from https://pynative.com/python-list-files-in-a-directory/
    
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
    n=1000#let's start by excluding drig cell whose occurence is lower than 100 individuals - should have a good reason behind this value!!    
    
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

    plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceContent/relationship/relationship_SAR_IceContent_occurence='+str(n)+'.png',dpi=300,bbox_inches='tight')
    
    
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