# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:00:42 2022

@author: JullienN
"""

def extraction_SAR(polygon_to_be_intersected,SAR_SW_00_00_in_func,SAR_NW_00_00_in_func,SAR_N_00_00_in_func,SAR_N_00_23_in_func):
    not_worked='FALSE'
    ######################################################################
    ############## This is from SAR_SlabsThickness_test.py ###############
    ######################################################################
    #SAR_SW_00_23 and SAR_NW_00_23 are not needed as no slabs in these regions
    
    #3.c. Extract SAR values within the buffer - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html    
    #Suite of try except from https://stackoverflow.com/questions/17322208/multiple-try-codes-in-one-block       
    try:#from https://docs.python.org/3/tutorial/errors.html
        print('Try intersection with SAR ...')
        SAR_clipped = SAR_SW_00_00.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
    except rxr.exceptions.NoDataInBounds:#From https://corteva.github.io/rioxarray/html/_modules/rioxarray/exceptions.html#NoDataInBounds
        print('   Intersection not found, try again ...')
        try:
            SAR_clipped = SAR_NW_00_00.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
        except rxr.exceptions.NoDataInBounds:
            print('      Intersection not found, try again ...')
            try:
                SAR_clipped = SAR_N_00_00.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
            except rxr.exceptions.NoDataInBounds:
                print('         Intersection not found, try again ...')
                try:
                    SAR_clipped = SAR_N_00_23.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
                except rxr.exceptions.NoDataInBounds:
                    print('            Intersection not found!')
                    print('               Continue')
                    not_worked='TRUE'
                    #Store ice slabs transect not intersected with SAR
    
    if (not_worked=='TRUE'):
        SAR_clipped=np.array([-999])
    else:
        print("SAR intersection found!")
        
    return SAR_clipped
   

def SAR_to_vector(SAR_matrix,axplot_SAR):
    #Display clipped SAR    
    extent_SAR_matrix = [np.min(np.asarray(SAR_matrix.x)), np.max(np.asarray(SAR_matrix.x)),
                          np.min(np.asarray(SAR_matrix.y)), np.max(np.asarray(SAR_matrix.y))]#[west limit, east limit., south limit, north limit]
    axplot_SAR.imshow(SAR_matrix, extent=extent_SAR_matrix, transform=crs, origin='upper', cmap='gray',zorder=2,vmin=-20,vmax=0)

    #Convert SAR_clipped to a numpy matrix
    SAR_np=SAR_matrix.to_numpy()
    #Drop NaNs
    SAR_np=SAR_np[~np.isnan(SAR_np)]
    
    return SAR_np

     
'''
Do I want to do the intersection between the SAR and Ice content in the different sectors??? If yes, revisit this section
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
SAR_grid = SAR_vector.buffer(20, cap_style=3)
#Convert SAR_grid into a geopandas dataframe, where we keep the information of the centroids (i.e. the SAR signal)
SAR_grid_gpd = gpd.GeoDataFrame(SAR_pd,geometry=gpd.GeoSeries(SAR_grid),crs='epsg:3413')#from https://gis.stackexchange.com/questions/266098/how-to-convert-a-geoseries-to-a-geodataframe-with-geopandas
#There is indeed one unique index for each cell in SAR_grid_gpd - it worked!
######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########

#Perform the join between ice slabs thickness and SAR data
pointInPolys= gpd.tools.sjoin(Intersection_EmaxBuffer_slabs, SAR_grid_gpd, predicate="within", how='left',lsuffix='left',rsuffix='right') #This is from https://www.matecdev.com/posts/point-in-polygon.html

#6. Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
upsampled_SAR_and_IceSlabs=pointInPolys.groupby('index_right').mean()

#Display upsampling worked
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax_check_upsampling = plt.subplot(projection=crs)
SAR_grid_gpd.plot(ax=ax_check_upsampling,edgecolor='black')
ax_check_upsampling.scatter(upsampled_SAR_and_IceSlabs.lon_3413,upsampled_SAR_and_IceSlabs.lat_3413,c=upsampled_SAR_and_IceSlabs['20m_ice_content_m'],s=500,vmin=0,vmax=20)
ax_check_upsampling.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['20m_ice_content_m'],vmin=0,vmax=20)
'''

'''
#Export the grid to check on QGIS
SAR_grid_gpd.to_file(path+'SAR/HV_2017_2018/SAR_grid.shp')
'''
######################################################################
############## This is from SAR_SlabsThickness_test.py ###############
######################################################################

def save_slabs_as_csv(path_save_IceSlabs,df_to_save,sector,box_number,processed_year):
    df_to_save.to_csv(path_save_IceSlabs+sector+'/IceSlabs_'+sector+'_box_'+str(box_number)+'_year_'+str(processed_year)+'.csv',
                      columns=['Track_name', 'Tracenumber', 'lat', 'lon', 'alongtrack_distance_m',
                             '20m_ice_content_m', 'likelihood', 'lat_3413', 'lon_3413', 'key_shp',
                             'elevation', 'year', 'geometry', 'index_right_polygon', 'FID', 'rev_subs', 'index_right'])
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

#Define palette for areas of interest , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#ff7f7f", 'Above': "#7f7fff", 'InBetween': "#fee391"}

#Type of slabs product
type_slabs='high' #can be high or low

#Define which year to process
desired_year=2019

#Define radius
radius=250

#Define paths
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
path_data=path_switchdrive+'RT3/data/'
path_df_with_elevation=path_data+'export_RT1_for_RT3/'
path_2002_2003=path_switchdrive+'RT1/final_dataset_2002_2018/2002_2003/'

path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_NDWI=path_local+'data/NDWI_RT3_jullien/NDWI/'
path_SAR=path_local+'data/SAR/HV_2017_2018/'
path_save_SAR_IceSlabs=path_local+'SAR_and_IceContent/SAR_sectors/'

### -------------------------- Load shapefiles --------------------------- ###
#Load Rignot et al., 2016 Greenland drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp',rows=slice(51,57,1)) #the regions are the last rows of the shapefile
#Extract indiv regions and create related indiv shapefiles
NW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NW']
CW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='CW']
SW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SW']
NO_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NO']
NE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NE']
SE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SE']
### -------------------------- Load shapefiles --------------------------- ###

#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load NDWI data for display
NDWI_image = rxr.open_rasterio(path_NDWI+'NDWI_p10_'+str(desired_year)+'.vrt',
                              masked=True).squeeze() #No need to reproject satelite image
#Extract x and y coordinates of satellite image
x_coord_NDWI=np.asarray(NDWI_image.x)
y_coord_NDWI=np.asarray(NDWI_image.y)

#Open Boxes from Tedstone and Machguth (2022)
Boxes_Tedstone2022=gpd.read_file(path_data+'Boxes_Tedstone2022/boxes.shp')

#Sort Boxes_Tedstone2022 as a function of FID
Boxes_Tedstone2022=Boxes_Tedstone2022.sort_values(by=['FID'],ascending=True)#from https://sparkbyexamples.com/pandas/pandas-sort-dataframe-by-multiple-columns/

#After manual identification on QGIS, we do not need 1-4, 20, 27, 33, 39-40, 42, 44-53
nogo_polygon=np.concatenate((np.arange(1,4+1),np.arange(20,20+1),np.arange(27,27+1),
                             np.arange(33,33+1),np.arange(39,40+1),np.arange(42,42+1),
                             np.arange(44,53+1)))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

### ---------------------------- Load dataset ---------------------------- ###
############# IS THIS THE CORRECT DATASET TO USE????? #############
#Dictionnaries have already been created, load them
#Load 2010-2018
f_20102018 = open(path_df_with_elevation+'df_20102018_with_elevation_for_RT3_masked_rignotetalregions', "rb")
df_2010_2018 = pickle.load(f_20102018)
f_20102018.close()
############# IS THIS THE CORRECT DATASET TO USE????? #############

#Load 2002-2003 dataset
df_2002_2003=pd.read_csv(path_2002_2003+'2002_2003_green_excel.csv')
### ---------------------------- Load dataset ---------------------------- ###

#Load Emax from Tedstone and Machguth (2022)
Emax_TedMach=pd.read_csv(path_data+'/Emax/xytpd_NDWI_cleaned_2012_16_19_v2.csv',delimiter=',',decimal='.')

#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})

#Define df_2002_2003, df_2010_2018, Emax_TedMach as being a geopandas dataframes
points_2002_2003 = gpd.GeoDataFrame(df_2002_2003, geometry = gpd.points_from_xy(df_2002_2003['lon'],df_2002_2003['lat']),crs="EPSG:3413")
points_ice = gpd.GeoDataFrame(df_2010_2018, geometry = gpd.points_from_xy(df_2010_2018['lon_3413'],df_2010_2018['lat_3413']),crs="EPSG:3413")
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")

#Define empty dataframe
iceslabs_above_selected_overall=pd.DataFrame()
iceslabs_selected_overall=pd.DataFrame()
iceslabs_inbetween_overall=pd.DataFrame()

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
### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###

#Define empty vectors for SAR storing
SAR_within=[]
SAR_below=[]
SAR_above=[]

for indiv_index in Boxes_Tedstone2022.FID:
    
    if (indiv_index in nogo_polygon):
        #Zone excluded form processing, continue
        print(indiv_index,' excluded, continue')
        continue
    
    if (indiv_index < 8):
        continue
        
    print(indiv_index)
    
    #Extract individual polygon
    indiv_polygon=Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_index]
    
    #Prepare plot
    fig = plt.figure()
    fig.set_size_inches(16, 11.3) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(10, 6)
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    ax_sectors = plt.subplot(gs[0:10, 0:2],projection=crs)
    ax_SAR = plt.subplot(gs[0:10, 2:4],projection=crs)
    ax_ice_distrib = plt.subplot(gs[0:3, 4:6])
    ax_SAR_distrib = plt.subplot(gs[3:6, 4:6])
    ax_GrIS = plt.subplot(gs[7:10, 4:6],projection=crs)
    
    #Display coastlines
    ax_GrIS.coastlines(edgecolor='black',linewidth=0.75)
    #Display GrIS drainage bassins and polygon
    NO_rignotetal.plot(ax=ax_GrIS,color='white', edgecolor='black',linewidth=0.5)
    NE_rignotetal.plot(ax=ax_GrIS,color='white', edgecolor='black',linewidth=0.5) 
    SE_rignotetal.plot(ax=ax_GrIS,color='white', edgecolor='black',linewidth=0.5) 
    SW_rignotetal.plot(ax=ax_GrIS,color='white', edgecolor='black',linewidth=0.5) 
    CW_rignotetal.plot(ax=ax_GrIS,color='white', edgecolor='black',linewidth=0.5) 
    NW_rignotetal.plot(ax=ax_GrIS,color='white', edgecolor='black',linewidth=0.5)
    ax_GrIS.axis('off')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    ax_GrIS.set_extent([-634797, 856884, -3345483, -764054], crs=crs)# x0, x1, y0, y1
    gl=ax_GrIS.gridlines(draw_labels=True, xlocs=[-35, -50], ylocs=[65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
    gl.top_labels = False
    ###################### From Tedstone et al., 2022 #####################
    
    #Display polygon
    indiv_polygon.plot(ax=ax_sectors,color='none', edgecolor='black',linewidth=0.5,zorder=1)
    indiv_polygon.plot(ax=ax_GrIS,color='#faf6c8', edgecolor='black',linewidth=0.5)
    indiv_polygon.plot(ax=ax_SAR,color='none', edgecolor='black',linewidth=0.5,zorder=1)
    
    #Intersection between 2002-2003 ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_20022003 = gpd.sjoin(points_2002_2003, indiv_polygon, predicate='within')
    #Intersection between ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_ice = gpd.sjoin(points_ice, indiv_polygon, predicate='within')
    #Intersection between Emax and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Emax = gpd.sjoin(points_Emax, indiv_polygon, predicate='within')
    
    #rename colnames from join procedure to allow joining with Emax polygons
    within_points_ice=within_points_ice.rename(columns={"index_right":"index_right_polygon"})
    
    #Display antecedent ice slabs
    ax_sectors.scatter(within_points_20022003['lon'],within_points_20022003['lat'],color='#bdbdbd',s=1)
    
    for indiv_year in list([desired_year]):#,2012,2016,2019]):
        
        #Define empty dataframe
        subset_iceslabs_selected=pd.DataFrame()
        subset_iceslabs_above_selected=pd.DataFrame(columns=list(points_ice.keys()))

        #Select data of the desired year
        Emax_points=within_points_Emax[within_points_Emax.year==indiv_year]
        
        #We need at least 2 points per polygon to create an Emax line
        if (len(Emax_points)<2):
            continue
        
        #Add a column to add the elevation pickup on WGS84
        Emax_points['elevation_WGS84']=[np.nan]*len(Emax_points)
                
        #Define bounds of Emaxs in this box
        x_min=np.min(Emax_points['x'])-5e4
        x_max=np.max(Emax_points['x'])+5e4
        y_min=np.min(Emax_points['y'])-5e4
        y_max=np.max(Emax_points['y'])+5e4

        #Extract coordinates of NDWI image within Emaxs bounds
        logical_x_coord_within_bounds=np.logical_and(x_coord_NDWI>=x_min,x_coord_NDWI<=x_max)
        x_coord_within_bounds=x_coord_NDWI[logical_x_coord_within_bounds]
        logical_y_coord_within_bounds=np.logical_and(y_coord_NDWI>=y_min,y_coord_NDWI<=y_max)
        y_coord_within_bounds=y_coord_NDWI[logical_y_coord_within_bounds]

        #Define extents based on the bounds
        extent_NDWI = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
        
        #Display NDWI image
        ax_sectors.imshow(NDWI_image[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_NDWI, transform=crs, origin='upper', cmap='Blues',zorder=0,vmin=0,vmax=0.3) #NDWI
        
        #plot all the Emax points of the considered indiv_year
        ax_sectors.scatter(Emax_points['x'],Emax_points['y'],color='black',s=5,zorder=6)
        
        #Select ice slabs thickness to display distribution
        if (indiv_year == 2002):
            #Select ice slabs data from 2002            
            subset_iceslabs=within_points_20022003[within_points_20022003.year==2002]
            #Rename lat and lon columns to match existing routine
            subset_iceslabs=subset_iceslabs.rename(columns={"lat": "lat_3413","lon": "lon_3413"})
        elif (indiv_year == 2003):
            #Select ice slabs data from 2002 and 2003
            subset_iceslabs=within_points_20022003
            #Rename lat and lon columns to match existing routine
            subset_iceslabs=subset_iceslabs.rename(columns={"lat": "lat_3413","lon": "lon_3413"})
        elif (indiv_year == 2011):
            #Select ice slabs data from 2010 and 2011
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,within_points_ice.year==2011)]
        elif(indiv_year == 2012):
            #Select ice slabs data from 2011, 2012
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2011,within_points_ice.year==2012)]
        elif(indiv_year == 2013):
            #Select ice slabs data from 2012, 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,within_points_ice.year==2013)]
        elif(indiv_year == 2014):
            #Select ice slabs data from 2013, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2013,within_points_ice.year==2014)]
        elif (indiv_year == 2015):
            #Select ice slabs data of the closest indiv_year, i.e. 2014 and 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2013,within_points_ice.year==2014)]
        elif (indiv_year == 2016):
            #Select ice slabs data of the the 2 previous closest indiv_year, i.e. 2013 and 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2013,within_points_ice.year==2014)]
        elif (indiv_year == 2017):
            #Select ice slabs data from 2017 and 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,within_points_ice.year==2017)]
        elif (indiv_year == 2018):
            #Select ice slabs data from 2018, 2017
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2017,within_points_ice.year==2018)]
        elif (indiv_year == 2019):
            #Select ice slabs data from 2018, 2017
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2017,within_points_ice.year==2018)]
        else:
            #Select ice slabs data of the current indiv_year
            subset_iceslabs=within_points_ice[within_points_ice.year==indiv_year]
                
        if (len(subset_iceslabs)==0):
            #No slab for this particular year, continue
            continue
        
        #Display antecedent ice slabs
        ax_sectors.scatter(within_points_ice[within_points_ice.year<=indiv_year]['lon_3413'],within_points_ice[within_points_ice.year<=indiv_year]['lat_3413'],color='gray',s=1,zorder=1)
        '''
        #Display the tracks of the current year within the polygon
        ax_sectors.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],color='purple',s=40,zorder=4)
        '''
        ######################### Connect Emax points #########################
        #Keep only Emax points whose box_id is associated with the current box_id
        Emax_points=Emax_points[Emax_points.box_id==indiv_index]
        
        #Emax as tuples
        Emax_tuple=[tuple(row[['x','y']]) for index, row in Emax_points.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        #Connect Emax points between them
        lineEmax= LineString(Emax_tuple) #from https://shapely.readthedocs.io/en/stable/manual.html
        #Display Emax line
        ax_sectors.plot(lineEmax.xy[0],lineEmax.xy[1],zorder=5,color='#a50f15',linewidth=0.5) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ax_SAR.plot(lineEmax.xy[0],lineEmax.xy[1],zorder=5,color='#a50f15',linewidth=1) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ######################### Connect Emax points #########################

        ########################### Polygon within ############################
        #Create a buffer around this line
        buffer_within_Emax = lineEmax.buffer(radius, cap_style=1) #from https://shapely.readthedocs.io/en/stable/code/buffer.py
        #Create polygon patch from this buffer
        plot_buffer_within_Emax = PolygonPatch(buffer_within_Emax,zorder=2,color='grey',alpha=0.1)
        #Display patch
        ax_sectors.add_patch(plot_buffer_within_Emax)        
        #Convert polygon of Emax buffer around connected Emax line into a geopandas dataframe
        Emax_within_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[buffer_within_Emax]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between subset_iceslabs and Emax_polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
        Intersection_EmaxBuffer_slabs = gpd.sjoin(subset_iceslabs, Emax_within_polygon, predicate='within')
        #Plot the result of this selection
        ax_sectors.scatter(Intersection_EmaxBuffer_slabs['lon_3413'],Intersection_EmaxBuffer_slabs['lat_3413'],color='red',s=1,zorder=8)
        ########################### Polygon within ############################
                        
        ################################ Above ################################
        #Define a line for the above upper boundary 4000m away from Emax line        
        if ((indiv_index==7) & (indiv_year==2016)):
            lineEmax_upper_start_pre = lineEmax.parallel_offset(2000, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            lineEmax_upper_start = lineEmax_upper_start_pre.parallel_offset(2000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            #We choose 10km, should we choose another value??
            lineEmax_upper_end_b = lineEmax_upper_start.parallel_offset(6000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
        else:
            lineEmax_upper_start = lineEmax.parallel_offset(4000, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            #We choose 10km, should we choose another value??
            lineEmax_upper_end_a = lineEmax.parallel_offset(5000, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            lineEmax_upper_end_b = lineEmax_upper_end_a.parallel_offset(5000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
            
        #Plot the above upper boundaries        
        ax_sectors.plot(lineEmax_upper_start.xy[0],lineEmax_upper_start.xy[1],zorder=5,color='#045a8d') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ax_sectors.plot(lineEmax_upper_end_b.xy[0],lineEmax_upper_end_b.xy[1],zorder=5,color='#045a8d') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ax_SAR.plot(lineEmax_upper_end_b.xy[0],lineEmax_upper_end_b.xy[1],zorder=5,color='#045a8d') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        
        #Create a polygon with low end begin the Emax line and upper end being the Emax line + 10 km
        polygon_above=Polygon([*list(lineEmax_upper_end_b.coords),*list(lineEmax_upper_start.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
        #Create polygon patch of the polygon above
        plot_buffer_above_Emax = PolygonPatch(polygon_above,zorder=2,color='grey',alpha=0.1)
        #Display patch of polygone above
        ax_sectors.add_patch(plot_buffer_above_Emax)        
        #Convert polygon of Emax buffer above into a geopandas dataframe
        Emax_above_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_above]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between subset_iceslabs and Emax_above_polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        Intersection_EmaxBufferAbove_slabs = gpd.sjoin(subset_iceslabs, Emax_above_polygon, predicate='within')
        #Plot the result of this selection
        ax_sectors.scatter(Intersection_EmaxBufferAbove_slabs['lon_3413'],Intersection_EmaxBufferAbove_slabs['lat_3413'],color='blue',s=1,zorder=8)
        ################################ Above ################################
        
        ############################## In between ##############################
        lineEmax_radius = lineEmax.parallel_offset(radius, 'right', join_style=1)
        ax_sectors.plot(lineEmax_upper_start.xy[0],lineEmax_upper_start.xy[1],zorder=5,color='yellow')
        ax_SAR.plot(lineEmax_upper_start.xy[0],lineEmax_upper_start.xy[1],zorder=5,color='yellow')
        polygon_radius_4000=Polygon([*list(lineEmax_upper_start.coords),*list(lineEmax_radius.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
        plot_buffer_radius_4000 = PolygonPatch(polygon_radius_4000,zorder=2,color='yellow',alpha=0.2)
        #ax_sectors.add_patch(plot_buffer_radius_4000)        
             
        #Convert polygon of polygon_radius_4000 into a geopandas dataframe
        Emax_radius_4000_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_radius_4000]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between subset_iceslabs and Emax_radius_4000_polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        Intersection_Emaxradius4000_slabs = gpd.sjoin(subset_iceslabs, Emax_radius_4000_polygon, predicate='within')
        #Plot the result of this selection
        ax_sectors.scatter(Intersection_Emaxradius4000_slabs['lon_3413'],Intersection_Emaxradius4000_slabs['lat_3413'],color='yellow',s=1,zorder=7)
        '''
        ax_ice_distrib.hist(Intersection_Emaxradius4000_slabs['20m_ice_content_m'],color='yellow',label='In-between',alpha=0.5,bins=np.arange(0,17),density=True)
        '''
        ############################## In between ##############################
        
        ################################ Below ################################
        #Define a line for the below upper boundary 5000m away from Emax line   
        lineEmax_below = lineEmax.parallel_offset(5000, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
        #Plot the below boundaries        
        ax_sectors.plot(lineEmax_below.xy[0],lineEmax_below.xy[1],zorder=5,color='green') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        ax_SAR.plot(lineEmax_below.xy[0],lineEmax_below.xy[1],zorder=5,color='green') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
        #Create a polygon with low end begin the Emax line and upper end being the Emax line - 5000
        polygon_below=Polygon([*list(lineEmax_below.coords),*list(lineEmax_radius.coords)]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
        #Create polygon patch of the polygon below
        plot_buffer_below_Emax = PolygonPatch(polygon_below,zorder=2,color='grey',alpha=0.1)
        #Display patch of polygone below
        ax_sectors.add_patch(plot_buffer_below_Emax)        
        #Convert below polygon into a geopandas dataframe
        Emax_below_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_below]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        ################################ Below ################################
        
        #Set limits
        ax_sectors.set_xlim(np.min(Emax_points['x'])-1e4,
                            np.max(Emax_points['x'])+1e4)
        ax_sectors.set_ylim(np.min(Emax_points['y'])-1e4,
                            np.max(Emax_points['y'])+1e4)
        
        #Custom legend myself for ax_sectors - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
        legend_elements = [Line2D([0], [0], color='#bdbdbd', lw=2, label='2002-03 ice slabs'),
                           Line2D([0], [0], color='gray', lw=2, label='2010-18 ice slabs'),
                           Line2D([0], [0], color='black', lw=2, label='Emax retrieval', marker='o',linestyle='None'),
                           Line2D([0], [0], color='#a50f15', lw=1, label='Connected Emax retrieval'),
                           Line2D([0], [0], color='red', lw=3, label='Ice slabs within'),
                           Line2D([0], [0], color='yellow', lw=1, label='4 km upstream limit'),
                           Line2D([0], [0], color='yellow', lw=3, label='Ice slabs in-between 0-4 km'),
                           Line2D([0], [0], color='blue', lw=1, label='10 km upstream limit'),
                           Line2D([0], [0], color='blue', lw=3, label='Ice slabs above (4-10 km)'),
                           Line2D([0], [0], color='green', lw=1, label='5 km downstream limit')]
        
        fig.suptitle('Box '+str(indiv_index)+ ' - '+str(indiv_year)+' - 2 years running slabs - radius '+str(radius)+' m - cleanedxytpd V2')
        plt.show()
        '''
        ax_sectors.legend(handles=legend_elements)
        plt.legend()
        '''
        
        #Plot ice slabs thickness that are above and within Emax polygons
        ax_ice_distrib.hist(Intersection_EmaxBufferAbove_slabs['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17,0.5),density=True)
        ax_ice_distrib.hist(Intersection_EmaxBuffer_slabs['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17,0.5),density=True)
        ax_ice_distrib.set_xlabel('Ice thickness [m]')
        ax_ice_distrib.set_ylabel('Density [ ]')
        ax_ice_distrib.set_xlim(0,20)
        ax_ice_distrib.yaxis.set_label_position("right")#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.yaxis.tick_right()#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.xaxis.set_label_position("top")#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.xaxis.tick_top()#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.legend()
        
        #Export the extracted values as csv files
        save_slabs_as_csv(path_save_SAR_IceSlabs,Intersection_EmaxBufferAbove_slabs,'above',indiv_index,indiv_year)
        save_slabs_as_csv(path_save_SAR_IceSlabs,Intersection_EmaxBuffer_slabs,'within',indiv_index,indiv_year)
        
        #Save the iceslabs within and above of that polygon into another dataframe for overall plot
        iceslabs_above_selected_overall=pd.concat([iceslabs_above_selected_overall,Intersection_EmaxBufferAbove_slabs])
        iceslabs_selected_overall=pd.concat([iceslabs_selected_overall,Intersection_EmaxBuffer_slabs])
        iceslabs_inbetween_overall=pd.concat([iceslabs_inbetween_overall,Intersection_Emaxradius4000_slabs])
        
        #DEAL WITH PLACES OUTSIDE OF REGIONS OF INTEREST ('Out' CATEGORY!!!!) There should not be Out data
        
        #Extract and store SAR below, within, and above
        indiv_SAR_below_DF=extraction_SAR(Emax_below_polygon,SAR_SW_00_00,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)
        indiv_SAR_within_DF=extraction_SAR(Emax_within_polygon,SAR_SW_00_00,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)
        indiv_SAR_above_DF=extraction_SAR(Emax_above_polygon,SAR_SW_00_00,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)
        
        #Convert into a vector, and display SAR sector
        indiv_SAR_below=SAR_to_vector(indiv_SAR_below_DF,ax_SAR)
        indiv_SAR_within=SAR_to_vector(indiv_SAR_within_DF,ax_SAR)
        indiv_SAR_above=SAR_to_vector(indiv_SAR_above_DF,ax_SAR)
        
        #Adapt limits
        ax_SAR.set_xlim(np.min(Emax_points['x'])-1e4,
                        np.max(Emax_points['x'])+1e4)
        ax_SAR.set_ylim(np.min(Emax_points['y'])-1e4,
                        np.max(Emax_points['y'])+1e4)
                
        #Display SAR distributions
        ax_SAR_distrib.hist(indiv_SAR_below,color='green',label='Below',alpha=0.5,bins=np.arange(-16,1,0.25),density=True)
        ax_SAR_distrib.hist(indiv_SAR_above,color='blue',label='Above',alpha=0.5,bins=np.arange(-16,1,0.25),density=True)
        ax_SAR_distrib.set_xlabel('Signal strength [dB]')
        ax_SAR_distrib.set_ylabel('Density [ ]')
        ax_SAR_distrib.set_xlim(-20,0)
        ax_SAR_distrib.yaxis.set_label_position("right")#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_SAR_distrib.yaxis.tick_right()#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_SAR_distrib.legend()
        
        #Perform SAR extraction with ice slabs data
        
        pdb.set_trace()

        #Save below, from https://www.pythontutorial.net/python-basics/python-write-text-file/
        if (indiv_SAR_below[0]!=-999):
            with open(path_save_SAR_IceSlabs+'below/below_box_'+str(indiv_index)+'_2019.txt', 'w') as f_below:
                for indiv_SAR_below_line in indiv_SAR_below:
                    f_below.write(str(indiv_SAR_below_line))
                    f_below.write('\n')
        
        #save within
        if (indiv_SAR_within[0]!=-999):
            with open(path_save_SAR_IceSlabs+'within/within_box_'+str(indiv_index)+'_2019.txt', 'w') as f_within:
                for indiv_SAR_within_line in indiv_SAR_within:
                    f_within.write(str(indiv_SAR_within_line))
                    f_within.write('\n')
        
        #Save above
        if (indiv_SAR_above[0]!=-999):
            with open(path_save_SAR_IceSlabs+'above/above_box_'+str(indiv_index)+'_2019.txt', 'w') as f_above:
                for indiv_SAR_above_line in indiv_SAR_above:
                    f_above.write(str(indiv_SAR_above_line))
                    f_above.write('\n')                                             
        
        #Perform SAR extraction with ice slabs only in areas of interest! Do that here??
        
        #Save the figure
        plt.savefig(path_save_SAR_IceSlabs+'Emax_IceSlabs_SAR_box'+str(indiv_index)+'_year_'+str(indiv_year)+'_2YearsRunSlabsMasked_radius_'+str(radius)+'m_cleanedxytpdV2_with0mslabs_likelihood.png',dpi=500,bbox_inches='tight')
        #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        
        pdb.set_trace()
        plt.close()


pdb.set_trace()

#Get rid of the 'Out' region
iceslabs_above_selected_overall=iceslabs_above_selected_overall[iceslabs_above_selected_overall.key_shp!='Out']
iceslabs_selected_overall=iceslabs_selected_overall[iceslabs_selected_overall.key_shp!='Out']
iceslabs_inbetween_overall=iceslabs_inbetween_overall[iceslabs_inbetween_overall.key_shp!='Out']

#Save pandas dataframe
path_to_save='C:/Users/jullienn/switchdrive/Private/research/RT3/data/extracted_slabs/'
iceslabs_above_selected_overall.to_csv(path_to_save+'iceslabs_masked_above_Emax_'+str(indiv_year)+'_cleanedxytpdV2_2years.csv')
iceslabs_selected_overall.to_csv(path_to_save+'iceslabs_masked_within_Emax_'+str(indiv_year)+'_cleanedxytpdV2_2years.csv')
iceslabs_inbetween_overall.to_csv(path_to_save+'iceslabs_masked_inbetween_Emax_'+str(indiv_year)+'_cleanedxytpdV2_2years.csv')

pdb.set_trace()


'''

if (indiv_year in list([2002,2003])):
    #Display the ice slabs points that are inside this buffer
    ax_sectors.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
else:
    #Store an empty dataframe with the index so that index is displayed in plot even without data 
    if (len(subset_iceslabs_buffered)==0):
        #No slab for this particular year at these elevations
        subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,indiv_index, np.nan]]),columns=subset_iceslabs_buffered.columns.values)],ignore_index=True)# from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html and https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/
        #From https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean and https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work
        print(str(indiv_index)+' has no data')
        continue
    
    #Display the ice slabs points that are inside this buffer
    ax_sectors.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
    
    #Store subset_iceslabs_buffered 
    subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,subset_iceslabs_buffered],ignore_index=True)
    #From https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean and https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work

plt.show()
'''
