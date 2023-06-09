# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:00:42 2022

@author: JullienN
"""
def create_buffer_polygon(line_input,radius_around_line,ax_plot):
    #Create a buffer around Emax line
    buffer_around_line = line_input.buffer(radius_around_line, cap_style=1) #from https://shapely.readthedocs.io/en/stable/code/buffer.py    
    #Convert polygon of Emax buffer around connected Emax line into a geopandas dataframe
    line_buffer_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[buffer_around_line]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
    #Display buffer
    line_buffer_polygon.plot(ax=ax_plot,zorder=2,color='red',alpha=0.1)#'grey'
    
    return line_buffer_polygon

    
def create_polygon_above(line_input,radius_around_line,distance_start,distance_end,distance_iterator,ax_plot,ax_plot_SAR,color_plot):
    #Perform upper line start creation
    line_upper_start = line_input.parallel_offset(radius_around_line+distance_start, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
    #Perform upper line end creation step by step
    line_upper_end_a = line_input.parallel_offset(radius_around_line+distance_iterator, 'right', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
    final_dist=radius_around_line+distance_iterator
    
    while (final_dist <= (distance_end+radius_around_line)):
        print('   Building above polygon. Distance:',final_dist)
        line_upper_end_a = line_upper_end_a.parallel_offset(distance_iterator, 'left', join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py   
        final_dist=final_dist+distance_iterator
    
    #Create a polygon with upper line start and upper line end
    polygon_upper_start_end=Polygon([*list(line_upper_end_a.coords),*list(line_upper_start.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
    #Convert polygon into a geopandas dataframe
    polygon_upper_start_end_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_upper_start_end]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
    #Display polygon
    polygon_upper_start_end_gpd.plot(ax=ax_plot,zorder=2,color=color_plot,alpha=0.1)#'grey'
    
    #Plot the above upper boundaries
    ax_plot.plot(line_upper_start.xy[0],line_upper_start.xy[1],zorder=5,color=color_plot) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_plot.plot(line_upper_end_a.xy[0],line_upper_end_a.xy[1],zorder=5,color=color_plot) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_plot_SAR.plot(line_upper_start.xy[0],line_upper_start.xy[1],zorder=5,color=color_plot) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_plot_SAR.plot(line_upper_end_a.xy[0],line_upper_end_a.xy[1],zorder=5,color=color_plot) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    
    return polygon_upper_start_end_gpd, line_upper_end_a

def create_polygon_offset(line_input,radius_around_line,distance_end,type_offset,ax_plot,ax_plot_SAR,color_plot):    
    if (type_offset=='upstream'):
        direction='right'
    elif (type_offset=='downstream'):
        direction='left'        
    else:
        print('Enter a correct type of offset, i.e. downstream or upstream')
        pdb.set_trace()
    
    #Perform upper line start creation
    line_upper_start = line_input.parallel_offset(radius_around_line, direction, join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
    #Perform upper line end creation
    line_upper_end = line_input.parallel_offset(radius_around_line+distance_end, direction, join_style=1) #from https://shapely.readthedocs.io/en/stable/code/parallel_offset.py
    #Create a polygon with upper line start and upper line end
    polygon_upper_start_end=Polygon([*list(line_upper_end.coords),*list(line_upper_start.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
    #Convert polygon into a geopandas dataframe
    polygon_upper_start_end_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_upper_start_end]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
    #Display polygon
    polygon_upper_start_end_gpd.plot(ax=ax_plot,zorder=2,color=color_plot,alpha=0.1)#'grey'
    #Plot the above upper boundaries
    ax_plot.plot(line_upper_start.xy[0],line_upper_start.xy[1],zorder=5,color=color_plot,linestyle='dashed') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_plot.plot(line_upper_end.xy[0],line_upper_end.xy[1],zorder=6,color=color_plot,linestyle='dashed') #From https://shapely.readthedocs.io/en/stable/code/linestring.py

    ax_plot_SAR.plot(line_upper_start.xy[0],line_upper_start.xy[1],zorder=5,color=color_plot,linestyle='dashed') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_plot_SAR.plot(line_upper_end.xy[0],line_upper_end.xy[1],zorder=6,color=color_plot,linestyle='dashed') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    
    return polygon_upper_start_end_gpd,line_upper_end


def perform_extraction_in_polygon(dataframe_to_intersect,polygon_for_intersection,ax_plot,color_plot):                
    #Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
    Intersection_polygon_dataframe_slabs = gpd.sjoin(dataframe_to_intersect, polygon_for_intersection, predicate='within')
    #Plot the result of this intersection
    ax_plot.scatter(Intersection_polygon_dataframe_slabs['lon_3413'],Intersection_polygon_dataframe_slabs['lat_3413'],color=color_plot,s=1,zorder=8)
    
    return Intersection_polygon_dataframe_slabs

def extraction_SAR(polygon_to_be_intersected,SAR_SW_00_00_in_func,SAR_N_00_00_EW_in_func,SAR_NW_00_00_in_func,SAR_N_00_00_in_func,SAR_N_00_23_in_func,polygon_in_use):
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
            SAR_clipped = SAR_N_00_00_EW_in_func.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
        except rxr.exceptions.NoDataInBounds:
            print('      Intersection not found, try again ...')
            try:
                SAR_clipped = SAR_NW_00_00.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
            except rxr.exceptions.NoDataInBounds:
                print('         Intersection not found, try again ...')
                try:
                    SAR_clipped = SAR_N_00_00.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
                except rxr.exceptions.NoDataInBounds:
                    print('            Intersection not found, try again ...')
                    try:
                        SAR_clipped = SAR_N_00_23.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
                    except rxr.exceptions.NoDataInBounds:
                        print('               Intersection not found!')
                        print('                  Continue')
                        not_worked='TRUE'
                        #Store ice slabs transect not intersected with SAR
    
    if (not_worked=='TRUE'):
        SAR_clipped_within_polygon=np.array([-999])
    else:
        print("SAR intersection found!")        
        #Perform clip between indiv box and SAR to get rid of SAR data outside of box
        SAR_clipped_within_polygon = SAR_clipped.rio.clip(polygon_in_use.geometry.values, polygon_in_use.crs, drop=True, invert=False)
        
        '''
        #Check clip performed well: Yes it did!
        fig = plt.figure()
        gs = gridspec.GridSpec(10, 6)
        ax_1 = plt.subplot(gs[0:10, 0:6],projection=crs)
        polygon_in_use.plot(ax=ax_1,alpha=0.5)
        #Display clipped SAR    
        extent_SAR_clipped = [np.min(np.asarray(SAR_clipped.x)), np.max(np.asarray(SAR_clipped.x)),
                              np.min(np.asarray(SAR_clipped.y)), np.max(np.asarray(SAR_clipped.y))]#[west limit, east limit., south limit, north limit]
        ax_1.imshow(SAR_clipped, extent=extent_SAR_clipped, transform=crs, origin='upper', cmap='gray',zorder=2,vmin=-20,vmax=0)

        fig = plt.figure()
        gs = gridspec.GridSpec(10, 6)
        ax_2 = plt.subplot(gs[0:10, 0:6],projection=crs)
        polygon_in_use.plot(ax=ax_2,alpha=0.5)
        #Display clipped SAR    
        extent_SAR_clipped_within_polygon = [np.min(np.asarray(SAR_clipped_within_polygon.x)), np.max(np.asarray(SAR_clipped_within_polygon.x)),
                                             np.min(np.asarray(SAR_clipped_within_polygon.y)), np.max(np.asarray(SAR_clipped_within_polygon.y))]#[west limit, east limit., south limit, north limit]
        ax_2.imshow(SAR_clipped_within_polygon, extent=extent_SAR_clipped_within_polygon, transform=crs, origin='upper', cmap='gray',zorder=2,vmin=-20,vmax=0)
        '''
    
    return SAR_clipped_within_polygon
   

def SAR_to_vector(SAR_matrix,axplot_SAR):
    #Display clipped SAR    
    extent_SAR_matrix = [np.min(np.asarray(SAR_matrix.x)), np.max(np.asarray(SAR_matrix.x)),
                          np.min(np.asarray(SAR_matrix.y)), np.max(np.asarray(SAR_matrix.y))]#[west limit, east limit., south limit, north limit]
    axplot_SAR.imshow(SAR_matrix, extent=extent_SAR_matrix, transform=crs, origin='upper', cmap='gray',zorder=2,vmin=-20,vmax=0)
    
    #Convert SAR_clipped to a numpy matrix
    SAR_np=SAR_matrix.to_numpy()
    #Drop NaNs
    SAR_np=SAR_np[~np.isnan(SAR_np)]
    
    #Identify index of non NaN cells
    index_x_y_noNaN=np.where(~np.isnan(SAR_matrix.data))
    #Extract x and y coord where SAR not NaN
    xcoord_SAR=SAR_matrix.x[index_x_y_noNaN[1]].to_numpy()
    ycoord_SAR=SAR_matrix.y[index_x_y_noNaN[0]].to_numpy()
    
    #Make it a pandas df storing x, y and SAR data
    pd_return=pd.DataFrame(data={'x_coord_SAR': xcoord_SAR, 'y_coord_SAR': ycoord_SAR, 'SAR': SAR_np})
    
    return pd_return


def SAR_raster_to_polygon(SAR_to_vectorize):
    ######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
    x, y, radar_signal = SAR_to_vectorize.x.values, SAR_to_vectorize.y.values, SAR_to_vectorize.values
    x, y = np.meshgrid(x, y)
    x, y, radar_signal = x.flatten(), y.flatten(), radar_signal.flatten()
    
    #Convert to geodataframe
    SAR_pd = pd.DataFrame.from_dict({'radar_signal': radar_signal, 'x': x, 'y': y})
    #The SAR_vector is a geodataframe of points whose coordinates represent the centroid of each cell
    SAR_vector = gpd.GeoDataFrame(SAR_pd, geometry=gpd.GeoSeries.from_xy(SAR_pd['x'], SAR_pd['y'], crs=SAR_to_vectorize.rio.crs))
    #Create a square buffer around each centroid to reconsititute the raster but where each cell is an individual polygon
    SAR_grid = SAR_vector.buffer(20, cap_style=3)
    #Convert SAR_grid into a geopandas dataframe, where we keep the information of the centroids (i.e. the SAR signal)
    SAR_grid_gpd = gpd.GeoDataFrame(SAR_pd,geometry=gpd.GeoSeries(SAR_grid),crs='epsg:3413')#from https://gis.stackexchange.com/questions/266098/how-to-convert-a-geoseries-to-a-geodataframe-with-geopandas
    #There is indeed one unique index for each cell in SAR_grid_gpd - it worked!
    ######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
    return SAR_grid_gpd


def save_slabs_as_csv(path_save_IceSlabs,df_to_save,sector,box_number,processed_year):
    df_to_save.to_csv(path_save_IceSlabs+sector+'/IceSlabs_'+sector+'_box_'+str(box_number)+'_year_'+str(processed_year)+'.csv',
                      columns=['Track_name', 'Tracenumber', 'lat', 'lon', 'alongtrack_distance_m',
                             '20m_ice_content_m', 'likelihood', 'lat_3413', 'lon_3413', 'key_shp',
                             'elevation', 'year', 'geometry', 'index_right_polygon', 'FID', 'rev_subs', 'index_right'])
    return


def perform_processing(Emax_points_func,subset_iceslabs_func,radius_func,indiv_polygon_func,SAR_SW_00_00_func,SAR_N_00_00_EW_func,SAR_NW_00_00_func,SAR_N_00_00_func,SAR_N_00_23_func):
    #Emax as tuples
    Emax_tuple=[tuple(row[['x','y']]) for index, row in Emax_points_func.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
    #Connect Emax points between them
    lineEmax= LineString(Emax_tuple) #from https://shapely.readthedocs.io/en/stable/manual.html
    #Display Emax line
    ax_sectors.plot(lineEmax.xy[0],lineEmax.xy[1],zorder=5,color='#a50f15',linewidth=0.5) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_SAR.plot(lineEmax.xy[0],lineEmax.xy[1],zorder=5,color='#a50f15',linewidth=1) #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ######################### Connect Emax points #########################
        
    ################ Create polygons and extract ice slabs ################
    #Perform polygon above creation and slabs extraction
    above_polygon,upper_limit=create_polygon_above(lineEmax,radius_func,4000,9000,1000,ax_sectors,ax_SAR,'#045a8d')
    Intersection_slabs_above = perform_extraction_in_polygon(subset_iceslabs_func,above_polygon,ax_sectors,'blue')
    
    #Perform polygon in-between creation and slabs extraction
    in_between_polygon,lower_inbetween_limit=create_polygon_offset(lineEmax,radius_func,4000,'upstream',ax_sectors,ax_SAR,'yellow')
    Intersection_slabs_InBetween = perform_extraction_in_polygon(subset_iceslabs_func,in_between_polygon,ax_sectors,'yellow')
    
    #Perform polygon within creation and slabs extraction
    within_polygon = create_buffer_polygon(lineEmax,radius_func,ax_sectors)
    Intersection_slabs_within = perform_extraction_in_polygon(subset_iceslabs_func,within_polygon,ax_sectors,'red')
    
    #Perform polygon below creation and slabs extraction
    below_polygon,lower_limit=create_polygon_offset(lineEmax,radius_func,5000,'downstream',ax_sectors,ax_SAR,'green')
    Intersection_slabs_below = perform_extraction_in_polygon(subset_iceslabs_func,below_polygon,ax_sectors,'green')
    
    #Perform polygon from below to above creation
    polygon_below_to_above=Polygon([*list(upper_limit.coords),*list(lower_limit.coords)]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely
    #Convert from below to above polygon into a geopandas dataframe
    below_above_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_below_to_above]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
    '''
    #Display from below to above gpd
    below_above_gpd.plot(ax=ax_sectors,alpha=0.2,color='magenta',zorder=10)
    ax_sectors.plot(lower_limit.xy[0],lower_limit.xy[1],zorder=15,color='magenta',linestyle='dashed') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    ax_sectors.plot(upper_limit.xy[0],upper_limit.xy[1],zorder=15,color='magenta',linestyle='dashed') #From https://shapely.readthedocs.io/en/stable/code/linestring.py
    '''
    ################ Create polygons and extract ice slabs ################
    
    ########################## Extract SAR data ##########################
    #Extract and store SAR from below to above
    indiv_SAR_below_above_DF=extraction_SAR(below_above_gpd,SAR_SW_00_00_func,SAR_N_00_00_EW_func,SAR_NW_00_00_func,SAR_N_00_00_func,SAR_N_00_23_func,indiv_polygon_func)
        
    if (len(indiv_SAR_below_above_DF)==1):
        print('No intersection with SAR data, continue')
    else:
        #Perform clip between SAR_below_above with the individual sectors - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html    
        try:
            indiv_SAR_above_DF = indiv_SAR_below_above_DF.rio.clip(above_polygon.geometry.values, above_polygon.crs, drop=True, invert=False)
            #Convert SAR data into a vector, and display SAR sector
            indiv_SAR_above=SAR_to_vector(indiv_SAR_above_DF,ax_SAR)
            
            if (len(indiv_SAR_above)>1):
                #there is data, continue performing tasks
                indiv_SAR_above_return=indiv_SAR_above
            else:
                indiv_SAR_above_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
        except rxr.exceptions.NoDataInBounds:
            indiv_SAR_above_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
            print('No SAR above')
        
        try:
            indiv_SAR_inbetween_DF = indiv_SAR_below_above_DF.rio.clip(in_between_polygon.geometry.values, in_between_polygon.crs, drop=True, invert=False)
            #Convert SAR data into a vector, and display SAR sector
            indiv_SAR_inbetween=SAR_to_vector(indiv_SAR_inbetween_DF,ax_SAR)
            
            if (len(indiv_SAR_inbetween)>1):
                #there is data, continue performing tasks
                indiv_SAR_inbetween_return=indiv_SAR_inbetween
            else:
                indiv_SAR_inbetween_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
        except rxr.exceptions.NoDataInBounds:
            indiv_SAR_inbetween_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
            print('No SAR in-between')

        try:
            indiv_SAR_within_DF = indiv_SAR_below_above_DF.rio.clip(within_polygon.geometry.values, within_polygon.crs, drop=True, invert=False)
            #Convert SAR data into a vector, and display SAR sector
            indiv_SAR_within=SAR_to_vector(indiv_SAR_within_DF,ax_SAR)
            
            if (len(indiv_SAR_within)>1):
                #there is data, continue performing tasks
                indiv_SAR_within_return=indiv_SAR_within
            else:
                indiv_SAR_within_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
        except rxr.exceptions.NoDataInBounds:
            indiv_SAR_within_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
            print('No SAR within')

        try:
            indiv_SAR_below_DF = indiv_SAR_below_above_DF.rio.clip(below_polygon.geometry.values, below_polygon.crs, drop=True, invert=False)
            #Convert SAR data into a vector, and display SAR sector
            indiv_SAR_below=SAR_to_vector(indiv_SAR_below_DF,ax_SAR)
            
            if (len(indiv_SAR_below)>1):
                #there is data, continue performing tasks
                indiv_SAR_below_return=indiv_SAR_below
            else:
                indiv_SAR_below_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
        except rxr.exceptions.NoDataInBounds:
            indiv_SAR_below_return=pd.DataFrame(data={'x_coord_SAR': [], 'y_coord_SAR': [], 'SAR': []})
            print('No SAR below')
    
    return Intersection_slabs_above,Intersection_slabs_InBetween,Intersection_slabs_within,Intersection_slabs_below,indiv_SAR_above_return,indiv_SAR_inbetween_return,indiv_SAR_within_return,indiv_SAR_below_return

    
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
radius=500

#Define paths
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
path_data=path_switchdrive+'RT3/data/'
path_df_with_elevation=path_data+'export_RT1_for_RT3/'
path_2002_2003=path_switchdrive+'RT1/final_dataset_2002_2018/2002_2003/'

path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_NDWI=path_local+'data/NDWI_RT3_jullien/NDWI/'
path_SAR=path_local+'data/SAR/HV_2017_2018/'
path_save_SAR_IceSlabs=path_local+'SAR_and_IceThickness/SAR_sectors/'

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

# Consider ice slabs and SAR, we do not go through polygon where
# firn aquifer are no prominent, i.e. all boxes except 1-3, 20, 42, 44-53.
# Considering also complex topography, the final list of polygon we do not go tghrough is: 1-3, 20, 32-53
nogo_polygon=np.concatenate((np.arange(1,3+1),np.arange(20,20+1),np.arange(32,53+1)))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

### ---------------------------- Load dataset ---------------------------- ###
#Dictionnaries have already been created, load them
#Load 2010-2018
f_20102018 = open(path_df_with_elevation+'df_20102018_with_elevation_for_RT3_masked_rignotetalregions', "rb")
df_2010_2018 = pickle.load(f_20102018)
f_20102018.close()
#load 2010-2018 ice slabs high end extent from Jullien et al., (2023)
IceSlabsExtent_20102018_jullienetal2023=gpd.read_file(path_switchdrive+'RT1/final_dataset_2002_2018/shapefiles/iceslabs_jullien_highend_20102018.shp')

#Load 2002-2003 dataset
df_2002_2003=pd.read_csv(path_2002_2003+'2002_2003_green_excel.csv')
### ---------------------------- Load dataset ---------------------------- ###

#Load Emax from Tedstone and Machguth (2022)
Emax_TedMach=pd.read_csv(path_data+'/Emax/xytpd_NDWI_cleaned_2019_v3.csv',delimiter=',',decimal='.')

#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})

#Define df_2002_2003, df_2010_2018, Emax_TedMach as being a geopandas dataframes
points_2002_2003 = gpd.GeoDataFrame(df_2002_2003, geometry = gpd.points_from_xy(df_2002_2003['lon'],df_2002_2003['lat']),crs="EPSG:3413")
points_ice = gpd.GeoDataFrame(df_2010_2018, geometry = gpd.points_from_xy(df_2010_2018['lon_3413'],df_2010_2018['lat_3413']),crs="EPSG:3413")
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")

#Define empty dataframe
iceslabs_above_selected_overall=pd.DataFrame()
iceslabs_inbetween_overall=pd.DataFrame()
iceslabs_within_overall=pd.DataFrame()
iceslabs_below_overall=pd.DataFrame()

#Open SAR image
### --- This is from Fisg4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load SAR data
SAR_N_00_00_EW = rxr.open_rasterio(path_SAR+'ref_EW_HV_2017_2018_32_106_40m_ASCDESC_N_nscenes0_manual-0000000000-0000000000.tif',masked=True).squeeze()#No need to reproject satelite image
SAR_N_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_N_manual-0000000000-0000000000.tif',masked=True).squeeze()
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
    '''
    if (indiv_index <13):
        continue
    '''
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
    
    #Display 2010-2018 high end ice slabs jullien et al., 2023
    IceSlabsExtent_20102018_jullienetal2023.plot(ax=ax_sectors,facecolor='none',edgecolor='#ba2b2b',zorder=10)
    IceSlabsExtent_20102018_jullienetal2023.plot(ax=ax_SAR,facecolor='none',edgecolor='#ba2b2b',zorder=10)

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
        
        #Display antecedent ice slabs
        ax_sectors.scatter(within_points_ice[within_points_ice.year<=indiv_year]['lon_3413'],within_points_ice[within_points_ice.year<=indiv_year]['lat_3413'],color='gray',s=1,zorder=1)
        '''
        #Display the tracks of the current year within the polygon
        ax_sectors.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],color='purple',s=40,zorder=4)
        '''
                
        ######################### Connect Emax points #########################
        #Keep only Emax points whose box_id is associated with the current box_id
        Emax_points=Emax_points[Emax_points.box_id==indiv_index]
        
        #Modify line for some specific case
        if ((indiv_index==7)&(indiv_year==2019)):
            Emax_points.loc[Emax_points.index_Emax==902,'index_Emax']=948
            Emax_points.loc[Emax_points.index_Emax==924,'index_Emax']=949
            #sort
            Emax_points=Emax_points.sort_values('index_Emax')
            
        if ((indiv_index==15)&(indiv_year==2019)):
            #We do not go down to Emaqx point 829 because the above category will include lower elevation places due to decreasing elevations towards the south
            Emax_points=Emax_points[Emax_points.index_Emax<=388]
        
        if ((indiv_index==16)&(indiv_year==2019)):
            #sort
            Emax_points=Emax_points.sort_values('index_Emax')
            
        if ((indiv_index==19)&(indiv_year==2019)):
            Emax_points.loc[Emax_points.index_Emax==715,'index_Emax']=550
            Emax_points.loc[Emax_points.index_Emax==681,'index_Emax']=549
            Emax_points.loc[Emax_points.index_Emax==748,'index_Emax']=548
            #sort
            Emax_points=Emax_points.sort_values('index_Emax',ascending=False)
            
        if ((indiv_index==27)&(indiv_year==2019)):
            #Do not consider points in the north east sector of this polygon
            Emax_points=Emax_points[Emax_points.index_Emax<=772]
            #Modify position for box 27 in 2019, from https://stackoverflow.com/questions/40427943/how-do-i-change-a-single-index-value-in-pandas-dataframe
            Emax_points.loc[Emax_points.index_Emax==637,'index_Emax']=579
            Emax_points.loc[Emax_points.index_Emax==698,'index_Emax']=580
            Emax_points.loc[Emax_points.index_Emax==668,'index_Emax']=581
            Emax_points.loc[Emax_points.index_Emax==607,'index_Emax']=582
            #sort
            Emax_points=Emax_points.sort_values('index_Emax')
        
        if ((indiv_index==32)&(indiv_year==2019)):
            #Do not consider points in the north east sector of this polygon
            Emax_points=Emax_points[Emax_points.index_Emax<=2064]
        
        #Perform task of polygons creation, ice slabs and SAR extraction
        if ((indiv_index==25)&(indiv_year==2019)):
            #For box 25, need to divide the box into two independant suite of Emax points
            #Sect 1
            Emax_points_sect1=Emax_points[Emax_points.index_Emax>=237]
            Intersection_slabs_above_out_sect1,Intersection_slabs_InBetween_out_sect1,Intersection_slabs_within_out_sect1,Intersection_slabs_below_out_sect1,indiv_SAR_above_out_sect1,indiv_SAR_inbetween_out_sect1,indiv_SAR_within_out_sect1,indiv_SAR_below_out_sect1=perform_processing(Emax_points_sect1,subset_iceslabs,radius,indiv_polygon,SAR_SW_00_00,SAR_N_00_00_EW,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)
            
            #Sect 2
            Emax_points_sect2=Emax_points[Emax_points.index_Emax<=136]
            Intersection_slabs_above_out_sect2,Intersection_slabs_InBetween_out_sect2,Intersection_slabs_within_out_sect2,Intersection_slabs_below_out_sect2,indiv_SAR_above_out_sect2,indiv_SAR_inbetween_out_sect2,indiv_SAR_within_out_sect2,indiv_SAR_below_out_sect2=perform_processing(Emax_points_sect2,subset_iceslabs,radius,indiv_polygon,SAR_SW_00_00,SAR_N_00_00_EW,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)
       
            #Reunite data
            Intersection_slabs_above_out=pd.concat([Intersection_slabs_above_out_sect1,Intersection_slabs_above_out_sect2])
            Intersection_slabs_InBetween_out=pd.concat([Intersection_slabs_InBetween_out_sect1,Intersection_slabs_InBetween_out_sect2])
            Intersection_slabs_within_out=pd.concat([Intersection_slabs_within_out_sect1,Intersection_slabs_within_out_sect2])
            Intersection_slabs_below_out=pd.concat([Intersection_slabs_below_out_sect1,Intersection_slabs_below_out_sect2])
            indiv_SAR_above_out=pd.concat([indiv_SAR_above_out_sect1,indiv_SAR_above_out_sect2])
            indiv_SAR_inbetween_out=pd.concat([indiv_SAR_inbetween_out_sect1,indiv_SAR_inbetween_out_sect2])
            indiv_SAR_within_out=pd.concat([indiv_SAR_within_out_sect1,indiv_SAR_within_out_sect2])
            indiv_SAR_below_out=pd.concat([indiv_SAR_below_out_sect1,indiv_SAR_below_out_sect2])
       
        else:
            Intersection_slabs_above_out,Intersection_slabs_InBetween_out,Intersection_slabs_within_out,Intersection_slabs_below_out,indiv_SAR_above_out,indiv_SAR_inbetween_out,indiv_SAR_within_out,indiv_SAR_below_out=perform_processing(Emax_points,subset_iceslabs,radius,indiv_polygon,SAR_SW_00_00,SAR_N_00_00_EW,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)
        
        ########################## Extract SAR data ##########################
        
        ############# Display ice slabs thickness and SAR signal #############
        #Set limits
        ax_sectors.set_xlim(np.min(Emax_points['x'])-1e4,
                            np.max(Emax_points['x'])+1e4)
        ax_sectors.set_ylim(np.min(Emax_points['y'])-1e4,
                            np.max(Emax_points['y'])+1e4)
        
        ax_SAR.set_xlim(np.min(Emax_points['x'])-1e4,
                        np.max(Emax_points['x'])+1e4)
        ax_SAR.set_ylim(np.min(Emax_points['y'])-1e4,
                        np.max(Emax_points['y'])+1e4)
        
        #Display SAR distributions
        ax_SAR_distrib.hist(indiv_SAR_above_out.SAR,color='blue',label='Above',alpha=0.5,bins=np.arange(-16,1,0.25),density=True)
        ax_SAR_distrib.hist(indiv_SAR_below_out.SAR,color='green',label='Below',alpha=0.5,bins=np.arange(-16,1,0.25),density=True)
        
        #Plot ice slabs thickness that are above, within and below Emax polygons
        ax_ice_distrib.hist(Intersection_slabs_below_out['20m_ice_content_m'],color='green',label='Below',alpha=0.5,bins=np.arange(0,17,0.5),density=True)
        ax_ice_distrib.hist(Intersection_slabs_above_out['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17,0.5),density=True)
        ax_ice_distrib.hist(Intersection_slabs_within_out['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17,0.5),density=True)
        ax_ice_distrib.set_xlabel('Ice thickness [m]')
        ax_ice_distrib.set_ylabel('Density [ ]')
        ax_ice_distrib.set_xlim(0,20)
        ax_ice_distrib.yaxis.set_label_position("right")#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.yaxis.tick_right()#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.xaxis.set_label_position("top")#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.xaxis.tick_top()#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_ice_distrib.legend()
        
        #Improve display of SAR distributions
        ax_SAR_distrib.set_xlabel('Signal strength [dB]')
        ax_SAR_distrib.set_ylabel('Density [ ]')
        ax_SAR_distrib.set_xlim(-20,0)
        ax_SAR_distrib.yaxis.set_label_position("right")#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_SAR_distrib.yaxis.tick_right()#from https://stackoverflow.com/questions/13369888/matplotlib-y-axis-label-on-right-side
        ax_SAR_distrib.legend()
        
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
        
        fig.suptitle('Box '+str(indiv_index)+ ' - '+str(indiv_year)+' - 2 years running slabs - radius '+str(radius)+' m - cleanedxytpd V3')
                
        #Save the figure
        plt.savefig(path_save_SAR_IceSlabs+'Emax_IceSlabs_SAR_box'+str(indiv_index)+'_year_'+str(indiv_year)+'_radius_'+str(radius)+'m_cleanedxytpdV3_with0mslabs.png',dpi=500,bbox_inches='tight')
        #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        
        '''
        ax_sectors.legend(handles=legend_elements)
        plt.legend()
        '''
        ############# Display ice slabs thickness and SAR signal #############
        plt.close()
        
        ### Export extracted Ice Slabs and SAR sectors as csv files ###
        # Export the extracted ice slabs in sectors as csv files
        save_slabs_as_csv(path_save_SAR_IceSlabs,Intersection_slabs_above_out,'above',indiv_index,indiv_year)
        save_slabs_as_csv(path_save_SAR_IceSlabs,Intersection_slabs_InBetween_out,'in_between',indiv_index,indiv_year)
        save_slabs_as_csv(path_save_SAR_IceSlabs,Intersection_slabs_within_out,'within',indiv_index,indiv_year)
        save_slabs_as_csv(path_save_SAR_IceSlabs,Intersection_slabs_below_out,'below',indiv_index,indiv_year)
        
        # Export the extracted SAR in sectors as csv files
        indiv_SAR_above_out.to_csv(path_save_SAR_IceSlabs+'above'+'/SAR_'+'above'+'_box_'+str(indiv_index)+'_year_'+str(indiv_year)+'.csv')
        indiv_SAR_inbetween_out.to_csv(path_save_SAR_IceSlabs+'in_between'+'/SAR_'+'in_between'+'_box_'+str(indiv_index)+'_year_'+str(indiv_year)+'.csv')
        indiv_SAR_within_out.to_csv(path_save_SAR_IceSlabs+'within'+'/SAR_'+'within'+'_box_'+str(indiv_index)+'_year_'+str(indiv_year)+'.csv')
        indiv_SAR_below_out.to_csv(path_save_SAR_IceSlabs+'below'+'/SAR_'+'below'+'_box_'+str(indiv_index)+'_year_'+str(indiv_year)+'.csv')
        ### Export extracted Ice Slabs and SAR sectors as csv files ###
        
        #DEAL WITH PLACES OUTSIDE OF REGIONS OF INTEREST ('Out' CATEGORY!!!!) There should not be Out data
        '''
        ########### Store ice slabs data to generate a full csv file ##########
        iceslabs_above_selected_overall=pd.concat([iceslabs_above_selected_overall,Intersection_slabs_above])
        iceslabs_inbetween_overall=pd.concat([iceslabs_inbetween_overall,Intersection_slabs_InBetween])
        iceslabs_within_overall=pd.concat([iceslabs_within_overall,Intersection_slabs_within])
        iceslabs_below_overall=pd.concat([iceslabs_below_overall,Intersection_slabs_below])
        ########### Store ice slabs data to generate a full csv file ##########
        '''

print('--- End of code ---')
