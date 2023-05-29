# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:46:44 2023

@author: JullienN
"""



def ExtractRadarData_at_PointLayer(raster_clipped,Extraction_IceSlabs_transect_InFunc,IceSlabsTransect_InFunc,buffered_transect_polygon_gpd_InFunc,IceSlabsTransect_name_InFunc,vmin_display_InFunc,vmax_display_InFunc,raster_type_InFunc):
    #Determine extent of raster_clipped
    extent_raster_clipped = [np.min(np.asarray(raster_clipped.x)), np.max(np.asarray(raster_clipped.x)),
                             np.min(np.asarray(raster_clipped.y)), np.max(np.asarray(raster_clipped.y))]#[west limit, east limit., south limit, north limit]
    
    #Make sure SAR was extracted at the correct place, and that it matches where ice slabs were extracted
    if (fig_display=='TRUE'):
        #Prepare plot
        fig = plt.figure()
        fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
        ax1 = plt.subplot(projection=crs)
        #Display coastlines
        ax1.coastlines(edgecolor='black',linewidth=0.075)
        #Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
        GrIS_drainage_bassins.plot(ax=ax1,facecolor='none',edgecolor='black')
        ### ---------- --This is from 'Greenland_Hydrology_Summary.py' ------------ ###
        #Display raster_clipped image
        cbar=ax1.imshow(raster_clipped, extent=extent_raster_clipped, transform=crs, origin='upper', cmap='gray',zorder=1,vmin=vmin_display_InFunc,vmax=vmax_display_InFunc)
        
        #Display Ice Thickness
        ax1.scatter(Extraction_IceSlabs_transect_InFunc['lon_3413'],
                    Extraction_IceSlabs_transect_InFunc['lat_3413'],
                    c=Extraction_IceSlabs_transect_InFunc['20m_ice_content_m'],zorder=3)
        
        #Display full Ice Slabs transect (from radargram)
        ax1.scatter(IceSlabsTransect_InFunc['longitude_EPSG_3413'],IceSlabsTransect_InFunc['latitude_EPSG_3413'],color='black')
        plt.close()

    if (check_oversampling_over=='TRUE'):
        ### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
        ### -------------- Proof that SAR data are not oversampled anymore -------------- ###
        #Define bounds of Emaxs in this box
        x_min=np.min(IceSlabsTransect_InFunc['longitude_EPSG_3413'])
        x_max=np.max(IceSlabsTransect_InFunc['longitude_EPSG_3413'])
        y_min=np.min(IceSlabsTransect_InFunc['latitude_EPSG_3413'])
        y_max=np.max(IceSlabsTransect_InFunc['latitude_EPSG_3413'])
        
        #Extract x and y coordinates of upsampled SAR image
        x_coord_focus=np.asarray(raster_clipped.x)
        y_coord_focus=np.asarray(raster_clipped.y)
        
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
        raster_clipped[logical_y_coord_within_bounds,logical_x_coord_within_bounds].plot(ax=ax_focus,edgecolor='black')
        
        #Set similar x and y limits
        ax_focus.set_xlim(x_min,x_max)
        ax_focus.set_ylim(y_min,y_max)
        ax_focus.set_title('Check data are not oversampled anymore')
        ### -------------- Proof that SAR data are not oversampled anymore -------------- ###
        ### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
        pdb.set_trace()
        plt.close()
        
    #4. Vectorise the SAR raster
    ######### This is from https://spatial-dev.guru/2022/04/16/polygonize-raster-using-rioxarray-and-geopandas/ #########
    x, y, raster_values = raster_clipped.x.values, raster_clipped.y.values, raster_clipped.values
    x, y = np.meshgrid(x, y)
    x, y, raster_values = x.flatten(), y.flatten(), raster_values.flatten()
    
    #Convert to geodataframe
    raster_pd = pd.DataFrame.from_dict({'raster_values': raster_values, 'x': x, 'y': y})
    #Keep only data where raster not NaN
    raster_pd_noNaN=raster_pd[~raster_pd.raster_values.isna()].copy()
    #The raster_vector is a geodataframe of points whose coordinates represent the centroid of each cell
    raster_vector = gpd.GeoDataFrame(raster_pd_noNaN, geometry=gpd.GeoSeries.from_xy(raster_pd_noNaN['x'], raster_pd_noNaN['y'], crs=raster_clipped.rio.crs))
    #Create a square buffer around each centroid to reconsititute the raster but where each cell is an individual polygon
    raster_grid = raster_vector.buffer(resolution_raster, cap_style=3)
    #Convert raster_grid into a geopandas dataframe, where we keep the information of the centroids (i.e. the raster values)
    raster_grid_gpd = gpd.GeoDataFrame(raster_pd_noNaN,geometry=gpd.GeoSeries(raster_grid),crs='epsg:3413')#from https://gis.stackexchange.com/questions/266098/how-to-convert-a-geoseries-to-a-geodataframe-with-geopandas
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
        ax_check_centroid.imshow(raster_clipped, extent=extent_raster_clipped, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=vmin_display_InFunc,vmax=vmax_display_InFunc)
        #Display the polygons corresponding to raster upsampled
        raster_grid_gpd.plot(ax=ax_check_centroid,alpha=0.2,facecolor='none',edgecolor='red')
        #Display the centroid of each polygon
        ax_check_centroid.scatter(raster_pd_noNaN.x,raster_pd_noNaN.y,color='blue')
        #Display buffered_transect_polygon_gpd_InFunc
        buffered_transect_polygon_gpd_InFunc.plot(ax=ax_check_centroid,facecolor='none',edgecolor='red',zorder=4)
        ax_check_centroid.set_title('Clipped raster and corresponding vector grid')
        #This does not look correct, but I am convinced this is due to a matplotlib diplay
        plt.close()
    
    
    #5. Perform the intersection between each cell of the polygonized raster data and Ice Slabs transect data
    ### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###
    #Convert Extraction_IceSlabs_transect_InFunc into a geopandas dataframe
    Extraction_IceSlabs_transect_gpd = gpd.GeoDataFrame(Extraction_IceSlabs_transect_InFunc, geometry=gpd.points_from_xy(Extraction_IceSlabs_transect_InFunc.lon_3413, Extraction_IceSlabs_transect_InFunc.lat_3413), crs="EPSG:3413")
            
    #Perform the join between ice slabs thickness and raster data
    pointInPolys= gpd.tools.sjoin(Extraction_IceSlabs_transect_gpd, raster_grid_gpd, predicate="within", how='left',lsuffix='left',rsuffix='right') #This is from https://www.matecdev.com/posts/point-in-polygon.html
    ### This is from Fig2andS7andS8andS12.py from paper 'Greenland Ice Slabs Expansion and Thickening' ###
    
    if (fig_display=='TRUE'):
        #Check extraction is correct
        #Prepare plot
        fig = plt.figure()
        fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
        ax_check_extraction_clipped = plt.subplot(projection=crs)
        #Display the raster SAR upsampled
        ax_check_extraction_clipped.imshow(raster_clipped, extent=extent_raster_clipped, transform=crs, origin='upper', cmap='Blues',zorder=1,vmin=vmin_display_InFunc,vmax=vmax_display_InFunc)
        #Display the extracted SAR signal
        ax_check_extraction_clipped.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['raster_values'],cmap='Blues',vmin=vmin_display_InFunc,vmax=vmax_display_InFunc,edgecolor='black')
        ax_check_extraction_clipped.set_title('Clipped raster data and corresponding extracted raster signal')
        #This does not look correct, but I am convinced this is due to a matplotlib diplay
        plt.close()
    
    #6. Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
    upsampled_raster_and_IceSlabs=pointInPolys.groupby('index_right').mean()
    
    if (fig_display=='TRUE'):
        
        #Display upsampling worked
        fig = plt.figure()
        fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
        ax_check_upsampling = plt.subplot(projection=crs)
        raster_grid_gpd.plot(ax=ax_check_upsampling,edgecolor='black')
        ax_check_upsampling.scatter(upsampled_raster_and_IceSlabs.lon_3413,upsampled_raster_and_IceSlabs.lat_3413,c=upsampled_raster_and_IceSlabs['20m_ice_content_m'],s=500,vmin=0,vmax=20)
        ax_check_upsampling.scatter(pointInPolys.lon_3413,pointInPolys.lat_3413,c=pointInPolys['20m_ice_content_m'],vmin=0,vmax=20)
        plt.close()
    
    #Sort by longitude for display
    upsampled_raster_and_IceSlabs_sorted=upsampled_raster_and_IceSlabs.sort_values(by=['index_right']).copy()
    upsampled_raster_and_IceSlabs_sorted['distances']=compute_distances(np.array(upsampled_raster_and_IceSlabs_sorted.lon_3413),
                                                                        np.array(upsampled_raster_and_IceSlabs_sorted.lat_3413))
    
        
    #7. Plot relationship raster VS Ice slabs thickness: display the oversampled and upsampled raster values with upsampled ice content
    #Prepare plot
    fig_indiv = plt.figure()
    fig_indiv.set_size_inches(14, 8) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    gs = gridspec.GridSpec(6, 10)
    gs.update(wspace=2)
    gs.update(hspace=1)
    ax_map_raster = plt.subplot(gs[0:4, 0:5],projection=crs)
    ax_upsampled_raster = plt.subplot(gs[0:4, 5:10])
    ax_radargram = plt.subplot(gs[4:6, 0:10])
            
    #Display
    ax_map_raster.coastlines()
    GrIS_drainage_bassins.plot(ax=ax_map_raster,facecolor='none',edgecolor='black')
    ax_map_raster.scatter(upsampled_raster_and_IceSlabs_sorted['lon_3413'],upsampled_raster_and_IceSlabs_sorted['lat_3413'],c=upsampled_raster_and_IceSlabs_sorted['distances']/1000,cmap='magma')
    cbar_dist=ax_upsampled_raster.scatter(upsampled_raster_and_IceSlabs_sorted['raster_values'],upsampled_raster_and_IceSlabs_sorted['20m_ice_content_m'],c=upsampled_raster_and_IceSlabs_sorted['distances']/1000,cmap='magma')
    ax_radargram.pcolor(IceSlabsTransect_InFunc['distances']/1000,
                        IceSlabsTransect_InFunc['depth'],
                        IceSlabsTransect_InFunc['IceSlabs_Mask'],
                        cmap='gray_r')
    #Improve axis
    ax_map_raster.set_xlim(upsampled_raster_and_IceSlabs_sorted['lon_3413'].mean()-2.5e5, upsampled_raster_and_IceSlabs_sorted['lon_3413'].mean()+2.5e5)
    ax_map_raster.set_ylim(upsampled_raster_and_IceSlabs_sorted['lat_3413'].mean()-2.5e5, upsampled_raster_and_IceSlabs_sorted['lat_3413'].mean()+2.5e5)
    ax_upsampled_raster.set_xlim(vmin_display_InFunc,vmax_display_InFunc)
    ax_upsampled_raster.set_ylim(-0.5,20.5)
    ax_radargram.invert_yaxis() #Invert the y axis = avoid using flipud.
    ax_radargram.set_aspect(0.1)
    ax_radargram.set_title('Ice slab transect')
    #Display titles
    ax_map_raster.set_title('Localisation')
    ax_upsampled_raster.set_title('Upsampled dataset')
    ax_radargram.set_title('Radargram')
    ax_radargram.set_xlabel('Distance [km]')
    ax_radargram.set_ylabel('Depth [m]')
    fig_indiv.colorbar(cbar_dist, ax=ax_upsampled_raster,label='Distance [km]') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot
    
    if (raster_type_InFunc=='SAR'):
        #Determine best fit curve
        #Prepare data for fit
        appended_df_no_NaN=upsampled_raster_and_IceSlabs_sorted.copy()
        appended_df_no_NaN=appended_df_no_NaN[~pd.isna(appended_df_no_NaN['raster_values'])]
        appended_df_no_NaN=appended_df_no_NaN[~pd.isna(appended_df_no_NaN['20m_ice_content_m'])]
        appended_df_no_NaN=appended_df_no_NaN.sort_values(by=['raster_values'])   
        
        if (len(appended_df_no_NaN)>0):
            xdata = np.array(appended_df_no_NaN['raster_values'])
            ydata = np.array(appended_df_no_NaN['20m_ice_content_m'])   
            #Perform and display fit on figure
            popt, pcov = curve_fit(func, xdata, ydata,p0=[1,0.26,-2],bounds=([0,-1,-4],[2,1,2]))
            ax_upsampled_raster.plot(xdata, func(xdata, *popt),'b',label='best fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))#, 'r-'
            ax_upsampled_raster.legend()
           
    #Save the figure
    fig_indiv.suptitle(IceSlabsTransect_name_InFunc)
    plt.savefig(path_local+raster_type_InFunc+'_and_IceThickness/images/'+IceSlabsTransect_name_InFunc+'_Upsampled.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    #Export the extracted values as csv
    pointInPolys.to_csv(path_local+raster_type_InFunc+'_and_IceThickness/csv/'+IceSlabsTransect_name_InFunc+'_NotUpsampled.csv',
                        columns=['Track_name', 'lat', 'lon','20m_ice_content_m',
                                 'likelihood', 'lat_3413', 'lon_3413', 'key_shp',
                                 'elevation', 'year', 'index_right', 'raster_values'])
    return


def SAR_clipping_check(SAR_source,SAR_clipped_by_polygon,polygon_mask):
    #Prepare figure to check clip performed well
    fig = plt.figure()
    gs = gridspec.GridSpec(10, 6)
    ax_check_clip_SAR = plt.subplot(gs[0:5, 0:6],projection=crs)
    ax_clipped_SAR = plt.subplot(gs[5:10, 0:6],projection=crs)
    extent_SAR_source = [np.min(np.asarray(SAR_source.x)), np.max(np.asarray(SAR_source.x)),
                          np.min(np.asarray(SAR_source.y)), np.max(np.asarray(SAR_source.y))]#[west limit, east limit., south limit, north limit]
    extent_SAR_clipped = [np.min(np.asarray(SAR_clipped_by_polygon.x)), np.max(np.asarray(SAR_clipped_by_polygon.x)),
                          np.min(np.asarray(SAR_clipped_by_polygon.y)), np.max(np.asarray(SAR_clipped_by_polygon.y))]#[west limit, east limit., south limit, north limit]
    ax_check_clip_SAR.imshow(SAR_source, extent=extent_SAR_source, transform=crs, origin='upper', cmap='gray',zorder=2,vmin=50,vmax=150)#vmin=-20,vmax=0
    ax_clipped_SAR.imshow(SAR_clipped_by_polygon, extent=extent_SAR_clipped, transform=crs, origin='upper', cmap='gray',zorder=2,vmin=50,vmax=150)#vmin=-20,vmax=0

    #Display polygon_mask
    SAR_clipped_by_polygon.plot(ax=ax_check_clip_SAR,facecolor='none',edgecolor='red',zorder=4)
    SAR_clipped_by_polygon.plot(ax=ax_clipped_SAR,facecolor='none',edgecolor='red',zorder=4)
    pdb.set_trace()
    plt.close()
    return

### This function is adapted from Extraction_IceThicknessAndSAR_InSectors.py ###
def extraction_SAR(polygon_to_be_intersected,SAR_SW_00_00_in_func,SAR_NW_00_00_in_func,SAR_N_00_00_in_func,SAR_N_00_23_in_func):#,polygon_in_use):
    #SAR_SW_00_23 and SAR_NW_00_23 are not needed as no slabs in these regions
    
    #Define clipping_SAR_ok to perform loop
    clipping_SAR_ok='FALSE'
    
    while (clipping_SAR_ok=='FALSE'):
        for regional_SAR in (SAR_SW_00_00_in_func,SAR_NW_00_00_in_func,SAR_N_00_00_in_func,SAR_N_00_23_in_func):
                        
            #3.c. Extract SAR values within the buffer - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html    
            #Suite of try except from https://stackoverflow.com/questions/17322208/multiple-try-codes-in-one-block       
            try:#from https://docs.python.org/3/tutorial/errors.html
                print('Try intersection with SAR ...')
                SAR_clipped = regional_SAR.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
                print('   Intersection found ...')
                '''
                #Check clipping performed well
                SAR_clipping_check(regional_SAR,SAR_clipped,polygon_to_be_intersected)
                '''
                #If sum of NaNs equals size of dataArray, this means a clip was performed but data held are all NaNs
                if (SAR_clipped.isnull().values.sum() == SAR_clipped.size):
                    print('      Intersection with empty data, try again ...')
                    SAR_clipped=np.array([-999])
                else:
                    #Clipping performed well with data in it, leave!
                    print('      Intersection holds data, done clipping')
                    clipping_SAR_ok='TRUE'
                    break
            except rxr.exceptions.NoDataInBounds:#From https://corteva.github.io/rioxarray/html/_modules/rioxarray/exceptions.html#NoDataInBounds
                print('   Intersection not found, try again ...')
                SAR_clipped=np.array([-999])

        #If no correct clip was found, exit
        clipping_SAR_ok='TRUE'
    
    return SAR_clipped
### This function is adapted from Extraction_IceThicknessAndSAR_InSectors.py ###


### This function is adapted from Extraction_IceThicknessAndSAR_InSectors.py ###
def extraction_CumHydro(polygon_to_be_intersected,master_SW_mean_in_func,master_NW_mean_in_func,master_NE_mean_in_func):    

    #3.c. Extract CumHydro values within the buffer - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html    
    #Suite of try except from https://stackoverflow.com/questions/17322208/multiple-try-codes-in-one-block       
    try:#from https://docs.python.org/3/tutorial/errors.html
        print('Try intersection with SW CumHydro ...')
        CumHydro_clipped = master_SW_mean_in_func.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
        print('   Intersection found with master_SW_mean')
        '''
        #Check clipping performed well
        SAR_clipping_check(master_SW_mean_in_func,CumHydro_clipped,polygon_to_be_intersected)
        '''
    except rxr.exceptions.NoDataInBounds:#From https://corteva.github.io/rioxarray/html/_modules/rioxarray/exceptions.html#NoDataInBounds
        print('   Intersection not found, try again ...')
        try:
            CumHydro_clipped = master_NW_mean_in_func.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
            print('      Intersection found with master_NW_mean')
            #Check clipping performed well
            '''
            SAR_clipping_check(master_NW_mean_in_func,CumHydro_clipped,polygon_to_be_intersected)
            '''
        except rxr.exceptions.NoDataInBounds:
            print('      Intersection not found, try again ...')
            try:
                CumHydro_clipped = master_NE_mean_in_func.rio.clip(polygon_to_be_intersected.geometry.values, polygon_to_be_intersected.crs, drop=True, invert=False)
                print('         Intersection found with master_NE_mean')
                #Check clipping performed well
                '''
                SAR_clipping_check(master_NE_mean_in_func,CumHydro_clipped,polygon_to_be_intersected)
                '''
            except rxr.exceptions.NoDataInBounds:
                print('            Intersection not found!')
                CumHydro_clipped=np.array([-999])

    return CumHydro_clipped
### This function is adapted from Extraction_IceThicknessAndSAR_InSectors.py ###


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
import os.path

generate_data='TRUE' #If true, generate the individual csv files and figures
fig_display='FALSE' #If TRUE, generate figures
check_oversampling_over='FALSE'

#Which raster extraction is to perform
SAR_process='TRUE'
CumHydro_process='FALSE'

#For all of these transects, there was no SAR extracted while there are, because a SAR mosaic is called beforehand, which extends there but is empty. This is now fixed!
list_check_clip=list(['20170331_01_095_098', '20170331_01_109_111','20170331_01_120_122',
                      '20170331_01_127_130', '20170331_01_134_136', '20170331_01_138_138',
                      '20170413_01_126_134', '20170414_01_004_005', '20170510_02_129_130'])

#Define projection
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Define paths where data are stored
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

path_jullienetal2023=path_switchdrive+'RT1/final_dataset_2002_2018/'
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
path_SAR=path_local+'data/SAR/HV_2017_2018/'

#Load IMBIE drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

#1. Load ice slabs data
### ------------------------- Load df_2010_2018 --------------------------- ###
#Load 2010-2018 clipped to polygons high estimate datatset
f_20102018_high_cleaned = open(path_jullienetal2023+'final_excel/high_estimate/clipped/df_20102018_with_elevation_high_estimate_rignotetalregions_cleaned', "rb")
df_20102018_high_cleaned = pickle.load(f_20102018_high_cleaned)
f_20102018_high_cleaned.close
### ------------------------- Load df_2010_2018 --------------------------- ###

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

#Open regional cumulative hydrology rasters
master_SW_mean = rxr.open_rasterio(path_local+'data/master_maps/merge_sw/master_SW_mean.vrt',masked=True).squeeze()
master_NW_mean = rxr.open_rasterio(path_local+'data/master_maps/merge_nw/master_NW_mean.vrt',masked=True).squeeze()
master_NE_mean = rxr.open_rasterio(path_local+'data/master_maps/merge_ne/master_NE_mean.vrt',masked=True).squeeze()

#Generate the csv files and figures of individual relationship    
if (generate_data=='TRUE'):
    #Loop over all the 2018 transects
    for IceSlabsTransect_name in list(df_20102018_high_cleaned[df_20102018_high_cleaned.year==2017].Track_name.unique()):
        print('Treating',IceSlabsTransect_name)
        '''
        if (IceSlabsTransect_name not in list_check_clip):
            continue
        '''
        #If transect already processes, continue
        if (os.path.isfile(path_local+'SAR_and_IceThickness/csv/'+IceSlabsTransect_name+'_NotUpsampled.csv')):#this is from https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
            print(IceSlabsTransect_name,' already generated, continue')    
            continue
        
        
        if (IceSlabsTransect_name in list(['20170410_01_086_088','20170422_01_138_139','20170506_01_117_118','20180423_01_056_056'])):
            print('No SAR intersections, do not process ',IceSlabsTransect_name)
            continue
        
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
            ### ------------ This is from 'Greenland_Hydrology_Summary.py' ------------ ###
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
            plt.close()
        ### ------------ This is from 'Greenland_Hydrology_Summary.py' ------------ ###
        
        #2. Extract ice slabs transect of interest
        Extraction_IceSlabs_transect=df_20102018_high_cleaned[df_20102018_high_cleaned.Track_name==IceSlabsTransect_name]
        
        #3. Extract SAR values in the vicinity of the transect to reduce computation
        ### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###
        #3.a. Transform transect of interest into a line
        Extraction_IceSlabs_transect_tuple=[tuple(row[['lon_3413','lat_3413']]) for index, row in Extraction_IceSlabs_transect.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        transect_line=LineString(Extraction_IceSlabs_transect_tuple)
        
        #3.b. Create a buffer around this line
        buffered_transect_polygon=transect_line.buffer(200,cap_style=3)
        #Convert polygon of buffered_transect_polygon into a geopandas dataframe
        buffered_transect_polygon_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[buffered_transect_polygon]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        ### --------------- This is from CaseStudy_Emax_IceSlabs.py --------------- ###
        
        if (SAR_process=='TRUE'):
            print('Performing SAR extraction')
            vmin_display=-20
            vmax_display=0
            resolution_raster=20
            
            #3.c. Extract SAR values within the buffer - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html                
            #Extract SAR in the vicinity of transect by considering all SAR files
            raster_clipped_SAR=extraction_SAR(buffered_transect_polygon_gpd,SAR_SW_00_00,SAR_NW_00_00,SAR_N_00_00,SAR_N_00_23)#,polygon_in_use)
                        
            #If clipping, perform extraction. If not, continue
            if (len(raster_clipped_SAR)>1):
                ExtractRadarData_at_PointLayer(raster_clipped_SAR,Extraction_IceSlabs_transect,IceSlabsTransect,buffered_transect_polygon_gpd,IceSlabsTransect_name,vmin_display,vmax_display,'SAR')       
        else:
            print('Do not perform SAR extraction')
        
        if (CumHydro_process=='TRUE'):
            print('Performing Cumulative Hydrology extraction')
            vmin_display=0
            vmax_display=250
            resolution_raster=15
            
            #3.c. Extract CumHydro values within the buffer - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html                
            #Extract CumHydro in the vicinity of transect by considering all CumHydro files
            raster_clipped_CumHydro=extraction_CumHydro(buffered_transect_polygon_gpd,master_SW_mean,master_NW_mean,master_NE_mean)
            
            #If clipping, perform extraction. If not, continue
            if (len(raster_clipped_CumHydro)>1):
                ExtractRadarData_at_PointLayer(raster_clipped_CumHydro,Extraction_IceSlabs_transect,IceSlabsTransect,buffered_transect_polygon_gpd,IceSlabsTransect_name,vmin_display,vmax_display,'CumHydro')
        else:
            print('Do not perform CumHydro extraction')
                
    print('Done in generating 2018 data')

print('-----------------------------------------------------')
print('End of function')
print('-----------------------------------------------------')
