# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:44:06 2022

@author: jullienn

This code include part of codes 'FigS7.py' from manuscript "Greenland Ice Slabs Thickening and Expansion"
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

import pickle
import scipy.io
import numpy as np
import pdb
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib.gridspec as gridspec
from pyproj import Transformer
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.image as mpimg
from pyproj import Transformer
import cartopy.crs as ccrs
import pickle
from shapely.geometry import LineString
from shapely.geometry import Polygon
import shapely

from descartes import PolygonPatch

'''
#Things to be done
1. Display 2002-03 data together with 2010-2018 (restricted to identical lon bounds)
    -> Open the radargrams saved (L1 product)
2. Display the correspoding Emax location of the specific year on the radargram
3. Display the Emax on a map aside for each year.
4. Display groups of Emax to observe any change
'''

method_keep='closest'#chan be 'closest' highest'

#Define variables
#Compute the speed (Modified Robin speed):
# self.C / (1.0 + (coefficient*density_kg_m3/1000.0))
v= 299792458

#Define paths to data
path_data_Jullien='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/i_out_from_IceBridgeGPR_Manager_v2.py/pickles_and_images/'
path_data='C:/Users/jullienn/Documents/working_environment/iceslabs_MacFerrin/data/'
path_20022003_data='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/radargram_data/'
path_data_switchdrive='C:/Users/jullienn/switchdrive/Private/research/RT3/data/'
path_rignotetal2016_GrIS_drainage_bassins='C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'

#Define palette for time periods, this is from fig2_paper_icelsabs.py
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'2002': 'yellow', '2003': 'yellow', '2010': "#fdd49e", '2011': "#fc8d59", '2012': "#fc8d59", '2013':"#d7301f",'2014':"#d7301f",'2017':"#7f0000",'2018':"#7f0000"}

#Load Rignot et al., 2016 Greenland drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3.shp')

#Define transformer for coordinates transform from "EPSG:4326" to "EPSG:3413"
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

#Define transformer_3413_to_4326 for coordinates transform from "EPSG:4326" to "EPSG:3413"
transformer_3413_to_4326 = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Case study 1, in box nb 7
CaseStudy1={2002:['jun04_02proc_4.mat'],
            2003:['may12_03_36_aggregated'],
            2010:['Data_20100513_01_001.mat','Data_20100513_01_002.mat'],
            2011:['Data_20110411_01_116.mat','Data_20110411_01_117.mat','Data_20110411_01_118.mat'],
            2012:['Data_20120428_01_125.mat','Data_20120428_01_126.mat'],
            2013:'empty',
            2014:['Data_20140408_11_024.mat','Data_20140408_11_025.mat','Data_20140408_11_026.mat'],
            2017:['Data_20170508_02_165.mat','Data_20170508_02_166.mat','Data_20170508_02_167.mat','Data_20170508_02_168.mat','Data_20170508_02_169.mat','Data_20170508_02_170.mat','Data_20170508_02_171.mat'],
            2018:'empty'}
'''
CaseStudy1
2011  is reversed
2012  is reversed
2014  is reversed
2017  is reversed
'''

#Case study 2, in box nb 9
CaseStudy2={2002:'empty',
            2003:['may12_03_1_aggregated','may12_03_2_aggregated'],
            2010:['Data_20100508_01_114.mat','Data_20100508_01_115.mat'],
            2011:['Data_20110419_01_008.mat','Data_20110419_01_009.mat','Data_20110419_01_010.mat'],
            2012:['Data_20120418_01_129.mat','Data_20120418_01_130.mat','Data_20120418_01_131.mat'],
            2013:['Data_20130405_01_165.mat','Data_20130405_01_166.mat','Data_20130405_01_167.mat'],
            2014:['Data_20140424_01_002.mat','Data_20140424_01_003.mat','Data_20140424_01_004.mat'],
            2017:['Data_20170422_01_168.mat','Data_20170422_01_169.mat','Data_20170422_01_170.mat','Data_20170422_01_171.mat'],
            2018:['Data_20180427_01_170.mat','Data_20180427_01_171.mat','Data_20180427_01_172.mat']}
'''
CaseStudy2
2010  is reversed
2012  is reversed
2013  is reversed
2017  is reversed
2018  is reversed
'''

#Define the panel to study
investigation_year=CaseStudy2

#Create figures
plt.rcParams.update({'font.size': 20})
fig2 = plt.figure()
ax_map = plt.subplot(projection=crs)

fig1 = plt.figure()
gs = gridspec.GridSpec(34, 101)
gs.update(wspace=0.1)
gs.update(wspace=0.5)

if (investigation_year==CaseStudy1):
    ax1 = plt.subplot(gs[0:4, 0:100])
    ax2 = plt.subplot(gs[4:8, 0:100])
    ax3 = plt.subplot(gs[8:12, 0:100])
    ax4 = plt.subplot(gs[12:16, 0:100])
    ax5 = plt.subplot(gs[16:20, 0:100])
    ax7 = plt.subplot(gs[20:24, 0:100])
    ax8 = plt.subplot(gs[24:28, 0:100])
    axc = plt.subplot(gs[0:28, 100:101])

elif (investigation_year==CaseStudy2):
    ax2 = plt.subplot(gs[0:4, 0:100])
    ax3 = plt.subplot(gs[4:8, 0:100])
    ax4 = plt.subplot(gs[8:12, 0:100])
    ax5 = plt.subplot(gs[12:16, 0:100])
    ax6 = plt.subplot(gs[16:20, 0:100])
    ax7 = plt.subplot(gs[20:24, 0:100])
    ax8 = plt.subplot(gs[24:28, 0:100])
    ax9 = plt.subplot(gs[28:32, 0:100])
    axc = plt.subplot(gs[0:32, 100:101])

else:
    print('Wrong transect name input')

#Define empty dataframe to store data
dataframe={}

for single_year in investigation_year.keys():
    #If no data, continue
    if (investigation_year[single_year]=='empty'):
        print('No data for year '+str(single_year)+', continue')
        continue
    
    print('Load data for year '+str(single_year))
    if (single_year in list([2002,2003])):

        #Prepare transect matrix
        #radargram_20022003=pd.Series(dtype='float64')
        radargram_20022003=np.zeros((99,0)) #From https://stackoverflow.com/questions/55561608/append-array-in-for-loop
        lat_20022003=np.zeros(0) #From https://stackoverflow.com/questions/55561608/append-array-in-for-loop
        lon_20022003=np.zeros(0) #From https://stackoverflow.com/questions/55561608/append-array-in-for-loop

        for indiv_file_load in investigation_year[single_year]:
            
            #Open file
            f_20022003_L1 = open(path_20022003_data+str(single_year)+'/L1_'+indiv_file_load.split('.mat')[0]+'.pickle', "rb")
            L1_2002003 = pickle.load(f_20022003_L1)
            f_20022003_L1.close()
                        
            #Store data for the whole transect
            radargram_20022003=np.concatenate((radargram_20022003,L1_2002003['radar_slice_0_30m']),axis=1)
            
            if (single_year==2002):
                lat_store=L1_2002003['lat_3413']
                lat_store.shape=(len(lat_store),)#from https://stackoverflow.com/questions/17869840/numpy-vector-n-1-dimension-n-dimension-conversion
                lon_store=L1_2002003['lon_3413']
                lon_store.shape=(len(lon_store),)#from https://stackoverflow.com/questions/17869840/numpy-vector-n-1-dimension-n-dimension-conversion
                
                lat_20022003=np.concatenate((lat_20022003,lat_store))
                lon_20022003=np.concatenate((lon_20022003,lon_store))
            else:
                lat_20022003=np.concatenate((lat_20022003,L1_2002003['lat_3413']))
                lon_20022003=np.concatenate((lon_20022003,L1_2002003['lon_3413']))
        
        #If from low to high elevations, fliplr
        if (np.sum(np.diff(lon_20022003)<0)):
            print(single_year,' is reversed')
            radargram_20022003=np.fliplr(radargram_20022003)
            lat_20022003=np.flipud(lat_20022003)
            lon_20022003=np.flipud(lon_20022003)

        #Calculate distances
        distances_with_start_transect=compute_distances(lon_20022003,lat_20022003)
        
        #Convert lat/lon into EPSG:4326
        #Transform the coordinates from EPSG:3413 to EPSG:4326
        #Example from: https://pyproj4.github.io/pyproj/stable/examples.html
        points=transformer_3413_to_4326.transform(np.array(lon_20022003),np.array(lat_20022003))
        lon_appended=points[0]
        lat_appended=points[1]
        
        #Store into a dictionnary:
        dataframe[str(single_year)]={'lat_appended':lat_appended,
                                     'lon_appended':lon_appended,
                                     'lat_3413':lat_20022003,
                                     'lon_3413':lon_20022003,
                                     'distances':distances_with_start_transect,
                                     'depth':L1_2002003['depths'],
                                     'radargram_30m':radargram_20022003}
    else:
        
        ###1. Load the depth corrected radargrams
        start_date_track=investigation_year[single_year][0]
        end_date_track=investigation_year[single_year][-1]
        date_track=start_date_track[5:20]+'_'+end_date_track[17:20]
        
        filename_depth_corrected=path_data_Jullien+'Depth_Corrected_Picklefiles/'+date_track+'_Depth_CORRECTED.pickle'
        #Open files
        f_depth_corrected = open(filename_depth_corrected, "rb")
        depth_corr = pickle.load(f_depth_corrected)
        f_depth_corrected.close()
    
        ###2. Load the latitude and longitude
        #Define empy vectors for storing
        lat_appended=[]
        lon_appended=[]
        
        for indiv_file_load in investigation_year[single_year]:
            
            #Create the path
            path_raw_data=path_data+str(single_year)+'_Greenland_P3/CSARP_qlook/'+indiv_file_load[5:16]+'/'
    
            #Load data
            if (single_year>=2014):
                
                fdata_filename = h5py.File(path_raw_data+indiv_file_load)
                lat_filename=fdata_filename['Latitude'][:,:]
                lon_filename=fdata_filename['Longitude'][:,:]
                time_filename=fdata_filename['Time'][:,:]
                
            else:
                fdata_filename = scipy.io.loadmat(path_raw_data+indiv_file_load)
                lat_filename = fdata_filename['Latitude']
                lon_filename = fdata_filename['Longitude']
                time_filename = fdata_filename['Time']
                
            #Append data
            lat_appended=np.append(lat_appended,lat_filename)
            lon_appended=np.append(lon_appended,lon_filename)
            
        #Check whether the data are acquired ascending or descending elevation wise.
        #I choose the ascending format. For the data that are descending, reverse them
        #To have ascending data, the longitude should increase
        #(-48 = low elevations, -46 = higher elevations)
        
        if (np.sum(np.diff(lon_appended))<0):
            #It is going toward lower elevations, thus flip left-right (or up-down) data
            print(single_year,' is reversed')
            lat_appended=np.flipud(lat_appended)
            lon_appended=np.flipud(lon_appended)
            depth_corr=np.fliplr(depth_corr)
        
        #Calculate the depth from the time
        #########################################################################
        # From plot_2002_2003.py - BEGIN
        #########################################################################
        depth_check = v * time_filename / 2.0
        
        #If 2014, transpose the vector
        if (str(single_year)>='2014'):
            depth_check=np.transpose(depth_check)
        
        #Reset times to zero! This is from IceBridgeGPR_Manager_v2.py
        if (depth_check[10]<0):
            #depth_check[10] so that I am sure that the whole vector is negative and
            #not the first as can be for some date were the proccessing is working
            depth_check=depth_check+abs(depth_check[0])
            depth = depth_check
        else:
            depth = depth_check
        
        if (str(single_year) in list(['2011','2012','2014','2017','2018'])):
            if (depth_check[10]>1):
                #depth_check[10] so that I am sure that the whole vector is largely positive and
                #not the first as can be for some date were the proccessing is working
                depth_check=depth_check-abs(depth_check[0])
                depth = depth_check
        
        #Transform the coordinates from WGS84 to EPSG:3413
        #Example from: https://pyproj4.github.io/pyproj/stable/examples.html
        points=transformer.transform(np.array(lon_appended),np.array(lat_appended))
        lon_3413=points[0]
        lat_3413=points[1]
        
        #Calculate distances
        distances_with_start_transect=compute_distances(lon_3413,lat_3413)
                
        #Store data into a dictionnary, but only from the surface down to 30m depth:
        dataframe[str(single_year)]={'lat_appended':lat_appended,
                                     'lon_appended':lon_appended,
                                     'lat_3413':lat_3413,
                                     'lon_3413':lon_3413,
                                     'distances':distances_with_start_transect,
                                     'depth':depth[depth<30],
                                     'radargram_30m':depth_corr[np.where(depth<30)[0],:]}

### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###
#Load Emax from Tedstone and Machguth (2022)
Emax_TedMach=pd.read_csv(path_data_switchdrive+'Emax/xytpd_NDWI_cleaned_2012_16_19_v2.csv',delimiter=',',decimal='.')
#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})
#Define Emax_TedMach as being a geopandas dataframes
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")
#Open Boxes from Tedstone and Machguth (2022)
Boxes_Tedstone2022=gpd.read_file(path_data_switchdrive+'Boxes_Tedstone2022/boxes.shp')
#Sort Boxes_Tedstone2022 as a function of FID
Boxes_Tedstone2022=Boxes_Tedstone2022.sort_values(by=['FID'],ascending=True)#from https://sparkbyexamples.com/pandas/pandas-sort-dataframe-by-multiple-columns/

#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
import rioxarray as rxr
path_cum_raster='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/master_maps/'
#Load cumulative raster data for display
cum_raster = rxr.open_rasterio(path_cum_raster+'master_map_GrIS_mean.vrt',
                              masked=True).squeeze() #No need to reproject satelite image
#Extract x and y coordinates of cumulative raster image
x_coord_cum_raster=np.asarray(cum_raster.x)
y_coord_cum_raster=np.asarray(cum_raster.y)
### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###

#For storing transect coordinates
lat_transet=[]
lon_transet=[]

for single_year in investigation_year.keys():

    if (investigation_year[single_year]=='empty'):
        continue

    print(single_year)
    
    if (investigation_year==CaseStudy2):
        start_transect=-47.70785561652585
        end_transect=-46.41555609606877
        vmin_plot=-4.5
        vmax_plot=4.5
        box_to_load=9
    elif (investigation_year==CaseStudy1):
        start_transect=-48.21060856534727
        end_transect=-46.88764316176339
        vmin_plot=-4.5
        vmax_plot=4.5
        box_to_load=7
    else:
        print('Wrong transect name input')
    
    if (single_year==2002):
        ax_plot=ax1
        color_toplot="grey"
    elif (single_year==2003):
        ax_plot=ax2
        color_toplot="grey"
    elif (single_year==2010):
        ax_plot=ax3
        color_toplot="#4dac26"
    elif (single_year==2011):
        ax_plot=ax4
        color_toplot="#0571b0"
    elif (single_year==2012):
        ax_plot=ax5
        color_toplot="#e31a1c"
    elif (single_year==2013):
        ax_plot=ax6
        color_toplot="#e31a1c"
    elif (single_year==2014):
        ax_plot=ax7
        color_toplot="#2171b5"
    elif (single_year==2017):
        ax_plot=ax8
        color_toplot="#2171b5"
    elif (single_year==2018):
        ax_plot=ax9
        color_toplot="#e31a1c"
        ax_plot.set_xlabel('Longitude [°]')
        #Activate ticks xlabel
        ax_plot.xaxis.tick_bottom()
    else:
        print('year not know')
    
    #Load data
    X=dataframe[str(single_year)]['lon_appended']
    Y=np.arange(0,30,30/dataframe[str(single_year)]['radargram_30m'].shape[0])
    C=dataframe[str(single_year)]['radargram_30m']
    
    #plot data
    if ((single_year==2002)|(single_year==2003)):
        cb=ax_plot.pcolor(X, Y, C,cmap=plt.get_cmap('gray'),zorder=-1,vmin=np.percentile(C.flatten(),2.5), vmax=np.percentile(C.flatten(),97.5))
    else:
        cb=ax_plot.pcolor(X, Y, C,cmap=plt.get_cmap('gray'),zorder=-1,vmin=vmin_plot, vmax=vmax_plot)
    
    ax_plot.invert_yaxis() #Invert the y axis = avoid using flipud.
    #Activate ticks ylabel
    ax_plot.yaxis.tick_left()
    #Set lims
    ax_plot.set_ylim(20,0)
    #Set yticklabels
    ax_plot.set_yticks([0,10,20])
    ax_plot.set_yticklabels(['0','10',''])
    #Set transect limits
    ax_plot.set_xlim(start_transect,end_transect)
    #Get rid of xticklabels
    ax_plot.set_xticklabels([])
    
    #Display year
    ax_plot.text(0.98, 0.875,str(single_year),ha='center', va='center', transform=ax_plot.transAxes,weight='bold',fontsize=15,color=my_pal[str(single_year)])#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    
    #Display radargram track on the map
    index_within_bounds=np.logical_and(dataframe[str(single_year)]['lon_appended']>=start_transect,dataframe[str(single_year)]['lon_appended']<=end_transect)
    ax_map.scatter(dataframe[str(single_year)]['lon_3413'][index_within_bounds],dataframe[str(single_year)]['lat_3413'][index_within_bounds],s=0.1,color=my_pal[str(single_year)],zorder=100)
    '''
    #Store the coordinates of displayed transect
    lat_transet=np.append(lat_transet,dataframe[str(single_year)]['lat_3413'][index_within_bounds])
    lon_transet=np.append(lon_transet,dataframe[str(single_year)]['lon_3413'][index_within_bounds])
    '''
#Display the map
if (investigation_year==CaseStudy1):
    year_limit=2017
elif (investigation_year==CaseStudy2):
    year_limit=2014
else:
    print('Year not known')
    pdb.set_trace()

#Find index of start and end of displayed transects
index_start_map=np.argmin(np.abs(np.abs(dataframe[str(year_limit)]['lon_appended'])-np.abs(start_transect)))
index_end_map=np.argmin(np.abs(np.abs(dataframe[str(year_limit)]['lon_appended'])-np.abs(end_transect)))

#Define bounds of cumulative raster map to display
x_min=dataframe[str(year_limit)]['lon_3413'][index_start_map]
x_max=dataframe[str(year_limit)]['lon_3413'][index_end_map]
y_min=dataframe[str(year_limit)]['lat_3413'][index_start_map]-5e3
y_max=dataframe[str(year_limit)]['lat_3413'][index_end_map]+5e3

#Extract coordinates ofcumulative raster within Emaxs bounds
logical_x_coord_within_bounds=np.logical_and(x_coord_cum_raster>=x_min,x_coord_cum_raster<=x_max)
x_coord_within_bounds=x_coord_cum_raster[logical_x_coord_within_bounds]
logical_y_coord_within_bounds=np.logical_and(y_coord_cum_raster>=y_min,y_coord_cum_raster<=y_max)
y_coord_within_bounds=y_coord_cum_raster[logical_y_coord_within_bounds]

#Define extents based on the bounds
extent_cum_raster = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
#Display cumulative raster image
cbar=ax_map.imshow(cum_raster[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_cum_raster, transform=crs, origin='upper', cmap='viridis',vmin=0,vmax=250,zorder=0) #NDWI
#Set xlims
ax_map.set_xlim(x_min,x_max)

#Create upper and lower line around transect centroid: 2017 and 2018 are roughly the centroid of the transect to extract Emax points
index_within_bounds_transect=np.logical_and(dataframe[str(2017)]['lon_appended']>=start_transect,dataframe[str(2017)]['lon_appended']<=end_transect)
upper_transect_lim = pd.DataFrame({'lon_3413_transect': dataframe[str(2017)]['lon_3413'][index_within_bounds_transect], 'lat_3413_transect': dataframe[str(2017)]['lat_3413'][index_within_bounds_transect]+1.1e3})#I choose 1.1e3 to include the closest 2012 Emax point
transect_centroid = pd.DataFrame({'lon_3413_transect': dataframe[str(2017)]['lon_3413'][index_within_bounds_transect], 'lat_3413_transect': dataframe[str(2017)]['lat_3413'][index_within_bounds_transect]})
lower_transect_lim = pd.DataFrame({'lon_3413_transect': dataframe[str(2017)]['lon_3413'][index_within_bounds_transect], 'lat_3413_transect': dataframe[str(2017)]['lat_3413'][index_within_bounds_transect]-1.1e3})#I choose 1.1e3 to include the closest 2012 Emax point

### ------------------ This is from Emax_SlabsTickness.py ----------------- ###
#Upper and lower max as tuples
upper_transect_tuple=[tuple(row[['lon_3413_transect','lat_3413_transect']]) for index, row in upper_transect_lim.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
transect_centroid_tuple=[tuple(row[['lon_3413_transect','lat_3413_transect']]) for index, row in transect_centroid.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
lower_transect_tuple=[tuple(row[['lon_3413_transect','lat_3413_transect']]) for index, row in lower_transect_lim.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples

#Make upper/lower_transect_tuple as a line
line_upper_transect= LineString(upper_transect_tuple) #from https://shapely.readthedocs.io/en/stable/manual.html
transect_centroid_transect= LineString(transect_centroid_tuple) #from https://shapely.readthedocs.io/en/stable/manual.html
line_lower_transect= LineString(lower_transect_tuple) #from https://shapely.readthedocs.io/en/stable/manual.html

#Create a polygon for Emax extraction
polygon_Emax_extraction=Polygon([*list(line_upper_transect.coords),*list(line_lower_transect.coords)[::-1]]) #from https://gis.stackexchange.com/questions/378727/creating-polygon-from-two-not-connected-linestrings-using-shapely

'''
############################# TO COMMENT LATER ON #############################
#Create polygon patch of the polygon above
plot_poylgon_Emax_extraction = PolygonPatch(polygon_Emax_extraction,zorder=2,color='blue',alpha=0.2)
#Display plot_poylgon_Emax_extraction
ax_map.add_patch(plot_poylgon_Emax_extraction)
#Display upper and lower limits
ax_map.plot(upper_transect_lim['lon_3413_transect'],upper_transect_lim['lat_3413_transect'],color='black',zorder=10)
ax_map.plot(lower_transect_lim['lon_3413_transect'],lower_transect_lim['lat_3413_transect'],color='black',zorder=10)
#Display all Emax points to check the selection worked
ax_map.scatter(points_Emax['x'],points_Emax['y'],color='green',s=5,zorder=8)
############################# TO COMMENT LATER ON #############################
'''

#Convert polygon of polygon_Emax_extraction into a geopandas dataframe
polygon_Emax_extraction_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_Emax_extraction]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
#Intersection between Emax points and polygon_Emax_extraction_gpd, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
Emax_extraction = gpd.sjoin(points_Emax, polygon_Emax_extraction_gpd, op='within')
#Plot the result of this selection
ax_map.scatter(Emax_extraction['x'],Emax_extraction['y'],color='red',s=5,zorder=0)
### ------------------ This is from Emax_SlabsTickness.py ----------------- ###

#count for zorder
count=0
#Display Emax
for single_year in range(2002,2020):
    print(single_year)
    
    cbar=ax_map.imshow(cum_raster[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_cum_raster, transform=crs, origin='upper', cmap='viridis',vmin=0,vmax=250,zorder=count) #NDWI
    #Set xlims
    ax_map.set_xlim(x_min,x_max)

    ### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###
    #Select data of the desired year
    Emax_points=Emax_extraction[Emax_extraction.year==single_year]
    #Display Emax points of that year within the transect bounds
    ax_map.scatter(Emax_points['x'],Emax_points['y'],s=10,zorder=count+1),#c=Emax_points['year'],cmap=plt.cm.plasma)
    pdb.set_trace()

    if (method_keep=='closest'):
        #Keep only the closest Emax point from the transect
        distances_Emax_points_transect=Emax_points.geometry.distance(transect_centroid_transect)#Calculate distance of each Emax points with respect to the transect centroid, this is from https://shapely.readthedocs.io/en/stable/manual.html
        best_Emax_point=Emax_points.iloc[np.where(distances_Emax_points_transect==np.min(distances_Emax_points_transect))]#Select this closest point
        ax_map.scatter(best_Emax_point['x'],best_Emax_point['y'],c='magenta',s=10,zorder=count+1)#Display this closest point
    elif (method_keep=='highest'):
        #Keep only the highest Emax point from the transect
        best_Emax_point=Emax_points.iloc[np.where(Emax_points.elev==np.max(Emax_points.elev))]#Select this highest point
        #if two points, select the closest
        if (len(best_Emax_point)>1):
            pdb.set_trace()
            #Keep only the closest Emax point
            distances_Emax_points_transect=best_Emax_point.geometry.distance(transect_centroid_transect)#Calculate distance of each Emax points with respect to the transect centroid, this is from https://shapely.readthedocs.io/en/stable/manual.html
            best_Emax_point=best_Emax_point.iloc[np.where(distances_Emax_points_transect==np.min(distances_Emax_points_transect))]#Select this closest point
        ax_map.scatter(best_Emax_point['x'],best_Emax_point['y'],c='magenta',s=10,zorder=count+1)#Display this closest point
    else:
        print('Method not known')
        pdb.set_trace()
    ### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###
    pdb.set_trace()
    count=count+2

    if (len(best_Emax_point)==0):
        continue
    else:
        #Convert to coordinates into EPSG:4326
        coord_Emax=transformer_3413_to_4326.transform(np.array(best_Emax_point['x']),np.array(best_Emax_point['y']))
        
        #Plot Emax on the radargrams
        if (single_year<2010):
            ax2.scatter(coord_Emax[0],2,c=my_pal['2002'])
        elif (single_year==2010):
            ax3.scatter(coord_Emax[0],2,c=my_pal['2010'])
        elif (single_year==2011):
            ax4.scatter(coord_Emax[0],2,c=my_pal['2011'])
        elif (single_year==2012):
            ax5.scatter(coord_Emax[0],2,c=my_pal['2012'])
        elif (single_year==2013):
            ax6.scatter(coord_Emax[0],2,c=my_pal['2013'])
        elif (single_year==2014):
            ax7.scatter(coord_Emax[0],2,c=my_pal['2014'])
        elif ((single_year>2014) & (single_year<=2017)):
            ax8.scatter(coord_Emax[0],2,c=my_pal['2017'])
        elif (single_year>=2018):
            ax9.scatter(coord_Emax[0],2,c=my_pal['2018'])
        else:
            print('year not know')
    
pdb.set_trace()

if (investigation_year==CaseStudy2):
    ax5.set_ylabel('Depth [m]')
    ax9.set_yticklabels(['0','10','20'])
    ticks_through=ax9.get_xticks()
    year_ticks=2018
    ax_tick_plot=ax9
    ax_top=ax2
    
elif (investigation_year==CaseStudy1):
    ax4.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax1
else:
    print('Wrong transect name input')

#Display colorbar. This is from FigS1.py
cbar_depth=fig1.colorbar(cb, cax=axc, aspect=5)#aspect is from https://stackoverflow.com/questions/33443334/how-to-decrease-colorbar-width-in-matplotlib
cbar_depth.set_label('Radar signal strength [dB]')

plot_dist=[]
for indiv_tick in ticks_through:
    lon_diff=[]
    lon_diff=np.abs(dataframe[str(year_ticks)]['lon_appended']-indiv_tick)
    index_min=np.argmin(lon_diff)
    if (lon_diff[index_min]>0.2):
        plot_dist=np.append(plot_dist,999)
    else:
        plot_dist=np.append(plot_dist,dataframe[str(year_ticks)]['distances'][index_min]/1000-dataframe[str(year_ticks)]['distances'][np.argmin(np.abs(dataframe[str(year_ticks)]['lon_appended']-start_transect))]/1000)


ax_tick_plot.xaxis.set_ticks_position('bottom') 
ax_tick_plot.set_xticklabels(np.round(plot_dist).astype(int))
ax_tick_plot.set_xlabel('Distance [km]')

ax_top.set_title('Case study 2')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()

pdb.set_trace()
'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Section1/Radargrams_Emax_highest_map.png',dpi=300,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''


    
    
