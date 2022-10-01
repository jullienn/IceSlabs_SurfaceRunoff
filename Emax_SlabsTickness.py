# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:00:42 2022

@author: JullienN
"""

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

### -------------------------- Load GrIS DEM ----------------------------- ###
#This is from paper Greenland Ice Sheet Ice Slab Expansion and Thickening, function 'extract_elevation.py'
#https://towardsdatascience.com/reading-and-visualizing-geotiff-images-with-python-8dcca7a74510
import rasterio
from rasterio.plot import show
path_GrIS_DEM = r'C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/elevations/greenland_dem_mosaic_100m_v3.0.tif'
GrIS_DEM = rasterio.open(path_GrIS_DEM)
### -------------------------- Load GrIS DEM ----------------------------- ###

#Define path flowlines
path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/'
#Open flowlignes
polygons_Machguth2022=gpd.read_file(path_data+'polygons_Machguth2022/Ys_polygons_v3.2b.shp')
#After manual identification on QGIS, we do not need 0-79, 82-84, 121-124, 146-154, 207-208, 212-217, 223-230, 232, 242-244, 258-267, 277-282, 289-303, 318-333
nogo_polygon=np.concatenate((np.arange(0,79+1),np.arange(82,84+1),np.arange(121,124+1),np.arange(146,154+1),np.arange(207,208+1),
                             np.arange(212,217+1),np.arange(223,230+1),np.arange(232,233),np.arange(242,244+1),np.arange(258,267+1),
                             np.arange(277,282+1),np.arange(289,303+1),np.arange(318,333+1)))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

### ---------------------------- Load dataset ---------------------------- ###
#Dictionnaries have already been created, load them
path_df_with_elevation='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/' 

#Load 2010-2018 high estimate
f_20102018_high = open(path_df_with_elevation+'final_excel/high_estimate/df_20102018_with_elevation_high_estimate_rignotetalregions', "rb")
df_2010_2018_high = pickle.load(f_20102018_high)
f_20102018_high.close()

#Load 2002-2003 dataset
path_2002_2003='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/'
df_2002_2003=pd.read_csv(path_2002_2003+'2002_2003_green_excel.csv')
### ---------------------------- Load dataset ---------------------------- ###

#Load max Ys from Machguth et al., (2022)
table_complete_annual_max_Ys=pd.read_csv(path_data+'_table_complete_annual_max_Ys.csv',delimiter=';',decimal=',')

#Load Emax from Tedstone and Machguth (2022)
Emax_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd.csv',delimiter=',',decimal='.')
#Rename columns preventing intersection
Emax_TedMach=Emax_TedMach.rename(columns={"index":"index_Emax"})
#Emax_TedMach.drop(columns=['Unnamed: 0'])

Emax_plus_Mad_TedMach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_plus_mad.csv',delimiter=',',decimal='.')
Emax_Ted_minus_Mad_Mach=pd.read_csv(path_data+'rlim_annual_maxm/xytpd_minus_mad.csv',delimiter=',',decimal='.')

'''
#Plot to check
fig = plt.figure(figsize=(10,5))
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)

#Display GrIS drainage bassins
polygons_Machguth2022.plot(ax=ax1,color='red', edgecolor='black',linewidth=0.5)
ax1.scatter(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413'],c=df_2010_2018_high['20m_ice_content_m'],s=0.1)
ax1.scatter(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y'],c=table_complete_annual_max_Ys['year'],s=10,cmap='magma')
ax1.scatter(Emax_TedMach['x'],Emax_TedMach['y'],c=Emax_TedMach['year'],s=5,cmap='magma')
plt.show()
'''

#Define df_2002_2003, df_2010_2018_high, Emax_TedMach, table_complete_annual_max_Ys as being a geopandas dataframes
points_2002_2003 = gpd.GeoDataFrame(df_2002_2003, geometry = gpd.points_from_xy(df_2002_2003['lon'],df_2002_2003['lat']),crs="EPSG:3413")
points_ice = gpd.GeoDataFrame(df_2010_2018_high, geometry = gpd.points_from_xy(df_2010_2018_high['lon_3413'],df_2010_2018_high['lat_3413']),crs="EPSG:3413")
points_Emax = gpd.GeoDataFrame(Emax_TedMach, geometry = gpd.points_from_xy(Emax_TedMach['x'],Emax_TedMach['y']),crs="EPSG:3413")
points_Ys = gpd.GeoDataFrame(table_complete_annual_max_Ys, geometry = gpd.points_from_xy(table_complete_annual_max_Ys['X'],table_complete_annual_max_Ys['Y']),crs="EPSG:3413")

#Define width of the buffer for ice slabs pick up (elevation-wise, in meters)
buffer=10

#Add a flag column in points_ice to flag data that are above, within and below the elevation band
points_ice['flag']=[np.nan]*len(points_ice)

#Define palette for time , this if From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'2010': "#1a9850", '2011': "#66bd63", '2012': "#a6d96a", '2013':"#d9ef8b", '2014':"#fee08b", '2016':"#fdae61", '2017':"#f46d43", '2018':"#d73027", '2019':"#d73027"}
pal_violin = {0: "#bdbdbd", 1:"#ffffcc", 2: "#fecc5c", 3: "#fd8d3c", 4:"#f03b20", 5:"#bd0026", 6:"#980043", 7:"#dd1c77", 8:"#df65b0", 9:"#d7b5d8",
              10: "#f1eef6", 11: "#bae4b3", 12: "#74c476", 13:"#31a354", 14:"#006d2c", 15:"#bdd7e7", 16:"#6baed6", 17:"#3182bd", 18:"#08519c",
              19: "#08306b"}
pal_violin_plot = ["#bdbdbd","#ffffcc","#fecc5c","#fd8d3c","#f03b20","#bd0026","#980043","#dd1c77","#df65b0","#d7b5d8",
                   "#f1eef6","#bae4b3","#74c476","#31a354","#006d2c","#bdd7e7","#6baed6","#3182bd","#08519c","#08306b"] #from https://towardsdatascience.com/how-to-use-your-own-color-palettes-with-seaborn-a45bf5175146
#sns.set_palette(sns.color_palette(pal_violin_plot))

#Define empty dataframe
iceslabs_above_selected_overall=pd.DataFrame()
iceslabs_selected_overall=pd.DataFrame()

#Define the radius [m]
radius=500

for indiv_index in polygons_Machguth2022.index:
    if (indiv_index in nogo_polygon):
        #Zone excluded form proicessing, continue
        print(indiv_index,' exluded, continue')
        continue
    '''
    if (indiv_index !=16):
        continue
    '''
    print(indiv_index)
    
    #Prepare plot
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(20, 6)
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    ax2 = plt.subplot(gs[0:20, 0:3],projection=crs)
    ax3 = plt.subplot(gs[0:20, 4:6])
    ax4 = plt.subplot(gs[0:20, 3:4],projection=crs)

    #Extract individual polygon
    indiv_polygon=polygons_Machguth2022[polygons_Machguth2022.index==indiv_index]

    #Display polygon
    '''
    indiv_polygon.plot(ax=ax1,color='orange', edgecolor='black',linewidth=0.5)
    '''
    indiv_polygon.plot(ax=ax2,color='#faf6c8', edgecolor='black',linewidth=0.5)
    
    indiv_polygon.plot(ax=ax4,color='#faf6c8', edgecolor='black',linewidth=0.5)
    ax4.set_xlim(-667470, 738665)
    ax4.set_ylim(-3365680, -666380)
    #Display coastlines
    ax4.coastlines(edgecolor='black',linewidth=0.75)
    
    #Intersection between 2002-2003 ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_20022003 = gpd.sjoin(points_2002_2003, indiv_polygon, op='within')
    #Intersection between ice slabs and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_ice = gpd.sjoin(points_ice, indiv_polygon, op='within')
    #Intersection between Emax and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Emax = gpd.sjoin(points_Emax, indiv_polygon, op='within')
    #Intersection between Ys and polygon of interest, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
    within_points_Ys = gpd.sjoin(points_Ys, indiv_polygon, op='within')
    
    '''
    #plot
    ax1.scatter(within_points_20022003['lon'],within_points_20022003['lat'],c='#bdbdbd',s=0.1)
    ax1.scatter(within_points_Emax['x'],within_points_Emax['y'],c=within_points_Emax['year'],s=5,cmap='magma')
    ax1.scatter(within_points_ice['lon_3413'],within_points_ice['lat_3413'],c=within_points_ice['20m_ice_content_m'],s=0.1)
    ax1.scatter(within_points_Ys['X'],within_points_Ys['Y'],c=within_points_Ys['year'],s=10,cmap='magma')
    '''
    
    #Display antecedent ice slabs
    ax2.scatter(within_points_20022003['lon'],within_points_20022003['lat'],color='#bdbdbd',s=10)
    
    plt.show()
    for indiv_year in list([2019]):#,2012,2016,2019]): #list([2010,2011,2012,2013,2014,2016,2017,2018]):#np.asarray(within_points_Ys.year):
        
        #Define empty dataframe
        subset_iceslabs_selected=pd.DataFrame()
        subset_iceslabs_above_selected=pd.DataFrame(columns=list(points_ice.keys()))

        #Select data of the desired year
        Emax_points=within_points_Emax[within_points_Emax.year==indiv_year]
        
        if (len(Emax_points)==0):
            continue
        
        #plot all the Emax points of the considered indiv_year
        ax2.scatter(Emax_points['x'],Emax_points['y'],color='black',s=10,zorder=2)
        
        #Define the yearly Ys point
        Ys_point=np.transpose(np.asarray([np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['X']),np.asarray(within_points_Ys[within_points_Ys.year==indiv_year]['Y'])]))   
        
        #Display the Ys of the current indiv_year
        if (len(Ys_point>0)):
            #There is an Ys of that year for this polygon, plot it
            ax2.scatter(Ys_point[0][0],Ys_point[0][1],color='magenta',s=10,zorder=10)
        
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
            #Select ice slabs data from 2010, 2011, 2012
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2010,(within_points_ice.year==2011)|(within_points_ice.year==2012))]
        elif(indiv_year == 2013):
            #Select ice slabs data from 2011, 2012, 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2011,(within_points_ice.year==2012)|(within_points_ice.year==2013))]
        elif(indiv_year == 2014):
            #Select ice slabs data from 2012, 2013, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,(within_points_ice.year==2013)|(within_points_ice.year==2014))]
        elif (indiv_year == 2015):
            #Select ice slabs data of the closest indiv_year, i.e. 2014 and the 2 previous ones
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,(within_points_ice.year==2013)|(within_points_ice.year==2014))]
        elif (indiv_year == 2016):
            #Select ice slabs data of the closest indiv_year, i.e. 2014 and the 2 previous ones
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2012,(within_points_ice.year==2013)|(within_points_ice.year==2014))]
        elif (indiv_year == 2017):
            #Select ice slabs data from 2017, 2014, 2013
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2013,(within_points_ice.year==2014)|(within_points_ice.year==2017))]
        elif (indiv_year == 2018):
            #Select ice slabs data from 2018, 2017, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,(within_points_ice.year==2017)|(within_points_ice.year==2018))]
        elif (indiv_year == 2019):
            #Select ice slabs data from 2018, 2017, 2014
            subset_iceslabs=within_points_ice[np.logical_or(within_points_ice.year==2014,(within_points_ice.year==2017)|(within_points_ice.year==2018))]
        else:
            #Select ice slabs data of the current indiv_year
            subset_iceslabs=within_points_ice[within_points_ice.year==indiv_year]
                
        if (len(subset_iceslabs)==0):
            #No slab for this particular year, continue
            continue
        
        #Display antecedent ice slabs
        ax2.scatter(within_points_ice[within_points_ice.year<=indiv_year]['lon_3413'],within_points_ice[within_points_ice.year<=indiv_year]['lat_3413'],color='gray',s=10)
        #Display the tracks of the current year within the polygon
        ax2.scatter(subset_iceslabs['lon_3413'],subset_iceslabs['lat_3413'],color='purple',s=10)
        
        #Look up tree and extraction of points were done thanks to the scipy documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html#scipy.spatial.KDTree.query_ball_point
        #Organise ice slabs coordinates for tree definition
        IceSlabs_points_xy=np.transpose(np.asarray([np.asarray(subset_iceslabs['lon_3413']),np.asarray(subset_iceslabs['lat_3413'])]))
        #Build up the tree
        tree= spatial.KDTree(IceSlabs_points_xy)
        #Extract indexes of the ice slabs points within a radius of 1000m around Emax points
        for indiv_Emax in range(0,len(Emax_points)):
            #Extract elevation of that Emax point
            Emax_point=np.asarray([np.asarray((Emax_points['x'].iloc[indiv_Emax],Emax_points['y'].iloc[indiv_Emax]))])
            for val in GrIS_DEM.sample(Emax_point): 
                #Calculate the corresponding elevation
                Emax_point_elevation=val[0]
            
            #Extract indexes of ice slabs points within 1000m radius around that Emax point
            indexes=tree.query_ball_point((Emax_points['x'].iloc[indiv_Emax],Emax_points['y'].iloc[indiv_Emax]),r=radius)
            
            if (len(subset_iceslabs.iloc[indexes])==0):
                continue
            
            #Extract corresponding datetrack
            list_datetracks=np.unique(subset_iceslabs.iloc[indexes]['Track_name'])
            #Display the corresponding transects
            for indiv_trackname in list(list_datetracks):
                print(indiv_trackname)
                #Display the whole track
                ax2.scatter(points_ice['lon_3413'][points_ice['Track_name']==indiv_trackname],points_ice['lat_3413'][points_ice['Track_name']==indiv_trackname],color='yellow',s=10,zorder=2)
                
                ''' 
                #If transect is not saved already, save data of that transect which are higher. OR xkm away ???
                if (indiv_trackname not in list(np.unique(subset_iceslabs_above_selected['Track_name']))):
                
                    #Save the ice slabs points of the transect of interest whose elevation is larger than the max elevation of the select ice slabs point within the radius of Emax point
                    indiv_transect=points_ice[points_ice['Track_name']==indiv_trackname]
                    subset_iceslabs_above_selected=pd.concat([subset_iceslabs_above_selected,indiv_transect[indiv_transect['elevation']>np.max(subset_iceslabs['elevation'].iloc[indexes])]])
                    #Plot resulting ice slabs points higher than picked ice slabs
                    ax2.scatter(indiv_transect[indiv_transect['elevation']>np.max(subset_iceslabs['elevation'].iloc[indexes])]['lon_3413'],indiv_transect[indiv_transect['elevation']>np.max(subset_iceslabs['elevation'].iloc[indexes])]['lat_3413'],color='blue',s=10,zorder=2)
                '''
                #WE LEAVE THE POSSIBILITY TO SELECT THE SAME DATA SEVERAL TIMES!
                ########## ---------- FOR ABOVE, ADD A CONDITION TO SELECT ONLY PERCODICULAR TO ELEVATION DATA ---------- ######## DONE
                
                ########## ---------- FOR ABOVE, ADD A CONDITION TO SELECT ONLY UPSTREAM DATA ?? ---------- ######## 
                ########## ----------- BECAUSE SOME CASES WHERE DOWNSTREAM DATA ARE SELECTED ... ---------- ######## 

                #Select all the data belonging to this track
                indiv_transect=points_ice[points_ice['Track_name']==indiv_trackname]
                #Select the ice slabs points of the transect of interest whose elevation is larger than the max elevation of the select ice slabs point within the radius of Emax point
                iceslabs_above=indiv_transect[indiv_transect['elevation']>np.max(subset_iceslabs['elevation'].iloc[indexes])]
                #Check whether the transect is more or less perpendicular to elevation contour. If yes, keep it, else continue
                
                #Calculate the variation of lat/lon
                vari=np.abs(np.mean(np.diff(iceslabs_above['lat_3413']))/np.mean(np.diff(iceslabs_above['lon_3413'])))
                print('variation is: ',str(np.round(vari,2)))
                if (vari <1): #If vari=1, angle is ~45°
                    #Not so strong variation, we keep it, otherwise we do not
                    #Save the data that are above and perpendicular to elevation contour
                    subset_iceslabs_above_selected=pd.concat([subset_iceslabs_above_selected,iceslabs_above])
                    #Plot resulting ice slabs points higher than picked ice slabs
                    ax2.scatter(iceslabs_above['lon_3413'],iceslabs_above['lat_3413'],color='blue',s=40,zorder=2)
            
            #Save the picked ice slabs points in the vicinity of Emax points
            subset_iceslabs_selected=pd.concat([subset_iceslabs_selected,subset_iceslabs.iloc[indexes]])#There might be points that are picked several times because of the used radius
            #Plot resulting extracted ice slabs points
            ax2.scatter(subset_iceslabs['lon_3413'].iloc[indexes],subset_iceslabs['lat_3413'].iloc[indexes],color='red',s=40,zorder=2)
            
            #Plot Emax points
            ax2.scatter(Emax_points['x'].iloc[indiv_Emax],Emax_points['y'].iloc[indiv_Emax],color='green',s=20,zorder=10)
            
            plt.show()
                
        if (len(subset_iceslabs_selected)==0):
            #No data, continue
            plt.close()
            continue
        
        #On the map, zoom on Emax points
        ax2.set_xlim(np.min(Emax_points['x'])-3e3,np.max(Emax_points['x'])+3e3)
        ax2.set_ylim(np.min(Emax_points['y'])-9e3,np.max(Emax_points['y'])+2e3)

        #Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
        legend_elements = [Line2D([0], [0], color='#bdbdbd', lw=2, label='2002-03 ice slabs'),
                           Line2D([0], [0], color='gray', lw=2, label='2010-18 ice slabs'),
                           Line2D([0], [0], color='purple', lw=2, label='Considered ice slabs (3 years)'),
                           Line2D([0], [0], color='yellow', lw=2, label='Transect matching with Emax'),
                           Line2D([0], [0], color='red', lw=2, label='Ice slabs within Emax radius'),
                           Line2D([0], [0], color='blue', lw=2, label='Ice slabs above Emax radius'),
                           Line2D([0], [0], color='black', lw=2, label='Emax retrieval', marker='o',linestyle='None'),
                           Line2D([0], [0], color='green', lw=2, label='Matched Emax retrieval', marker='o',linestyle='None'),
                           Line2D([0], [0], color='magenta', lw=2, label='Max Ys', marker='o',linestyle='None')]
        ax2.legend(handles=legend_elements,loc='lower left')
        plt.legend()
        
        #Plot ice slabs thickness that are above and within Ys elevation band
        ax3.hist(subset_iceslabs_above_selected['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax3.hist(subset_iceslabs_selected['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax3.set_xlabel('Ice content [m]')
        ax3.set_ylabel('Density [ ]')
        ax3.set_xlim(0,16)

        fig.suptitle('Polygon '+str(indiv_index)+ ' - '+str(indiv_year)+' - 3 years running slabs -'+' radius = '+str(int(radius))+'m')
        
        ax3.legend()
        plt.show()
        
        #Maximize plot size - This is from Fig1.py from Grenland ice slabs expansion and thickening paper.
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        pdb.set_trace()
        #Save the figure
        plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Emax_VS_IceSlabs_'+str(indiv_year)+'_polygon'+str(indiv_index)+'_3YearsRunSlabs_radius_'+str(int(radius))+'m.png',dpi=500)
        plt.close()
        
        #Save the iceslabs within and above of that polygon into another dataframe for overall plot
        iceslabs_above_selected_overall=pd.concat([iceslabs_above_selected_overall,subset_iceslabs_above_selected])
        iceslabs_selected_overall=pd.concat([iceslabs_selected_overall,subset_iceslabs_selected])#There might be points that are picked several times because of the used radius

#Prepare plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(20, 6)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax4 = plt.subplot(gs[0:20, 0:6])

#Region is already present =)
ax4.hist(iceslabs_above_selected_overall['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
ax4.hist(iceslabs_selected_overall['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
ax4.set_xlabel('Ice content [m]')
ax4.set_ylabel('Density [ ]')
fig.suptitle('Overall - '+str(indiv_year)+' - 3 years running slabs -'+' radius = '+str(int(radius))+'m')

ax4.legend()
plt.show()

#Maximize plot size - This is from Fig1.py from Grenland ice slabs expansion and thickening paper.
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

pdb.set_trace()
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Emax_VS_Iceslabs/whole_GrIS/Overall_Emax_VS_IceSlabs_'+str(indiv_year)+'_3YearsRunSlabs_radius_'+str(int(radius))+'m.png',dpi=500)

'''

if (indiv_year in list([2002,2003])):
    #Display the ice slabs points that are inside this buffer
    ax2.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
else:
    #Store an empty dataframe with the index so that index is displayed in plot even without data 
    if (len(subset_iceslabs_buffered)==0):
        #No slab for this particular year at these elevations
        subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,indiv_index, np.nan]]),columns=subset_iceslabs_buffered.columns.values)],ignore_index=True)# from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html and https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/
        #From https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean and https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work
        print(str(indiv_index)+' has no data')
        continue
    
    #Display the ice slabs points that are inside this buffer
    ax2.scatter(subset_iceslabs_buffered['lon_3413'],subset_iceslabs_buffered['lat_3413'],color='green',s=10)
    
    #Store subset_iceslabs_buffered 
    subset_iceslabs_buffered_summary=pd.concat([subset_iceslabs_buffered_summary,subset_iceslabs_buffered],ignore_index=True)
    #From https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis-mean and https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work

plt.show()
'''
