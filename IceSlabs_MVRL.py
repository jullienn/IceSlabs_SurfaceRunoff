# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:33:47 2023

@author: jullienn
"""


def display_summary(vector_todisplay,type_metric,bin_dist):
        
    #Prepare plot
    fig = plt.figure()
    fig.set_size_inches(19, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    gs = gridspec.GridSpec(6, 10)
    axsummary = plt.subplot(gs[0:6, 0:5])
    axsummary_distrib = plt.subplot(gs[0:3, 5:10])
    axsummary_boxplot = plt.subplot(gs[3:6, 5:10])
    gs.update(wspace=2)
    gs.update(hspace=1)
    axsummary.plot(vector_todisplay)
    axsummary.set_ylabel(type_metric+' difference 2012-2019 [m]')
    axsummary.axhline(y=np.nanmean(vector_todisplay),linestyle='dashed',color='red')
    axsummary.axhline(y=np.nanmedian(vector_todisplay),linestyle='dashed',color='green')
    axsummary_distrib.hist(vector_todisplay,density=True,bins=np.arange(np.nanmin(vector_todisplay),np.nanmax(vector_todisplay),bin_dist))#10 m bin width
    axsummary_distrib.set_xlabel(type_metric+' difference 2012-2019 [m]')
    axsummary_distrib.set_ylabel('Density [-]')
    axsummary_boxplot.boxplot(vector_todisplay[~np.isnan(vector_todisplay)],vert=False)
    axsummary_boxplot.set_xlabel(type_metric+' difference 2012-2019 [m]')
    
    #Display the same but with seaborn
    fig = plt.figure()
    fig.set_size_inches(10, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    gs = gridspec.GridSpec(6, 5)
    axsummary_distrib = plt.subplot(gs[0:3, 0:5])
    axsummary_boxplot = plt.subplot(gs[3:6, 0:5])
    gs.update(hspace=1)
    sns.histplot(data=vector_todisplay,ax=axsummary_distrib)
    axsummary_distrib.set_xlabel(type_metric+' difference 2012-2019 [m]')
    axsummary_distrib.set_ylabel('Count [-]')
    
    sns.boxplot(data=vector_todisplay,orient="h",ax=axsummary_boxplot)#10 m bin width
    axsummary_boxplot.set_xlabel(type_metric+' difference 2012-2019 [m]')
    axsummary_boxplot.text(x=-1000,y=-0.3,s='q0.25: '+str(np.round(np.nanquantile(vector_todisplay,0.25),1)),color='black',fontsize=20)
    axsummary_boxplot.text(x=-1000,y=-0.2,s='q0.50: '+str(np.round(np.nanquantile(vector_todisplay,0.50),1)),color='black',fontsize=20)
    axsummary_boxplot.text(x=-1000,y=-0.1,s='q0.75: '+str(np.round(np.nanquantile(vector_todisplay,0.75),1)),color='black',fontsize=20)
    
    #Elevation:
    #Median of differnece is 0 => identical MVRL locations.
    #Distribution slightly skewed towards negative values -> elev_2019 > elev_2012

    return

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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scalebar import scale_bar

#Define paths
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'
path_rignotetal2016_GrIS=path_switchdrive+'backup_Aglaja/working_environment/greenland_topo_data/'
path_jullienetal2023=path_switchdrive+'RT1/final_dataset_2002_2018/'

path_data=path_switchdrive+'RT3/data/'
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'

#Define palette for time , this is From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'SW': "#9e9ac8", 'CW': "#9e9ac8", 'NW': "#9e9ac8", 'NO': "#9e9ac8", 'NE': "#9e9ac8"}

### -------------------------- Load shapefiles --------------------------- ###
#Load Rignot et al., 2016 Greenland drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS+'GRE_Basins_IMBIE2_v1.3/GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp',rows=slice(51,57,1)) #the regions are the last rows of the shapefile
#Extract indiv regions and create related indiv shapefiles
NW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NW']
CW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='CW']
SW_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SW']
NO_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NO']
NE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NE']
SE_rignotetal=GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SE']

#Load Rignot et al., 2016 Greenland Ice Sheet mask
GrIS_mask=gpd.read_file(path_rignotetal2016_GrIS+'GRE_IceSheet_IMBIE2/GRE_IceSheet_IMBIE2/GRE_IceSheet_IMBIE2_v1_EPSG3413.shp',rows=slice(1,2,1)) 

#load 2012-2018 ice slabs high end from Jullien et al., (2023)
iceslabs_20102012_jullienetal2023=gpd.read_file(path_jullienetal2023+'/shapefiles/iceslabs_jullien_highend_2010_11_12.shp')
#load 2010-2018 ice slabs high end from Jullien et al., (2023)
iceslabs_20102018_jullienetal2023=gpd.read_file(path_jullienetal2023+'/shapefiles/iceslabs_jullien_highend_20102018.shp')

#Load MVRL in 2012, 2012, 2016, 2019
poly_2010=gpd.read_file(path_local+'data/runoff_limit_polys/poly_2010.shp')
poly_2012=gpd.read_file(path_local+'data/runoff_limit_polys/poly_2012.shp')
poly_2016=gpd.read_file(path_local+'data/runoff_limit_polys/poly_2016.shp')
poly_2019=gpd.read_file(path_local+'data/runoff_limit_polys/poly_2019.shp')
### -------------------------- Load shapefiles --------------------------- ###

### ---------------------------- Load xytpd ------------------------------ ###
'''
df_xytpd_all=pd.read_csv(path_switchdrive+'RT3/data/Emax/xytpd.csv',delimiter=',',decimal='.')
'''
df_xytpd_all=pd.read_csv(path_switchdrive+'RT3/data/Emax/xytpd_NDWI_cleaned_2019_v3.csv',delimiter=',',decimal='.')
### ---------------------------- Load xytpd ------------------------------ ###

### -------------------------- Load CumHydro ----------------------------- ###
#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
CumHydro = rxr.open_rasterio(path_local+'data/master_maps/'+'master_map_GrIS_mean.vrt',
                             masked=True).squeeze() #No need to reproject satelite image
#Extract x and y coordinates of satellite image
x_coord_CumHydro=np.asarray(CumHydro.x)
y_coord_CumHydro=np.asarray(CumHydro.y)
### -------------------------- Load CumHydro ----------------------------- ###

### ---------------- Load firn aquifers Miège et al., 2016 ---------------- ###
path_aquifers=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/firn_aquifers_miege/'
df_firn_aquifer_all=pd.DataFrame()
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2010.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2011.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2012.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2013.csv',delimiter=',',decimal='.'))
df_firn_aquifer_all=df_firn_aquifer_all.append(pd.read_csv(path_aquifers+'MiegeFirnAquiferDetections2014.csv',delimiter=',',decimal='.'))

#Transform miege coordinates from WGS84 to EPSG:3413
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
points=transformer.transform(np.asarray(df_firn_aquifer_all["LONG"]),np.asarray(df_firn_aquifer_all["LAT"]))
df_firn_aquifer_all['lon_3413']=points[0]
df_firn_aquifer_all['lat_3413']=points[1]
### ---------------- Load firn aquifers Miège et al., 2016 ---------------- ###

#Open Boxes from Tedstone and Machguth (2022)
Boxes_Tedstone2022=gpd.read_file(path_data+'Boxes_Tedstone2022/boxes.shp')

# Consider exept firn aquifer are prominent, i.e. all boxes except 1-3, 32-53.
nogo_polygon=np.concatenate((np.arange(1,3+1),np.arange(32,53+1)))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

#Prepare plot
#Set fontsize plot
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
fig.set_size_inches(12, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
gs = gridspec.GridSpec(10, 7)
gs.update(hspace=0.1)
gs.update(wspace=0)
ax1 = plt.subplot(gs[0:10, 0:5],projection=crs)
axsummary_elev = plt.subplot(gs[0:5, 5:7])
axsummary_signed_dist = plt.subplot(gs[5:10, 5:7])

#Display coastlines
ax1.coastlines(edgecolor='black',linewidth=0.075)
#Display 2010-2018 high end ice slabs jullien et al., 2023
iceslabs_20102018_jullienetal2023.plot(ax=ax1,facecolor='#ba2b2b',edgecolor='#ba2b2b')
#Display 2012-2018 high end ice slabs jullien et al., 2023
iceslabs_20102012_jullienetal2023.plot(ax=ax1,facecolor='#6baed6',edgecolor='#6baed6')

#Display firn aquifers Miège et al., 2016
ax1.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#74c476',s=1,zorder=2)

#Display MVRL
#poly_2010.plot(ax=ax1,facecolor='none',edgecolor='#dadaeb',linewidth=1,zorder=3)
poly_2012.plot(ax=ax1,facecolor='none',edgecolor='#dadaeb',linewidth=1,zorder=3)
#poly_2016.plot(ax=ax1,facecolor='none',edgecolor='#756bb1',linewidth=1,zorder=3)
poly_2019.plot(ax=ax1,facecolor='none',edgecolor='#54278f',linewidth=1,zorder=3)

#Display boxes not processed
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax1,color='#f6e8c3',edgecolor='none',zorder=4)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
#Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
GrIS_drainage_bassins.plot(ax=ax1,facecolor='none',edgecolor='black',zorder=5)

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax1.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=6)
#Customize lat labels
gl.right_labels = False
gl.bottom_labels = False
ax1.axis('off')
#ax8map.legend(loc='upper right')
###################### From Tedstone et al., 2022 #####################

ax1.set_xlim(-642397, 1005201)
ax1.set_ylim(-3366273, -784280)


#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Patch(facecolor='#6baed6',edgecolor='none',label='2010-2012 ice slabs'),
                   Patch(facecolor='#ba2b2b',edgecolor='none',label='2010-2018 ice slabs'),
                   Line2D([0], [0], color='#74c476', lw=2, marker='o',linestyle='None', label='2010-2014 firn aquifers'),
                   #Line2D([0], [0], color='#dadaeb', lw=2, label='2010 MVRL'),
                   Line2D([0], [0], color='#dadaeb', lw=2, label='2012 runoff limit'),
                   #Line2D([0], [0], color='#756bb1', lw=2, label='2016 MVRL'),
                   Line2D([0], [0], color='#54278f', lw=2, label='2019 runoff limit'),
                   Patch(facecolor='#f6e8c3',edgecolor='none',label='Ignored areas')]
ax1.legend(handles=legend_elements,loc='lower right',fontsize=12.5,framealpha=1).set_zorder(7)
plt.show()

#Display scalebar - from Fig2andS6andS7andS10.py
scale_bar(ax1, (0.7, 0.28), 200, 3,5)# axis, location (x,y), length, linewidth, rotation of text
#by measuring on the screen, the difference in precision between scalebar and length of transects is about ~200m

pdb.set_trace()
#plt.close()

#Transform xytpd dataframe into geopandas dataframe for distance calculation
df_xytpd_all_gpd = gpd.GeoDataFrame(df_xytpd_all, geometry=gpd.GeoSeries.from_xy(df_xytpd_all['x'], df_xytpd_all['y'], crs="EPSG:3413"))
#Intersection between df_xytpd_all_gpd and GrIS drainage bassins, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
df_xytpd_all_gpd = gpd.sjoin(df_xytpd_all_gpd, GrIS_drainage_bassins, predicate='within')
#Drop index_right
df_xytpd_all_gpd=df_xytpd_all_gpd.drop(columns=["index_right"])

#Calculate the difference in elevation between xytpd in 2012 VS 2019 in each slice_id
df_xytpd_2012=df_xytpd_all_gpd[df_xytpd_all_gpd.year==2012].copy()
df_xytpd_2019=df_xytpd_all_gpd[df_xytpd_all_gpd.year==2019].copy()

elevation_differences=[]
distance_differences=[]
signed_distances=[]
summary_df=pd.DataFrame()

for indiv_box in df_xytpd_2012.box_id.unique():
    
    if (indiv_box in nogo_polygon):
        print('Ignored polygon, continue')
        continue
    
    print(indiv_box)
    
    #Set an empty pandas dataframe to store results of distance and elevation difference
    indiv_elev_dist=pd.DataFrame()
    
    #Select 2012 and set index as slice_id to match dataframes
    df_xytpd_2012_indiv_box=df_xytpd_2012[df_xytpd_2012.box_id==indiv_box].copy()
    df_xytpd_2012_indiv_box=df_xytpd_2012_indiv_box.set_index('slice_id')
    
    #Select 2019 and set index as slice_id to match dataframes
    df_xytpd_2019_indiv_box=df_xytpd_2019[df_xytpd_2019.box_id==indiv_box].copy()
    df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box.set_index('slice_id')
    
    #spatial join
    joined_df=df_xytpd_2012_indiv_box.join(df_xytpd_2019_indiv_box,lsuffix='_2012',rsuffix='_2019')
    #If needed, it could be usefull to have this joined_df outside after the loop
    
    #Perform the difference in elevation
    indiv_elev_diff=(joined_df.elev_2012-joined_df.elev_2019)
    elevation_differences=np.append(elevation_differences,indiv_elev_diff.to_numpy())
    #If negative difference, this means elev_2012 < elev_2019
    
    #Calculate distance difference between each slice_id point
    df_xytpd_2012_indiv_box_for_dist=gpd.GeoSeries(df_xytpd_2012_indiv_box.geometry)
    df_xytpd_2019_indiv_box_for_dist=gpd.GeoSeries(df_xytpd_2019_indiv_box.geometry)
    indiv_dist_diff=df_xytpd_2012_indiv_box_for_dist.distance(df_xytpd_2019_indiv_box_for_dist,align=True)
    #Store the distance
    distance_differences=np.append(distance_differences,indiv_dist_diff.to_numpy())
        
    #Assign the sign to the distance calculation. We calculate elev 2012 - elev 2019. If elev 2012 > elev 2019, then distance is positive.
    indiv_elev_dist['elev_diff']=indiv_elev_diff
    indiv_elev_dist['dist_diff']=indiv_dist_diff
    indiv_elev_dist['signed_dist_diff']=indiv_dist_diff*np.sign(indiv_elev_diff)
    #If region is the same for both 2012 and 2019, store the region name
    joined_df.loc[joined_df.SUBREGION1_2012==joined_df.SUBREGION1_2019,"common_region"]=joined_df.loc[joined_df.SUBREGION1_2012==joined_df.SUBREGION1_2019,"SUBREGION1_2012"]
    indiv_elev_dist['SUBREGION1']=joined_df["common_region"]
    #Store this dataframe
    summary_df=pd.concat([summary_df,indiv_elev_dist])
    
    #Store the signed distance
    signed_distances=np.append(signed_distances,indiv_elev_dist['signed_dist_diff'].to_numpy())
    #If negative difference, this means elev_2012 < elev_2019
        
    #Load cumulative hydrology
    #Define bounds of Emaxs in this box
    x_min=df_xytpd_2012_indiv_box.x.min()-5e4
    x_max=df_xytpd_2012_indiv_box.x.max()+5e4
    y_min=df_xytpd_2012_indiv_box.y.min()-1e4
    y_max=df_xytpd_2012_indiv_box.y.max()+1e4
    #Extract coordinates of cumulative hydrology image within bounds
    logical_x_coord_within_bounds=np.logical_and(x_coord_CumHydro>=x_min,x_coord_CumHydro<=x_max)
    x_coord_within_bounds=x_coord_CumHydro[logical_x_coord_within_bounds]
    logical_y_coord_within_bounds=np.logical_and(y_coord_CumHydro>=y_min,y_coord_CumHydro<=y_max)
    y_coord_within_bounds=y_coord_CumHydro[logical_y_coord_within_bounds]
    #Define extents based on the bounds
    extent_CumHydro = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
    
    #Display
    #Prepare plot
    fig = plt.figure()
    fig.set_size_inches(19, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    gs = gridspec.GridSpec(6, 10)
    gs.update(wspace=2)
    gs.update(hspace=1)
    axcheck = plt.subplot(gs[0:6, 0:5],projection=crs)
    fig.suptitle('Box ID '+str(indiv_box))

    #Display cumulative hydrology in the background
    axcheck.imshow(CumHydro[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_CumHydro, transform=crs, origin='upper', cmap='Blues',zorder=0,vmin=50,vmax=250)
    #Display coastlines
    axcheck.coastlines(edgecolor='black',linewidth=0.075)
    #Display boxes not processed
    Boxes_Tedstone2022.plot(ax=axcheck,color='none',edgecolor='black')
    #Display current box
    Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box].plot(ax=axcheck,color='none',edgecolor='magenta')
    #Display Rignot and Mouginot regions edges
    GrIS_drainage_bassins.plot(ax=axcheck,facecolor='none',edgecolor='black')
    
    #Display MVRL retrievals
    axcheck.scatter(df_xytpd_2012_indiv_box.x,df_xytpd_2012_indiv_box.y,c=df_xytpd_2012_indiv_box.index,s=100)
    axcheck.scatter(df_xytpd_2019_indiv_box.x,df_xytpd_2019_indiv_box.y,c=df_xytpd_2019_indiv_box.index,s=50,edgecolor='black')
    
    axcheck.set_xlim(x_min,x_max)
    axcheck.set_ylim(y_min,y_max)
    
    #Display a red line linking 2012 and 2019 xytpd having the same slice_id
    for indiv_index in indiv_elev_dist.index:
        if (np.isnan(indiv_elev_dist.loc[indiv_index].elev_diff)):
            #Nan, continue
            continue
        axcheck.plot([df_xytpd_2012_indiv_box.loc[indiv_index].x,df_xytpd_2019_indiv_box.loc[indiv_index].x],
                     [df_xytpd_2012_indiv_box.loc[indiv_index].y,df_xytpd_2019_indiv_box.loc[indiv_index].y],color='red')
    
    #Custom legend myself - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    legend_elements = [Line2D([0], [0], color='yellow', marker='o',linestyle='None', label='2012 MVRL'),
                       Line2D([0], [0], color='yellow', marker='o',linestyle='None', markeredgecolor='black',label='2019 MVRL'),
                       Line2D([0], [0], color='red',label='2012-2019 linked slide_id')]
    axcheck.legend(handles=legend_elements,loc='lower left')
    
    #Display the elevation difference
    axcheck_elev = plt.subplot(gs[0:3, 5:10])
    (joined_df.elev_2012-joined_df.elev_2019).plot(ax=axcheck_elev)
    #Display median and mean as horizontal lines
    axcheck_elev.axhline(y=(joined_df.elev_2012-joined_df.elev_2019).mean(),linestyle='dashed',color='red')
    axcheck_elev.axhline(y=(joined_df.elev_2012-joined_df.elev_2019).median(),linestyle='dashed',color='green')
    #Custom legend myself - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    legend_elements = [Line2D([0], [0], color='red',linestyle='dashed', label='Mean'),
                       Line2D([0], [0], color='green',linestyle='dashed', label='Median')]
    axcheck_elev.legend(handles=legend_elements,loc='upper left')
    axcheck_elev.set_xlabel('Slice id')
    axcheck_elev.set_ylabel('Elevation 2012 - 2019 [m]')
    axcheck_elev.set_title('Elevation difference (2012 - 2019)')
    
    #Display the distance difference
    axcheck_dist = plt.subplot(gs[3:6, 5:10])
    axcheck_dist.plot(indiv_elev_dist['signed_dist_diff'])
    #Display median and mean as horizontal lines
    axcheck_dist.axhline(y=indiv_elev_dist['signed_dist_diff'].mean(),linestyle='dashed',color='red')
    axcheck_dist.axhline(y=indiv_elev_dist['signed_dist_diff'].quantile(0.5),linestyle='dashed',color='green')    
    axcheck_dist.set_xlabel('Slice id')
    axcheck_dist.set_ylabel('Signed distance 2012 VS 2019 [m]')
    axcheck_dist.set_title('Distance difference (2012 - 2019)')
    
    #Maximize figure size
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    #Save figure
    plt.savefig(path_local+'MVRL/Difference_elevation_2012_2019_box_'+str(indiv_box)+'_cleanedV3.png',dpi=300,bbox_inches='tight')
    #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    
    #Possibly filter out outliers? (e.g. 300 m difference is very likely one). Do not do it,maybe consider later on
    plt.close()

#Reset index of summary_df
summary_df.reset_index(inplace=True)

#Display boxes summary
display_summary(elevation_differences,'Elevation',10)
display_summary(distance_differences,'Distance',100)
display_summary(signed_distances,'Signed distance',100)

#Display quantiles in the difference regions of signed distances and elevation difference.
print('signed_dist_diff:')
print(summary_df.groupby(["SUBREGION1"]).quantile([0.25,0.5,0.75]).signed_dist_diff)
print(' ')
print('elev_diff:')
print(summary_df.groupby(["SUBREGION1"]).quantile([0.25,0.5,0.75]).elev_diff)
print(' ')
print('Count:')
print(summary_df.groupby(["SUBREGION1"]).count())

#Transform signed distances from m to km
summary_df.signed_dist_diff=summary_df.signed_dist_diff/1000

#Display the violin plot of signed distances and elevation difference
sns.violinplot(data=summary_df[~(summary_df.SUBREGION1.astype(str)=='nan')],y="signed_dist_diff",x="SUBREGION1",ax=axsummary_signed_dist,orient='v',palette=my_pal)
sns.violinplot(data=summary_df[~(summary_df.SUBREGION1.astype(str)=='nan')],y="elev_diff",x="SUBREGION1",ax=axsummary_elev,orient='v',palette=my_pal)

axsummary_elev.set_ylabel('Runoff limit elevation difference [m]')
axsummary_elev.set_xlabel('Region',labelpad=10)
axsummary_elev.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
axsummary_elev.xaxis.set_label_position('top')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
axsummary_elev.yaxis.set_label_position('right')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
axsummary_elev.set_ylim(-500,500)
axsummary_elev.grid(linestyle='dashed')
axsummary_signed_dist.set_ylabel('Runoff limit distance difference [km]',labelpad=12.5)
axsummary_signed_dist.set_xlabel(' ')
axsummary_signed_dist.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
axsummary_signed_dist.yaxis.set_label_position('right')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
axsummary_signed_dist.set_ylim(-50,50)
axsummary_signed_dist.grid(linestyle='dashed')

#Add region name on map - this is from Fig. 2 paper Ice Slabs Expansion and Thickening
ax1.text(NO_rignotetal.centroid.x-50000,NO_rignotetal.centroid.y-100000,np.asarray(NO_rignotetal.SUBREGION1)[0])
ax1.text(NE_rignotetal.centroid.x-150000,NE_rignotetal.centroid.y-100000,np.asarray(NE_rignotetal.SUBREGION1)[0])
ax1.text(SE_rignotetal.centroid.x-100000,SE_rignotetal.centroid.y+30000,np.asarray(SE_rignotetal.SUBREGION1)[0])
ax1.text(SW_rignotetal.centroid.x-35000,SW_rignotetal.centroid.y-100000,np.asarray(SW_rignotetal.SUBREGION1)[0])
ax1.text(CW_rignotetal.centroid.x-50000,CW_rignotetal.centroid.y-60000,np.asarray(CW_rignotetal.SUBREGION1)[0])
ax1.text(NW_rignotetal.centroid.x,NW_rignotetal.centroid.y-50000,np.asarray(NW_rignotetal.SUBREGION1)[0])

#Add label panel
ax1.text(0.01, 0.9,'a',ha='center', va='center', transform=ax1.transAxes,weight='bold',fontsize=20,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
axsummary_elev.text(-0.1, 0.95,'b',ha='center', va='center', transform=axsummary_elev.transAxes,weight='bold',fontsize=20,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
axsummary_signed_dist.text(-0.1, 0.95,'c',ha='center', va='center', transform=axsummary_signed_dist.transAxes,weight='bold',fontsize=20,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

pdb.set_trace()
'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig1/v2/Fig1_cleanedxytpdV3.png',dpi=1000,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''

print('--- End of code ---')

