# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:33:47 2023

@author: jullienn
"""

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
df_xytpd_all=pd.read_csv(path_switchdrive+'RT3/data/Emax/xytpd.csv',delimiter=',',decimal='.')
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
fig = plt.figure()
fig.set_size_inches(8, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
ax1 = plt.subplot(projection=crs)
#Display coastlines
ax1.coastlines(edgecolor='black',linewidth=0.075)

#Display 2010-2018 high end ice slabs jullien et al., 2023
iceslabs_20102018_jullienetal2023.plot(ax=ax1,facecolor='#ba2b2b',edgecolor='#ba2b2b')
#Display 2012-2018 high end ice slabs jullien et al., 2023
iceslabs_20102012_jullienetal2023.plot(ax=ax1,facecolor='#6baed6',edgecolor='#6baed6')

#Display MVRL
poly_2010.plot(ax=ax1,facecolor='none',edgecolor='#dadaeb',linewidth=1)
poly_2012.plot(ax=ax1,facecolor='none',edgecolor='#9e9ac8',linewidth=1)
poly_2016.plot(ax=ax1,facecolor='none',edgecolor='#756bb1',linewidth=1)
poly_2019.plot(ax=ax1,facecolor='none',edgecolor='#54278f',linewidth=1)

#Display boxes not processed
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax1,color='#f6e8c3',edgecolor='none')#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
#Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
GrIS_drainage_bassins.plot(ax=ax1,facecolor='none',edgecolor='black')

#Display firn aquifers Miège et al., 2016
ax1.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#74c476',s=1,zorder=2)

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax1.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
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
                   Line2D([0], [0], color='#dadaeb', lw=2, label='2010 MVRL'),
                   Line2D([0], [0], color='#9e9ac8', lw=2, label='2012 MVRL'),
                   Line2D([0], [0], color='#756bb1', lw=2, label='2016 MVRL'),
                   Line2D([0], [0], color='#54278f', lw=2, label='2019 MVRL'),
                   Line2D([0], [0], color='#74c476', lw=2, marker='o',linestyle='None', label='2010-2014 firn aquifers'),
                   Patch(facecolor='#f6e8c3',edgecolor='none',label='Ignored areas')]
ax1.legend(handles=legend_elements,loc='lower right')
plt.show()

#Display scalebar - from Fig2andS6andS7andS10.py
scale_bar(ax1, (0.7, 0.28), 200, 3,5)# axis, location (x,y), length, linewidth, rotation of text
#by measuring on the screen, the difference in precision between scalebar and length of transects is about ~200m

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/fig_IceSlabs_MVRL/fig_IceSlabs_MVRL.png',dpi=1000,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''
pdb.set_trace()
plt.close()

#Calculate the difference in elevation between xytpd in 2012 VS 2019 in each slice_id
df_xytpd_2012=df_xytpd_all[df_xytpd_all.year==2012].copy()
df_xytpd_2019=df_xytpd_all[df_xytpd_all.year==2019].copy()

elevation_differences=[]

for indiv_box in df_xytpd_2012.box_id.unique():
    
    if (indiv_box in nogo_polygon):
        print('Ignored polygon, continue')
        continue
    
    print(indiv_box)
    
    #Select 2012 and set index as slice_id to match dataframes
    df_xytpd_2012_indiv_box=df_xytpd_2012[df_xytpd_2012.box_id==indiv_box].copy()
    df_xytpd_2012_indiv_box=df_xytpd_2012_indiv_box.set_index('slice_id')
    
    #Select 2019 and set index as slice_id to match dataframes
    df_xytpd_2019_indiv_box=df_xytpd_2019[df_xytpd_2019.box_id==indiv_box].copy()
    df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box.set_index('slice_id')
    
    #spatial join
    joined_df=df_xytpd_2012_indiv_box.join(df_xytpd_2019_indiv_box,lsuffix='_2012',rsuffix='_2019')
    
    #Perform the difference in elevation
    elevation_differences=np.append(elevation_differences,(joined_df.elev_2012-joined_df.elev_2019).to_numpy())
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
    axcheck = plt.subplot(gs[0:6, 0:5],projection=crs)
    fig.suptitle('Box ID '+str(indiv_box))

    #Display cumulative hydrology in the background
    axcheck.imshow(CumHydro[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_CumHydro, transform=crs, origin='upper', cmap='Blues',zorder=0,vmin=50,vmax=250)
    #Display coastlines
    axcheck.coastlines(edgecolor='black',linewidth=0.075)
    #Display boxes not processed
    Boxes_Tedstone2022.plot(ax=axcheck,color='none',edgecolor='black')
    #Display current box
    Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box].plot(ax=axcheck,color='none',edgecolor='red')
    #Display Rignot and Mouginot regions edges
    GrIS_drainage_bassins.plot(ax=axcheck,facecolor='none',edgecolor='black')
    
    #Display MVRL retrievals
    axcheck.scatter(df_xytpd_2012_indiv_box.x,df_xytpd_2012_indiv_box.y,c=df_xytpd_2012_indiv_box.index,s=100)
    axcheck.scatter(df_xytpd_2019_indiv_box.x,df_xytpd_2019_indiv_box.y,c=df_xytpd_2019_indiv_box.index,s=50,edgecolor='black')
    
    axcheck.set_xlim(x_min,x_max)
    axcheck.set_ylim(y_min,y_max)
        
    #Custom legend myself - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    legend_elements = [Line2D([0], [0], color='yellow', marker='o',linestyle='None', label='2012 MVRL'),
                       Line2D([0], [0], color='yellow', marker='o',linestyle='None', markeredgecolor='black',label='2019 MVRL')]
    axcheck.legend(handles=legend_elements,loc='lower left')
    
    #Display the difference
    axcheckdiff = plt.subplot(gs[0:6, 5:10])
    (joined_df.elev_2012-joined_df.elev_2019).plot(ax=axcheckdiff)
    #Display median and mean as horizontal lines
    axcheckdiff.axhline(y=(joined_df.elev_2012-joined_df.elev_2019).mean(),linestyle='dashed',color='red')
    axcheckdiff.axhline(y=(joined_df.elev_2012-joined_df.elev_2019).median(),linestyle='dashed',color='green')

    #Custom legend myself - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    legend_elements = [Line2D([0], [0], color='red',linestyle='dashed', label='Mean'),
                       Line2D([0], [0], color='green',linestyle='dashed', label='Median')]
    axcheckdiff.legend(handles=legend_elements,loc='upper left')
    
    axcheckdiff.set_xlabel('Slice id')
    axcheckdiff.set_ylabel('Elev 2012 - 2019')
    axcheckdiff.set_title('Elevation difference (2012 - 2019)')
    
    #Maximize figure size
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    #Save figure
    plt.savefig(path_local+'MVRL/Difference_elevation_2012_2019_box_'+str(indiv_box)+'.png',dpi=300,bbox_inches='tight')
    #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    
    #Possibly filter out outliers?? 300 m difference is probably one
    plt.close()    

print('--- End of code ---')
