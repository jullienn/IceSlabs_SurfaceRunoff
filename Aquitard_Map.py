# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:58:01 2023

@author: jullienn
"""
########## SAR signal from Relationship_IceThicknessAndSAR.py
### ABOVE
## --- quantiles 0.25, 0.5, 0.75
#SW=-8.100298;-7.345824-6.67245
#CW=-6.198427;-5.79103;-5.45288
#NW=-7.914589;-6.623534;-5.72343
#NO=-6.453673;-5.319692;-4.507481
#NE=-5.779304;-5.08063;-4.402721

### BELOW
## --- quantiles 0.25, 0.5, 0.75
#SW=-10.201291;-9.623054;-9.077484
#CW=-10.353483;-8.994008;-7.742007
#NW=-11.245815;-10.080444;-8.926942
#NO=-10.186774;-8.533312;-7.128138
#NE-8.31114;-7.101851;-6.259047

### WITHIN
## --- quantiles 0.25, 0.5, 0.75
#SW=-9.739717;-9.198477;-8.653624
#CW=-9.199174;-8.057186;-7.060745
#NW=-10.485064;-9.141042;-8.22071
#NO=-8.973496;-7.322135;-6.249159
#NE=-7.455909;-6.663285;-5.716195



def apply_MinMax_nornalisation(raster_to_rescale,lower_bound,upper_bound):

    #Identify indexes
    index_below_lower_bound=(raster_to_rescale<lower_bound)
    index_above_upper_bound=(raster_to_rescale>upper_bound)
    index_within_bounds=np.logical_and(raster_to_rescale>=lower_bound,raster_to_rescale<=upper_bound)
    
    #Rescale outside of bounds
    raster_to_rescale[index_below_lower_bound]=1
    raster_to_rescale[index_above_upper_bound]=0
    
    #Rescale within bounds
    #x'=(x-min(x))/(max(x)-min(x)) - Min is lower bound, max is upper bound
    raster_to_rescale[index_within_bounds]=1-(raster_to_rescale[index_within_bounds]-lower_bound)/(upper_bound-lower_bound)
    
    return raster_to_rescale

def intersection_SAR_GrIS_bassin(SAR_to_intersect,individual_bassin,axis_display_NO,axis_display_SW,axis_display_NW,axis_display_CW,axis_SW_zoom_display,axis_cbar,vmin_bassin,vmax_bassin,name_save,save_aquitard):
    #Perform clip between SAR with region - this is inspired from https://corteva.github.io/rioxarray/stable/examples/clip_geom.html    
    SAR_intersected = SAR_to_intersect.rio.clip(individual_bassin.geometry.values, individual_bassin.crs, drop=True, invert=False)
    #Determine extent of SAR_SW_00_00_SW
    extent_SAR_intersected = [np.min(np.asarray(SAR_intersected.x)), np.max(np.asarray(SAR_intersected.x)),
                              np.min(np.asarray(SAR_intersected.y)), np.max(np.asarray(SAR_intersected.y))]#[west limit, east limit., south limit, north limit]
    
    #Perform normalisation
    SAR_intersected.data = apply_MinMax_nornalisation(SAR_intersected.data,vmin_bassin,vmax_bassin)
        
    #Display SAR image
    if (name_save[0:11] in list(['aquitard_NW','aquitard_NO','aquitard_NE'])):
        cbar_Runoff=axis_display_NO.imshow(SAR_intersected, extent=extent_SAR_intersected, transform=crs, origin='upper', cmap='Blues',zorder=1)
    
    if (name_save[0:11] in list(['aquitard_NW'])):
        axis_display_NW.imshow(SAR_intersected, extent=extent_SAR_intersected, transform=crs, origin='upper', cmap='Blues',zorder=1)

    if (name_save[0:11] in list(['aquitard_CW'])):
        axis_display_CW.imshow(SAR_intersected, extent=extent_SAR_intersected, transform=crs, origin='upper', cmap='Blues',zorder=1)
        axis_display_SW.imshow(SAR_intersected, extent=extent_SAR_intersected, transform=crs, origin='upper', cmap='Blues',zorder=1)

    if (name_save[0:11] in list(['aquitard_SW'])):
        axis_display_SW.imshow(SAR_intersected, extent=extent_SAR_intersected, transform=crs, origin='upper', cmap='Blues',zorder=1)
        axis_SW_zoom_display.imshow(SAR_intersected, extent=extent_SAR_intersected, transform=crs, origin='upper', cmap='Blues',zorder=1)
            
    #Display colorbar
    if (name_save == 'aquitard_NO_1'):        
        #Display cbar Runoff and Retention        
        cbar_Runoff_label=fig.colorbar(cbar_Runoff, cax=axis_cbar,orientation='horizontal',ticklocation='top')#Inspired from https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
        cbar_Runoff_label.set_label('Supporting runoff likelihood [-]')

    
    if (save_aquitard=='TRUE'):
        print('Saving raster',name_save)
        #Save the resulting aquitard map - this is from https://corteva.github.io/rioxarray/stable/examples/convert_to_raster.html
        SAR_intersected.rio.to_raster(
            path_SAR+"binary_runoff_retention/aquitard/"+name_save+".tif",
            tiled=True,  # GDAL: By default striped TIFF files are created. This option can be used to force creation of tiled TIFF files.
            windowed=True,  # rioxarray: read & write one window at a time
            )
    
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
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar

#If saving aquitard raster is desired
save_aquitard_true='FALSE'

#Define paths
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'
path_rignotetal2016_GrIS=path_switchdrive+'backup_Aglaja/working_environment/greenland_topo_data/'
path_jullienetal2023=path_switchdrive+'RT1/final_dataset_2002_2018/'

path_data=path_switchdrive+'RT3/data/'

path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'
#path_SAR=path_local+'data/SAR/HV_2017_2018/'
path_SAR='X:/RT3_jullien/SAR/HV_2017_2018/'

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

#load 2010-2018 ice slabs high end from Jullien et al., (2023)
iceslabs_20102018_jullienetal2023=gpd.read_file(path_jullienetal2023+'/shapefiles/iceslabs_jullien_highend_20102018.shp')
### -------------------------- Load shapefiles --------------------------- ###

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

#Open dry snow zone mask created on QGIS based on aquitard maps
DrySnowZoneMask=gpd.read_file(path_data+'aquitard/filter_dry_snow_zone.shp')

#Open Boxes from Tedstone and Machguth (2022)
Boxes_Tedstone2022=gpd.read_file(path_data+'Boxes_Tedstone2022/boxes.shp')

#Load 2013-2020 runoff limit
poly_2013_2020_median_edited=gpd.read_file(path_local+'data/runoff_limit_polys/poly_2013_2020_median_edited.shp')

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

#Load ice thickness and SAR at FS
FS_pd=pd.DataFrame(data={'Station': ['FS2', 'FS4', 'FS5'], '10m_ice_content_%': [95.06, 56.50, 38.44], 'SAR': [-11.37, -6.58, -5.42]})
'''
#Convert FS coordinates into EPSG:3413
points=transformer.transform(np.asarray(FS_pd["lon"]),np.asarray(FS_pd["lat"]))
FS_pd['lon_3413']=points[0]
FS_pd['lat_3413']=points[1]

#Display FS location, to identify the sector
ax1.scatter(FS_pd['lon_3413'],FS_pd['lat_3413'],c='red',s=1,zorder=10)
'''
#Open SAR image
### --- This is from Fisg4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
#Load SAR data
SAR_N_00_00_EW = rxr.open_rasterio(path_SAR+'ref_EW_HV_2017_2018_32_106_40m_ASCDESC_N_nscenes0_manual-0000000000-0000000000.tif',masked=True).squeeze()#No need to reproject satelite image
SAR_N_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_N_manual-0000000000-0000000000.tif',masked=True).squeeze()#No need to reproject satelite image
SAR_N_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_N_manual-0000000000-0000023296.tif',masked=True).squeeze()
SAR_NW_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_NW_manual-0000000000-0000000000.tif',masked=True).squeeze()
SAR_NW_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_NW_manual-0000000000-0000023296.tif',masked=True).squeeze()
SAR_SW_00_00 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000000000-0000000000.tif',masked=True).squeeze()
SAR_SW_00_23 = rxr.open_rasterio(path_SAR+'ref_IW_HV_2017_2018_32_106_40m_ASCDESC_SW_manual-0000023296-0000000000.tif',masked=True).squeeze()
### --- This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' --- ###

#Prepare plot
plt.rcParams.update({'font.size': 8})
fig = plt.figure()
fig.set_size_inches(8.27,8.62)#(13.1, 13.65) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
#Nature pdf size = (8.27,9)

'''
gs = gridspec.GridSpec(14, 12)
gs.update(hspace=0.1)
gs.update(wspace=0.1)
ax_NO = plt.subplot(gs[0:5, 0:12],projection=crs)
ax_NW = plt.subplot(gs[5:14, 0:3],projection=crs)
ax_CW = plt.subplot(gs[5:14, 3:6],projection=crs)
ax_SW = plt.subplot(gs[5:14, 6:9],projection=crs)
ax_SW_zoom = plt.subplot(gs[5:14, 9:12],projection=crs)
'''

gs = gridspec.GridSpec(76, 120)
gs.update(hspace=0.1)
gs.update(wspace=0.1)
ax_cbar = plt.subplot(gs[0:1, 40:80])
ax_NO = plt.subplot(gs[6:31, 3:117],projection=crs)
ax_NW = plt.subplot(gs[31:76, 0:30],projection=crs)
ax_CW = plt.subplot(gs[31:76, 30:60],projection=crs)
ax_SW = plt.subplot(gs[31:76, 60:90],projection=crs)
ax_SW_zoom = plt.subplot(gs[31:76, 90:120],projection=crs)

#Display coastlines
ax_NO.coastlines(edgecolor='black',linewidth=0.075)
ax_NW.coastlines(edgecolor='black',linewidth=0.075)
ax_CW.coastlines(edgecolor='black',linewidth=0.075)
ax_SW.coastlines(edgecolor='black',linewidth=0.075)
ax_SW_zoom.coastlines(edgecolor='black',linewidth=0.075)

#Display ice sheet background
GrIS_drainage_bassins.plot(ax=ax_NO,facecolor='#f7fbff',edgecolor='none')
GrIS_drainage_bassins.plot(ax=ax_NW,facecolor='#f7fbff',edgecolor='none')
GrIS_drainage_bassins.plot(ax=ax_CW,facecolor='#f7fbff',edgecolor='none')
GrIS_drainage_bassins.plot(ax=ax_SW,facecolor='#f7fbff',edgecolor='none')
GrIS_drainage_bassins.plot(ax=ax_SW_zoom,facecolor='#f7fbff',edgecolor='none')

'''
--- SW ---
- Above
0.25   -7.973435
0.50   -7.243513
0.75   -6.613111
- Within
0.25    -9.74725
0.50   -9.196233
0.75   -8.638897
- Below
0.25   -10.226363
0.50    -9.654804
0.75    -9.110144

--- CW ---
- Above
0.25   -6.131196
0.50   -5.750285
0.75   -5.419163
- Within
0.25   -9.191771
0.50   -8.053114
0.75   -7.038067
- Below
0.25   -10.469379
0.50    -9.074578
0.75     -7.82364

--- NW ---
- Above
0.25   -7.991582
0.50   -6.810143
0.75   -5.905346
- Within
0.25   -10.610457
0.50    -9.192882
0.75    -8.266375
- Below
0.25   -11.335603
0.50   -10.204542
0.75    -8.982688

--- NO ---
- Above
0.25   -5.820288
0.50   -5.166023
0.75   -4.389923
- Within
0.25   -8.500205
0.50   -7.081633
0.75   -6.104299
- Below
0.25     -9.9055
0.50   -8.558269
0.75   -7.194321

--- NE ---
- Above
0.25   -5.696267
0.50   -4.993743
0.75   -4.297798
- Within
0.25   -7.451725
0.50   -6.661402
0.75   -5.715943
- Below
0.25   -8.455717
0.50   -7.189671
0.75   -6.329797
'''
pdb.set_trace()

#quantile 0.75 of below to quantile 0.75 of within
intersection_SAR_GrIS_bassin(SAR_SW_00_23,SW_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-9.110144,-8.638897,'aquitard_SW_1',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_SW_00_00,SW_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-9.110144,-8.638897,'aquitard_SW_2',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_SW_00_00,CW_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-7.82364,-7.038067,'aquitard_CW_1',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_NW_00_00,CW_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-7.82364,-7.038067,'aquitard_CW_2',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_NW_00_00,NW_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-8.982688,-8.266375,'aquitard_NW_1',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_N_00_00_EW,NW_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-8.982688,-8.266375,'aquitard_NW_2',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_N_00_00_EW,NO_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-7.194321,-6.104299,'aquitard_NO_1',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_N_00_23,NO_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-7.194321,-6.104299,'aquitard_NO_2',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_N_00_00_EW,NE_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-6.329797,-5.715943,'aquitard_NE_1',save_aquitard_true)
intersection_SAR_GrIS_bassin(SAR_N_00_23,NE_rignotetal,ax_NO,ax_SW,ax_NW,ax_CW,ax_SW_zoom,ax_cbar,-6.329797,-5.715943,'aquitard_NE_2',save_aquitard_true)

#Display dry snow zone mask
DrySnowZoneMask.plot(ax=ax_NO,facecolor='#f7fbff',edgecolor='none')
DrySnowZoneMask.plot(ax=ax_NW,facecolor='#f7fbff',edgecolor='none')
DrySnowZoneMask.plot(ax=ax_CW,facecolor='#f7fbff',edgecolor='none')
DrySnowZoneMask.plot(ax=ax_SW,facecolor='#f7fbff',edgecolor='none')
ax_SW_zoom.plot(ax=ax_SW,facecolor='#f7fbff',edgecolor='none')

#Display 2013-2020 edited runoff limit
poly_2013_2020_median_edited.plot(ax=ax_NO,facecolor='none',edgecolor='#fed976',linewidth=1)
poly_2013_2020_median_edited.plot(ax=ax_NW,facecolor='none',edgecolor='#fed976',linewidth=1)
poly_2013_2020_median_edited.plot(ax=ax_CW,facecolor='none',edgecolor='#fed976',linewidth=1)
poly_2013_2020_median_edited.plot(ax=ax_SW,facecolor='none',edgecolor='#fed976',linewidth=1)
poly_2013_2020_median_edited.plot(ax=ax_SW_zoom,facecolor='none',edgecolor='#fed976',linewidth=1)

#Display boxes not processed
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_NO,color='#d9d9d9',edgecolor='#d9d9d9')#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_NW,color='#d9d9d9',edgecolor='#d9d9d9')#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_CW,color='#d9d9d9',edgecolor='#d9d9d9')#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_SW,color='#d9d9d9',edgecolor='#d9d9d9')#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_SW_zoom,color='#d9d9d9',edgecolor='#d9d9d9')#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas

#Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
GrIS_drainage_bassins.plot(ax=ax_NO,facecolor='none',edgecolor='black')
GrIS_drainage_bassins.plot(ax=ax_NW,facecolor='none',edgecolor='black')
GrIS_drainage_bassins.plot(ax=ax_CW,facecolor='none',edgecolor='black')
GrIS_drainage_bassins.plot(ax=ax_SW,facecolor='none',edgecolor='black')
GrIS_drainage_bassins.plot(ax=ax_SW_zoom,facecolor='none',edgecolor='black')

#Display 2010-2018 high end ice slabs jullien et al., 2023
iceslabs_20102018_jullienetal2023.plot(ax=ax_NO,facecolor='none',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_NW,facecolor='none',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_CW,facecolor='none',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_SW,facecolor='none',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_SW_zoom,facecolor='none',edgecolor='#ba2b2b')

'''
#Display firn aquifers Miège et al., 2016
ax_NO.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=1,zorder=2)
ax_NW.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=5,zorder=2)
ax_CW.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=5,zorder=2)
ax_SW.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=5,zorder=2)
ax_SW_zoom.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=5,zorder=2)
'''

#Set limits
#NO
ax_NO.set_xlim(-605557.1790513115, 464264.83348328667)
ax_NO.set_ylim(-1214409.8703545197, -874562.5225317206)
#NW
ax_NW.set_xlim(-327640.39779065247, -218961.34771576658)
ax_NW.set_ylim(-1874638.6166935905, -1494261.94143149) 
#CW
ax_CW.set_xlim(-183736.7996412225, -62636.10288859956)
ax_CW.set_ylim(-2375526.002225974, -2022535.0326401703)
#SW
ax_SW.set_xlim(-171994.14895702014, -65379.43459850631)
ax_SW.set_ylim(-2755750.7046506493, -2374228.6276486944)
#SW zoom
ax_SW_zoom.set_xlim(-127109.2920592131, -99678.47675615078)
ax_SW_zoom.set_ylim(-2649282.3420033744, -2565299.618246693)

#Display panel label
ax_NO.text(0.02, 0.95,'a',ha='center', va='center', transform=ax_NO.transAxes,weight='bold',fontsize=12,color='black',backgroundcolor='white',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_NW.text(0.05, 0.975,'b',ha='center', va='center', transform=ax_NW.transAxes,weight='bold',fontsize=12,color='black',backgroundcolor='white',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_CW.text(0.05, 0.975,'c',ha='center', va='center', transform=ax_CW.transAxes,weight='bold',fontsize=12,color='black',backgroundcolor='white',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_SW.text(0.05, 0.975,'d',ha='center', va='center', transform=ax_SW.transAxes,weight='bold',fontsize=12,color='black',backgroundcolor='white',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_SW_zoom.text(0.05, 0.975,'e',ha='center', va='center', transform=ax_SW_zoom.transAxes,weight='bold',fontsize=12,color='black',backgroundcolor='white',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#Re add frame on top of plot
ax_NO.add_patch(patches.Rectangle(([ax_NO.get_xlim()[0],ax_NO.get_ylim()[0]][0],[ax_NO.get_xlim()[0],ax_NO.get_ylim()[0]][1]),
                                    np.abs([ax_NO.get_xlim()[0],ax_NO.get_ylim()[0]][0]-[ax_NO.get_xlim()[1],ax_NO.get_ylim()[1]][0]),
                                    np.abs([ax_NO.get_xlim()[0],ax_NO.get_ylim()[0]][1]-[ax_NO.get_xlim()[1],ax_NO.get_ylim()[1]][1]),
                                    angle=0, linewidth=1, edgecolor='black', facecolor='none',zorder=11))

ax_NW.add_patch(patches.Rectangle(([ax_NW.get_xlim()[0],ax_NW.get_ylim()[0]][0],[ax_NW.get_xlim()[0],ax_NW.get_ylim()[0]][1]),
                                    np.abs([ax_NW.get_xlim()[0],ax_NW.get_ylim()[0]][0]-[ax_NW.get_xlim()[1],ax_NW.get_ylim()[1]][0]),
                                    np.abs([ax_NW.get_xlim()[0],ax_NW.get_ylim()[0]][1]-[ax_NW.get_xlim()[1],ax_NW.get_ylim()[1]][1]),
                                    angle=0, linewidth=1, edgecolor='black', facecolor='none',zorder=11))

ax_CW.add_patch(patches.Rectangle(([ax_CW.get_xlim()[0],ax_CW.get_ylim()[0]][0],[ax_CW.get_xlim()[0],ax_CW.get_ylim()[0]][1]),
                                    np.abs([ax_CW.get_xlim()[0],ax_CW.get_ylim()[0]][0]-[ax_CW.get_xlim()[1],ax_CW.get_ylim()[1]][0]),
                                    np.abs([ax_CW.get_xlim()[0],ax_CW.get_ylim()[0]][1]-[ax_CW.get_xlim()[1],ax_CW.get_ylim()[1]][1]),
                                    angle=0, linewidth=1, edgecolor='black', facecolor='none',zorder=11))

ax_SW.add_patch(patches.Rectangle(([ax_SW.get_xlim()[0],ax_SW.get_ylim()[0]][0],[ax_SW.get_xlim()[0],ax_SW.get_ylim()[0]][1]),
                                    np.abs([ax_SW.get_xlim()[0],ax_SW.get_ylim()[0]][0]-[ax_SW.get_xlim()[1],ax_SW.get_ylim()[1]][0]),
                                    np.abs([ax_SW.get_xlim()[0],ax_SW.get_ylim()[0]][1]-[ax_SW.get_xlim()[1],ax_SW.get_ylim()[1]][1]),
                                    angle=0, linewidth=1, edgecolor='black', facecolor='none',zorder=11))

ax_SW_zoom.add_patch(patches.Rectangle(([ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][0],[ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][1]),
                                       np.abs([ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][0]-[ax_SW_zoom.get_xlim()[1],ax_SW_zoom.get_ylim()[1]][0]),
                                       np.abs([ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][1]-[ax_SW_zoom.get_xlim()[1],ax_SW_zoom.get_ylim()[1]][1]),
                                       angle=0, linewidth=1, edgecolor='black', facecolor='none',zorder=11))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py

#NO
gl=ax_NO.gridlines(draw_labels=True, xlocs=[-30,-50,-70], ylocs=[78,79,80,81], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
gl.bottom_labels = False

#NW
gl=ax_NW.gridlines(draw_labels=True, xlocs=[-52,-53,-54,-55,-56,-57,-58,], ylocs=[73,74,75,76], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
gl.top_labels = False

#CW
gl=ax_CW.gridlines(draw_labels=True, xlocs=[-47,-48,-49,-50], ylocs=[69,70,71,72], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
gl.top_labels = False

#SW
gl=ax_SW.gridlines(draw_labels=True, xlocs=[-49,-48,-47], ylocs=[65,66,67,68], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
gl.top_labels = False

#SW zoom
gl=ax_SW_zoom.gridlines(draw_labels=True, xlocs=[-47.50], ylocs=[66.00], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
gl.top_labels = False
###################### From Tedstone et al., 2022 #####################

#Display rectangle around SWW zoom - this is from Fig1.py paper Greenland Ice Sheet Ice Slabs Expansion and Thickening  
#This is from https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
# Create a Rectangle patch and add the patch to the Axes
ax_SW.add_patch(patches.Rectangle(([ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][0],[ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][1]),
                                    np.abs([ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][0]-[ax_SW_zoom.get_xlim()[1],ax_SW_zoom.get_ylim()[1]][0]),
                                    np.abs([ax_SW_zoom.get_xlim()[0],ax_SW_zoom.get_ylim()[0]][1]-[ax_SW_zoom.get_xlim()[1],ax_SW_zoom.get_ylim()[1]][1]),
                                    angle=0, linewidth=1, edgecolor='black', facecolor='none'))
#Add panel label
ax_SW.text(ax_SW_zoom.get_xlim()[1]+5000, (ax_SW_zoom.get_ylim()[0]+ax_SW_zoom.get_ylim()[1])/2-35000,'e',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Patch(facecolor='#072f6b',edgecolor='none',label='Areas supporting runoff'),
                   Patch(facecolor='none',edgecolor='#ba2b2b',label='2010-2018 ice slabs'),
                   Line2D([0], [0], color='#fed976', lw=2, label='2013-2020 runoff limit'),
                   Patch(facecolor='#d9d9d9',edgecolor='none',label='Ignored areas')]
ax_NO.legend(handles=legend_elements,loc='lower center',fontsize=8,framealpha=1, bbox_to_anchor=(0.65, 0)).set_zorder(7)

# Display scalebar with GeoPandas
ax_NO.add_artist(ScaleBar(1,location='upper left',box_alpha=0,box_color=None)).set_pad(2)
ax_NW.add_artist(ScaleBar(1,location='upper right',box_alpha=0,box_color=None))
ax_CW.add_artist(ScaleBar(1,location='upper right',box_alpha=0,box_color=None))
ax_SW.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))
ax_SW_zoom.add_artist(ScaleBar(1,location='lower right',box_alpha=0,box_color=None))

plt.show()
pdb.set_trace()
'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig3/v4/Fig3_abcde.png',dpi=300)
'''

#Display 2019 RL in panel e
RL_v3=pd.read_csv(path_switchdrive+'/RT3/data/Emax/xytpd_NDWI_cleaned_2019_v3.csv')
RL2019_v3=RL_v3[RL_v3.year==2019].copy()
ax_SW_zoom.scatter(RL2019_v3.x,RL2019_v3.y,s=5,c='#f46d43',label='2019 runoff limits')
ax_SW_zoom.legend(fontsize=8,loc='upper center' ,bbox_to_anchor=(0.5, 0.95),handletextpad=0)#handletextpad is from https://stackoverflow.com/questions/41827406/how-to-adjust-space-between-legend-markers-and-labels
'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig3/v4/Fig3_abcde_2019RL.png',dpi=300)
'''
pdb.set_trace()

############################################################################
########################### Aquitard properties ###########################
############################################################################

###################### Load ice slabs with SAR dataset ######################
#Path to data
path_SAR_And_IceThickness=path_local+'SAR_and_IceThickness/csv/'
#List all the files in the folder
list_composite=os.listdir(path_SAR_And_IceThickness) #this is inspired from https://pynative.com/python-list-files-in-a-directory/
#Define empty dataframe
upsampled_SAR_and_IceSlabs=pd.DataFrame()
#Loop over all the files
for indiv_file in list_composite:
    print(indiv_file)
    #Open the individual file
    indiv_csv=pd.read_csv(path_SAR_And_IceThickness+indiv_file)
    #Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
    indiv_upsampled_SAR_and_IceSlabs=indiv_csv.groupby('index_right').mean()  
    #Append the data to each other
    upsampled_SAR_and_IceSlabs=pd.concat([upsampled_SAR_and_IceSlabs,indiv_upsampled_SAR_and_IceSlabs])    
###################### Load ice slabs with SAR dataset ######################

pdb.set_trace()
#Transform upsampled_SAR_and_IceSlabs as a geopandas dataframe to identify region
upsampled_SAR_and_IceSlabs_gdp = gpd.GeoDataFrame(upsampled_SAR_and_IceSlabs,
                                                  geometry=gpd.GeoSeries.from_xy(upsampled_SAR_and_IceSlabs['lon_3413'],
                                                                                 upsampled_SAR_and_IceSlabs['lat_3413'],
                                                                                 crs='EPSG:3413'))

#Intersection between dataframe and polygon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
upsampled_SAR_and_IceSlabs_gdp_with_regions = gpd.sjoin(upsampled_SAR_and_IceSlabs_gdp, GrIS_drainage_bassins, predicate='within')

#Apply thresholds to differentiate between efficient aquitard VS non-efficient aquitard places - let's choose quantile 0.75 of below
upsampled_SAR_and_IceSlabs_gdp_with_regions.loc[np.logical_and(upsampled_SAR_and_IceSlabs_gdp_with_regions.SUBREGION1=='SW',
                                                               upsampled_SAR_and_IceSlabs_gdp_with_regions.raster_values<-9.07748),
                                                'aquitard']=1

upsampled_SAR_and_IceSlabs_gdp_with_regions.loc[np.logical_and(upsampled_SAR_and_IceSlabs_gdp_with_regions.SUBREGION1=='CW',
                                                               upsampled_SAR_and_IceSlabs_gdp_with_regions.raster_values<-7.742007),
                                                'aquitard']=1

upsampled_SAR_and_IceSlabs_gdp_with_regions.loc[np.logical_and(upsampled_SAR_and_IceSlabs_gdp_with_regions.SUBREGION1=='NW',
                                                               upsampled_SAR_and_IceSlabs_gdp_with_regions.raster_values<-8.926942),
                                                'aquitard']=1

upsampled_SAR_and_IceSlabs_gdp_with_regions.loc[np.logical_and(upsampled_SAR_and_IceSlabs_gdp_with_regions.SUBREGION1=='NO',
                                                               upsampled_SAR_and_IceSlabs_gdp_with_regions.raster_values<-7.128138),
                                                'aquitard']=1

upsampled_SAR_and_IceSlabs_gdp_with_regions.loc[np.logical_and(upsampled_SAR_and_IceSlabs_gdp_with_regions.SUBREGION1=='NE',
                                                               upsampled_SAR_and_IceSlabs_gdp_with_regions.raster_values<-6.259047),
                                                'aquitard']=1

#Where upsampled_SAR_and_IceSlabs_gdp_with_regions.aquitard is nan, assign a 0 (=non-efficient aquitard)
upsampled_SAR_and_IceSlabs_gdp_with_regions.loc[upsampled_SAR_and_IceSlabs_gdp_with_regions.aquitard.isna(),'aquitard']=0

#Display on aquitard map
ax1.scatter(upsampled_SAR_and_IceSlabs_gdp.lon_3413,upsampled_SAR_and_IceSlabs_gdp.lat_3413,c='magenta',s=1)
ax1.scatter(upsampled_SAR_and_IceSlabs_gdp_with_regions.lon_3413,upsampled_SAR_and_IceSlabs_gdp_with_regions.lat_3413,c=upsampled_SAR_and_IceSlabs_gdp_with_regions['aquitard'],s=0.5,cmap='Reds_r')

#Display the ice thickness distributions
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_SAR = plt.subplot(gs[0:10, 0:6])
sns.violinplot(data=pd.DataFrame(upsampled_SAR_and_IceSlabs_gdp_with_regions.to_dict()),
               y="SUBREGION1", x="20m_ice_content_m",hue="aquitard",ax=ax_SAR)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
ax_SAR.set_xlabel('Ice Thickness [m]')
ax_SAR.set_ylabel('Region')
ax_SAR.set_title('GrIS-wide - 0m slab is missing here!')






















