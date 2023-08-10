# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:44:06 2022

@author: jullienn

This code include part of codes 'FigS7.py' from manuscript "Greenland Ice Slabs Thickening and Expansion"
"""

##############################################################################
################### Define function for ice lenses logging ###################
##############################################################################
#This function if adapted from https://stackoverflow.com/questions/37363755/python-mouse-click-coordinates-as-simply-as-possible
def onclick(event):
    #This functions print and save the x and y coordinates in pixels!
    print(event.xdata, event.ydata)
    #Fill in the file to log on the information
    filename_flog='C:/Users/jullienn/Documents/working_environment/iceslabs_MacFerrin/flog_may12_03_36_aggregated.txt'
    f_log = open(filename_flog, "a")
    f_log.write(str(round(event.xdata,6))+','+str(round(event.ydata,6))+'\n')
    f_log.close() #Close the quality assessment file when we’re done!
##############################################################################
################### Define function for ice lenses logging ###################
##############################################################################


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



def legend_building(color_palette,year_color):
    #Custom legend myself for axmap - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    list_legend_year = [Line2D([0], [0], color=color_palette[str(year_color)], lw=2, marker='o',linestyle='None', label=str(year_color),markeredgecolor='black')]
    return list_legend_year



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
from scalebar import scale_bar

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

'''
#Things to be done
1. Display 2002-03 data together with 2010-2018 (restricted to identical lon bounds)
    -> Open the radargrams saved (L1 product)
2. Display the correspoding Emax location of the specific year on the radargram
3. Display the Emax on a map aside for each year.
4. Display groups of Emax to observe any change
'''

### Set sizes ###
# this is from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
plt.rc('axes',linewidth = 0.75)  #https://stackoverflow.com/questions/1639463/how-to-change-border-width-of-a-matplotlib-graph
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
### Set sizes ###

#Define variables
#Compute the speed (Modified Robin speed):
# self.C / (1.0 + (coefficient*density_kg_m3/1000.0))
v= 299792458

desired_map='master_map'#'master_map' or 'NDWI'
display_Emax="False"

#Define paths to data
path_data_Jullien='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/i_out_from_IceBridgeGPR_Manager_v2.py/pickles_and_images/'
path_data='C:/Users/jullienn/Documents/working_environment/iceslabs_MacFerrin/data/'
path_20022003_data='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/radargram_data/'
path_data_switchdrive='C:/Users/jullienn/switchdrive/Private/research/RT3/data/'
path_rignotetal2016_GrIS_drainage_bassins='C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

#Define palette for time periods, this is from fig2_paper_icelsabs.py
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
#my_pal = {'2002': 'yellow', '2003': 'yellow', '2010': "#fdd49e", '2011': "#fc8d59", '2012': "#fc8d59", '2013':"#d7301f",'2014':"#d7301f",'2017':"#7f0000",'2018':"#7f0000"}
'''
my_pal = {'2002': '#08519c', '2003': '#4292c6', '2004': '#9ecae1', '2005': '#deebf7',
          '2006': '#ffffbf', '2007': '#feb24c', '2008': '#fc4e2a', '2009': '#bd0026',
          '2010': "#67000d", '2011': "#ce1256", '2012': "#c51b7d", '2013': "#f1b6da",
          '2014': "#e6f5d0", '2015': '#a1d99b', '2016': '#41ab5d', '2017': "#006d2c",
          '2018': "#00441b", '2019': '#01665e', '2020': 'black'}
'''
'''
my_pal = {'2002': '#1b7837', '2003': '#5aae61', '2004': '#a6dba0', '2005': '#d9f0d3',
          '2006': '#e7d4e8', '2007': '#c2a5cf', '2008': '#9970ab', '2009': '#762a83',
          '2010': "#a50026", '2011': "#d73027", '2012': "#f46d43", '2013': "#fdae61",
          '2014': "#fee090", '2015': '#e0f3f8', '2016': '#abd9e9', '2017': "#74add1",
          '2018': "#4575b4", '2019': '#313695', '2020': '#1a1a1a'}
'''
my_pal = {'2002': '#980043', '2003': '#dd1c77', '2004': '#df65b0',
          '2005': '#54278f', '2006': '#756bb1', '2007': '#9e9ac8', '2008': '#cbc9e2', '2009': '#f2f0f7',
          '2010': "#bd0026", '2011': "#f03b20", '2012': "#fd8d3c", '2013': "#fecc5c", '2014': "#ffffb2",
          '2015': '#006d2c', '2016': '#31a354', '2017': "#74c476", '2018': "#bae4b3", '2019': '#edf8e9',
          '2020': '#993404'}


from matplotlib.colors import  ListedColormap
#Create a palette for traffic_light display
traffic_light_cmap=ListedColormap(['purple','red','orange','green'],name='traffic_light_cmap') #This is from matplotlib help 'Creating Colormaps in Matplotlib'

#Load Rignot et al., 2016 Greenland drainage bassins
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

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

#Load 2002-2003 ice slabs identification
path_2002_2003_slabs='C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/2002_2003/radargrams_and_identification/identification/'
xls_icelenses = pd.read_excel(path_2002_2003_slabs+'icelenses_22022020.xls', sheet_name=None,header=2)
trafic_light=pd.read_excel(path_2002_2003_slabs+'icelenses_22022020.xls', sheet_name=None,header=1)

#Herebelow are all the potential overal between 2002-2003 with 2010-2018

#Case study 1 (Fig 3.f in paper 1)
CaseStudy1={2002:['jun04_02proc_4.mat'],
            2003:['may12_03_36_aggregated'],
            2010:['Data_20100513_01_001.mat','Data_20100513_01_002.mat'],
            2011:['Data_20110411_01_116.mat','Data_20110411_01_117.mat','Data_20110411_01_118.mat'],#2011 is reversed
            2012:['Data_20120428_01_125.mat','Data_20120428_01_126.mat'],#2012 is reversed
            2013:'empty',
            2014:['Data_20140408_11_024.mat','Data_20140408_11_025.mat','Data_20140408_11_026.mat'],#2014 is reversed
            2017:['Data_20170508_02_165.mat','Data_20170508_02_166.mat','Data_20170508_02_167.mat','Data_20170508_02_168.mat','Data_20170508_02_169.mat','Data_20170508_02_170.mat','Data_20170508_02_171.mat'],#2017 is reversed
            2018:'empty'}

#Case study 2 (Fig. 3d in paper 1)
CaseStudy2={2002:'empty',
            2003:['may12_03_1_aggregated','may12_03_2_aggregated'],
            2010:['Data_20100508_01_114.mat','Data_20100508_01_115.mat'],#2010 is reversed
            2011:['Data_20110419_01_008.mat','Data_20110419_01_009.mat','Data_20110419_01_010.mat'],
            2012:['Data_20120418_01_129.mat','Data_20120418_01_130.mat','Data_20120418_01_131.mat'],#2012 is reversed
            2013:['Data_20130405_01_165.mat','Data_20130405_01_166.mat','Data_20130405_01_167.mat'],#2013 is reversed
            2014:['Data_20140424_01_002.mat','Data_20140424_01_003.mat','Data_20140424_01_004.mat'],
            2017:['Data_20170422_01_168.mat','Data_20170422_01_169.mat','Data_20170422_01_170.mat','Data_20170422_01_171.mat'],#2017 is reversed
            2018:['Data_20180427_01_170.mat','Data_20180427_01_171.mat','Data_20180427_01_172.mat']}#2018 is reversed

#Case study 3 (Fig. 3c in paper 1). For this one, it seems that we should enlarge the buffer where to extract Emax, and keep only high retrievals, not the closest ones.
CaseStudy3={2002:['jun04_02proc_52.mat','jun04_02proc_53.mat'],
            2003:'empty',
            2010:['Data_20100507_01_008.mat','Data_20100507_01_009.mat','Data_20100507_01_010.mat'],
            2011:['Data_20110426_01_009.mat','Data_20110426_01_010.mat','Data_20110426_01_011.mat'],
            2012:'empty',
            2013:'empty',
            2014:['Data_20140421_01_009.mat','Data_20140421_01_010.mat','Data_20140421_01_011.mat','Data_20140421_01_012.mat','Data_20140421_01_013.mat'],
            2017:['Data_20170424_01_008.mat','Data_20170424_01_009.mat','Data_20170424_01_010.mat','Data_20170424_01_011.mat','Data_20170424_01_012.mat','Data_20170424_01_013.mat','Data_20170424_01_014.mat'],
            2018:'empty'}

#Case study 1, 2 and 3 are in SW Greenland.

#Case study 4 is in NW Greenland and is divided into a lower part of transect and an upper part.
#Case study 4 - upper part transect: Not usable as it is above the Emax limit.
CaseStudy4={2002:['may18_02_0_aggregated'],
            2003:['may13_03_29_aggregated','may13_03_30_aggregated'], #['may14_03_51_aggregated','may14_03_52_aggregated'] is as good as the one used now
            2010:['Data_20100517_02_001.mat','Data_20100517_02_002.mat'],
            2011:['Data_20110502_01_171.mat'],
            2012:['Data_20120516_01_115.mat'],
            2013:['Data_20130426_01_006.mat','Data_20130426_01_007.mat'],
            2014:['Data_20140514_02_087.mat','Data_20140514_02_088.mat','Data_20140514_02_089.mat'],
            2017:['Data_20170417_01_171.mat','Data_20170417_01_172.mat','Data_20170417_01_173.mat','Data_20170417_01_174.mat'],
            2018:'empty'}

#Case study 4 - lower part transect: To catch Emax points, lower and upper limits must be expanded to +/-3e3.
#Transect not usable as no clear ice slabs signature in 2002, not a lot of Emax points through time, and the inland expansion of Emax
#is not as clear as for case study 1, 2, 3.
CaseStudy4={2002:'empty', #['may30_02_51_aggregated'] is bad
            2003:['may13_03_29_aggregated','may13_03_30_aggregated'],
            2010:['Data_20100517_02_001.mat','Data_20100517_02_002.mat'],
            2011:['Data_20110502_01_171.mat'],
            2012:['Data_20120330_01_124.mat','Data_20120330_01_125.mat'],#The one used is the best. Or ['TESTED Data_20120516_01_002.mat'],,#['TESTED Data_20120516_01_115.mat'],
            2013:['Data_20130419_01_004.mat','Data_20130419_01_005.mat'],#The one used is the best. Or ['TESTED Data_20130426_01_006.mat','Data_20130426_01_007.mat'],
            2014:['Data_20140507_03_007.mat','Data_20140507_03_008.mat'],#The one used is the best. Or ['TESTED Data_20140519_02_002.mat','Data_20140519_02_003.mat','Data_20140519_02_004.mat'],#['TESTED Data_20140515_02_173.mat','Data_20140515_02_174.mat','Data_20140515_02_175.mat'],#['TESTED Data_20140515_02_001.mat','Data_20140515_02_002.mat','Data_20140515_02_003.mat'],#['TESTED Data_20140429_02_160.mat','Data_20140429_02_161.mat'],#['BEST Data_20140514_02_087.mat','Data_20140514_02_088.mat','Data_20140514_02_089.mat'],
            2017:['Data_20170417_01_171.mat','Data_20170417_01_172.mat','Data_20170417_01_173.mat','Data_20170417_01_174.mat'],
            2018:'empty'}

#Case study 5 - Not usable: the transect is in the NO and is not perpendicular to the elevation contours,
#hence it intersects a first zone of runoff, the goes into the dry snow zone before intersecting a another zone of runoff.
#Furthermore, mo big changes in ice slabs thickness there
CaseStudy5={2002:'empty',
            2003:['may14_03_6_aggregated','may14_03_7_aggregated'],
            2010:'empty',
            2011:['Data_20110329_01_013.mat','Data_20110329_01_014.mat','Data_20110329_01_015.mat','Data_20110329_01_016.mat','Data_20110329_01_017.mat','Data_20110329_01_018.mat'],
            2012:['Data_20120330_01_018.mat','Data_20120330_01_019.mat','Data_20120330_01_020.mat','Data_20120330_01_021.mat','Data_20120330_01_022.mat','Data_20120330_01_023.mat'],
            2013:'empty',
            2014:'empty',
            2017:'empty',
            2018:'empty'}

#Case study 6 - Not usable as the transect is in the NE right a the edge of glacier 79, hence no Emax in the immediate vicinity of transect.
CaseStudy6={2002:['may18_02_28_aggregated','may18_02_29_aggregated'],
            2003:'empty',
            2010:'empty',
            2011:'empty',
            2012:'empty',
            2013:'empty',
            2014:['Data_20140429_02_076.mat','Data_20140429_02_077.mat','Data_20140429_02_078.mat','Data_20140429_02_079.mat'], #OR 20140508_03_019_024
            2017:['Data_20170328_01_095.mat','Data_20170328_01_096.mat','Data_20170328_01_097.mat','Data_20170328_01_098.mat','Data_20170328_01_099.mat','Data_20170328_01_100.mat','Data_20170328_01_101.mat'], 
            2018:'empty'}

#Case study 7 - Not usable as transect more or less parallel to elevation contours
CaseStudy7={2002:'empty',
            2003:['may12_03_9_aggregated','may12_03_10_aggregated'],
            2010:['Data_20100508_01_084.mat'], #2010 a bit offset towards the east
            2011:'empty',
            2012:'empty',
            2013:'empty',
            2014:'empty',
            2017:['Data_20170422_01_138.mat','Data_20170422_01_139.mat'],
            2018:'empty'}

#Case study FS - closest overlapping 2003 with 2010-2018 data. However, the transects seem to offset to me to be studied.
CaseStudyFS={2002:'empty',
            2003:['may09_03_1_aggregated'],
            2010:'empty',
            2011:'empty',
            2012:['Data_20120423_01_006.mat','Data_20120423_01_007.mat'],
            2013:'empty',
            2014:'empty',
            2017:['Data_20170505_02_008.mat','Data_20170505_02_009.mat','Data_20170505_02_010.mat'],
            2018:'empty'}

#Define the panel to study
investigation_year=CaseStudy2

#Create figures
plt.rcParams.update({'font.size': 20})
fig2 = plt.figure()
ax_map = plt.subplot(projection=crs)
GrIS_drainage_bassins.plot(ax=ax_map,facecolor='none',edgecolor='black')

plt.rcParams.update({'font.size': 8})
#fig1 = plt.figure(figsize=(8.27,5.435))#Nature pdf size = (8.27,10.87)
fig1 = plt.figure(figsize=(8.27,5.435))#Nature pdf size = (8.27,10.87)
gs = gridspec.GridSpec(30, 101)
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
    axc = plt.subplot(gs[8:28, 100:101])
    '''
    #For ice layers identification for Fig. 1 in paper Slabs thickening
    ax1 = plt.subplot(gs[0:6, 0:100])
    ax2 = plt.subplot(gs[6:12, 0:100])
    ax3 = plt.subplot(gs[12:18, 0:100])
    ax4 = plt.subplot(gs[18:20, 0:100])
    ax5 = plt.subplot(gs[20:22, 0:100])
    ax7 = plt.subplot(gs[22:24, 0:100])
    ax8 = plt.subplot(gs[24:26, 0:100])
    axc = plt.subplot(gs[12:26, 100:101])
    '''
elif (investigation_year==CaseStudy2):
    '''
    ax2 = plt.subplot(gs[0:4, 0:100])
    ax3 = plt.subplot(gs[4:8, 0:100])
    ax4 = plt.subplot(gs[8:12, 0:100])
    ax5 = plt.subplot(gs[12:16, 0:100])
    ax6 = plt.subplot(gs[16:20, 0:100])
    ax7 = plt.subplot(gs[20:24, 0:100])
    ax8 = plt.subplot(gs[24:28, 0:100])
    ax9 = plt.subplot(gs[28:32, 0:100])
    axc = plt.subplot(gs[4:32, 100:101])
    '''
    #Fig. 6 paper
    ax2 = plt.subplot(gs[0:4, 0:99])
    ax3 = plt.subplot(gs[4:8, 0:99])
    ax7 = plt.subplot(gs[8:12, 0:99])
    ax9 = plt.subplot(gs[12:16, 0:99])
    axc = plt.subplot(gs[4:16, 99:101])
    #Display map on the same figure
    ax_map = plt.subplot(gs[18:30, 0:99], projection=crs)
    axc_map = plt.subplot(gs[20:28, 99:101])
    '''
    #Display NDWI map below
    ax_NDWI = plt.subplot(gs[40:54, 0:99], projection=crs)
    axc_NDWI = plt.subplot(gs[42:52, 99:101])
    '''
elif (investigation_year==CaseStudy3):
    ax1 = plt.subplot(gs[0:4, 0:100])
    ax3 = plt.subplot(gs[4:8, 0:100])
    ax4 = plt.subplot(gs[8:12, 0:100])
    ax7 = plt.subplot(gs[12:16, 0:100])
    ax8 = plt.subplot(gs[16:20, 0:100])
    axc = plt.subplot(gs[4:20, 100:101])

elif (investigation_year==CaseStudy4):
    ax1 = plt.subplot(gs[0:4, 0:100])
    ax2 = plt.subplot(gs[4:8, 0:100])
    ax3 = plt.subplot(gs[8:12, 0:100])
    ax4 = plt.subplot(gs[12:16, 0:100])
    ax5 = plt.subplot(gs[16:20, 0:100])
    ax6 = plt.subplot(gs[20:24, 0:100])
    ax7 = plt.subplot(gs[24:28, 0:100])
    ax8 = plt.subplot(gs[28:32, 0:100])
    axc = plt.subplot(gs[8:32, 100:101])
elif (investigation_year==CaseStudy5):
    ax2 = plt.subplot(gs[0:4, 0:100])
    ax4 = plt.subplot(gs[4:8, 0:100])
    ax5 = plt.subplot(gs[8:12, 0:100])
    axc = plt.subplot(gs[4:12, 100:101])
elif (investigation_year==CaseStudy6):
    ax1 = plt.subplot(gs[0:4, 0:100])
    ax7 = plt.subplot(gs[4:8, 0:100])
    ax8 = plt.subplot(gs[8:12, 0:100])
    axc = plt.subplot(gs[4:12, 100:101])
elif (investigation_year==CaseStudy7):
    ax2 = plt.subplot(gs[0:4, 0:100])
    ax3 = plt.subplot(gs[4:8, 0:100])
    ax8 = plt.subplot(gs[8:12, 0:100])
    axc = plt.subplot(gs[4:12, 100:101])
    
elif (investigation_year==CaseStudyFS):
    ax2 = plt.subplot(gs[0:4, 0:100])
    ax5 = plt.subplot(gs[4:8, 0:100])
    ax8 = plt.subplot(gs[8:12, 0:100])
    axc = plt.subplot(gs[4:12, 100:101])
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
        
        #Set empty vector for transect lon, depth and color of ice lenses storing
        lon_transect_lenses=[]
        lat_transect_lenses=[]
        depth_transect_lenses=[]
        color_transect_lenses=[]

        for indiv_file_load in investigation_year[single_year]:
            
            #Open file
            f_20022003_L1 = open(path_20022003_data+str(single_year)+'/L1_'+indiv_file_load.split('.mat')[0]+'.pickle', "rb")
            L1_2002003 = pickle.load(f_20022003_L1)
            f_20022003_L1.close()
                        
            #Store data for the whole transect
            radargram_20022003=np.concatenate((radargram_20022003,L1_2002003['radar_slice_0_30m']),axis=1)
            
            #Load lat/lon
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
            
            #Set empty vector for lon, depth and color of ice lenses storing
            lon_lenses_indiv_file=[]
            lat_lenses_indiv_file=[]
            depth_lenses_indiv_file=[]
            color_lenses_indiv_file=[]
            
            #Load 2002-2003 ice lenses identification      
            ### This is from Fig1.py from paper 'Greenland Ice Slabs Thickening and Expansion ###
            if (indiv_file_load in list(xls_icelenses.keys())):                
                print(indiv_file_load+' hold ice lens!')
                #This file have ice lenses in it: read the data:
                df_temp=xls_icelenses[indiv_file_load]
                df_colnames = list(df_temp.keys())
                x_loc=[]
                
                #Trafic light information
                df_trafic_light=trafic_light[indiv_file_load]
                df_colnames_trafic_light = list(df_trafic_light.keys())
                
                for i in range (0,int(len(df_colnames)),2):
                    x_vect=df_temp[df_colnames[i]]
                    y_vect=df_temp[df_colnames[i+1]]
                    #Load trafic light color
                    trafic_light_indiv_color=df_colnames_trafic_light[i]
                    #Define the color in which to display the ice lens
                    if (trafic_light_indiv_color[0:3]=='gre'):
                        color_to_display=1
                    elif (trafic_light_indiv_color[0:3]=='ora'):
                        color_to_display=0
                    elif (trafic_light_indiv_color[0:3]=='red'):
                        color_to_display=-1
                    elif (trafic_light_indiv_color[0:3]=='pur'):
                        color_to_display=2
                    else:
                        print('The color is not known!')
                    #Convert from pixel space to geographical referenced space
                    #1. Get rid of NaNs and transform into int
                    x_vect=x_vect[~np.isnan(x_vect)].astype(int)
                    y_vect=y_vect[~np.isnan(y_vect)].astype(int)
                    #If identification out of radargram, pick the maximum len of radargram
                    x_vect[x_vect>(L1_2002003['radar_slice_0_30m'].shape[1]-1)]=(L1_2002003['radar_slice_0_30m'].shape[1]-1)
                    #If data deeper than maximum depth, pick the maximum depth
                    y_vect[y_vect>(len(L1_2002003['depths'])-1)]=(len(L1_2002003['depths'])-1)
                    #2. Extract latitude, longitude and depth
                    lon_lenses=L1_2002003['lon_3413'][x_vect]
                    lat_lenses=L1_2002003['lat_3413'][x_vect]
                    depth_lenses=L1_2002003['depths'][y_vect]
                    #3. Extract the colour code
                    color_lenses=np.ones(len(lon_lenses))*color_to_display
                    #4. Append data
                    lon_lenses_indiv_file=np.append(lon_lenses_indiv_file,lon_lenses)
                    lat_lenses_indiv_file=np.append(lat_lenses_indiv_file,lat_lenses)
                    depth_lenses_indiv_file=np.append(depth_lenses_indiv_file,depth_lenses)
                    color_lenses_indiv_file=np.append(color_lenses_indiv_file,color_lenses)

            #Append individual file data to each other
            lon_transect_lenses=np.append(lon_transect_lenses,lon_lenses_indiv_file)
            lat_transect_lenses=np.append(lat_transect_lenses,lat_lenses_indiv_file)
            depth_transect_lenses=np.append(depth_transect_lenses,depth_lenses_indiv_file)
            color_transect_lenses=np.append(color_transect_lenses,color_lenses_indiv_file)
            ### This is from Fig1.py from paper 'Greenland Ice Slabs Thickening and Expansion ###
                    
        #If from low to high elevations, fliplr
        if (np.sum(np.diff(lon_20022003)<0)):
            print(single_year,' is reversed')
            radargram_20022003=np.fliplr(radargram_20022003)
            lat_20022003=np.flipud(lat_20022003)
            lon_20022003=np.flipud(lon_20022003)
            lon_transect_lenses=np.flipud(lon_transect_lenses)
            lat_transect_lenses=np.flipud(lat_transect_lenses)
            depth_transect_lenses=np.flipud(depth_transect_lenses)
            color_transect_lenses=np.flipud(color_transect_lenses)

        #Calculate distances
        distances_with_start_transect=compute_distances(lon_20022003,lat_20022003)
        
        #Convert lat/lon into EPSG:4326
        #Transform the coordinates from EPSG:3413 to EPSG:4326
        #Example from: https://pyproj4.github.io/pyproj/stable/examples.html
        points=transformer_3413_to_4326.transform(np.array(lon_20022003),np.array(lat_20022003))
        lon_appended=points[0]
        lat_appended=points[1]
        
        #Convert lon_transect_lenses into EPSG:4326
        #Transform the coordinates from EPSG:3413 to EPSG:4326
        #Example from: https://pyproj4.github.io/pyproj/stable/examples.html
        points_lenses=transformer_3413_to_4326.transform(np.array(lon_transect_lenses),np.array(lat_transect_lenses))
        lon_transect_4326_lenses=points_lenses[0]
        
        #Store into a dictionnary:
        dataframe[str(single_year)]={'lat_appended':lat_appended,
                                     'lon_appended':lon_appended,
                                     'lat_3413':lat_20022003,
                                     'lon_3413':lon_20022003,
                                     'distances':distances_with_start_transect,
                                     'depth':L1_2002003['depths'],
                                     'radargram_30m':radargram_20022003,
                                     'lon_transect_4326_lenses':lon_transect_4326_lenses,
                                     'depth_transect_lenses':depth_transect_lenses,
                                     'color_transect_lenses':color_transect_lenses}
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

#For storing transect coordinates
lat_transet=[]
lon_transet=[]

for single_year in investigation_year.keys():

    #We do not display the following radargrams
    if (single_year in list([2002,2011,2012,2013,2017])):
       continue
    
    if (investigation_year[single_year]=='empty'):
        continue

    print(single_year)
    
    if (investigation_year==CaseStudy2):
        start_transect=-47.70785561652585
        end_transect=-46.85#-46.41555609606877
        vmin_plot=-4.5
        vmax_plot=4.5
    elif (investigation_year==CaseStudy1):
        '''
        #For ice layers identification for Fig. 1 in paper Slabs thickening
        start_transect=-49
        end_transect=-46.53
        '''
        start_transect=-48.21060856534727
        end_transect=-46.88764316176339
        vmin_plot=-4.5
        vmax_plot=4.5        
    elif (investigation_year==CaseStudy3):
        start_transect=-47.567
        end_transect=-46.5427
        '''
        #For ice layers identification for Fig. 1 in paper Slabs thickening
        start_transect=-48
        end_transect=-46.53
        '''
        vmin_plot=-4.5
        vmax_plot=4.5
    elif (investigation_year==CaseStudy4):
        start_transect=-67.0078#lower transect: -67.0078 #upper transect: -65.81
        end_transect=-65.81#lower transect: -65.81 # upper transect: -64.8225
        vmin_plot=-4.5
        vmax_plot=4.5
    elif (investigation_year==CaseStudy5):
        start_transect=-61.2749
        end_transect=-58.9
        vmin_plot=-4.5
        vmax_plot=4.5      
    elif (investigation_year==CaseStudy6):
        start_transect=-25.9
        end_transect=-24.0842
        vmin_plot=-4.5
        vmax_plot=4.5
    elif (investigation_year==CaseStudy7):
        start_transect=-34.982#-33.5882
        end_transect=-32.1472#-32.5967
        vmin_plot=-4.5
        vmax_plot=4.5
    elif (investigation_year==CaseStudyFS):
        start_transect=-47.7728
        end_transect=-46.4053
        vmin_plot=-4.5
        vmax_plot=4.5
        
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
        ax_plot.invert_yaxis() #Invert the y axis = avoid using flipud.
        
        '''
        #### ORIGINAL ICE LAYERS IDENTIFICATION ####
        #Display the 2002-2003 green ice lenses identification
        ax_plot.scatter(dataframe[str(single_year)]['lon_transect_4326_lenses'],
                        dataframe[str(single_year)]['depth_transect_lenses'],
                        c=dataframe[str(single_year)]['color_transect_lenses'],
                        cmap=traffic_light_cmap,vmin=-2,vmax=1,s=0.1)#add colour code!!
        '''
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
    
    '''
    #For ice layers identification for Fig. 1 in paper Slabs thickening
    #Set lims
    ax_plot.set_ylim(30,0)
    #Set yticklabels
    ax_plot.set_yticks([0,10,20,30])
    ax_plot.set_yticklabels(['0','10','20',''])
    '''
    #Set transect limits
    ax_plot.set_xlim(start_transect,end_transect)
    
    #Get rid of xticklabels
    ax_plot.set_xticklabels([])
    
    #Display year
    if (single_year<2004):
        ax_plot.text(0.98, 0.6,str(single_year),ha='center', va='center', transform=ax_plot.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    else:
        ax_plot.text(0.98, 0.875,str(single_year),ha='center', va='center', transform=ax_plot.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    
    #Display radargram track on the map
    index_within_bounds=np.logical_and(dataframe[str(single_year)]['lon_appended']>=start_transect,dataframe[str(single_year)]['lon_appended']<=end_transect)
    ax_map.scatter(dataframe[str(single_year)]['lon_3413'][index_within_bounds],dataframe[str(single_year)]['lat_3413'][index_within_bounds],s=0.1,zorder=1,color='black')#color=my_pal[str(single_year)])#)
    '''
    ax_NDWI.scatter(dataframe[str(single_year)]['lon_3413'][index_within_bounds],dataframe[str(single_year)]['lat_3413'][index_within_bounds],s=0.1,zorder=1,color='black')#color=my_pal[str(single_year)])#)
    '''
    
    '''
    #Store the coordinates of displayed transect
    lat_transet=np.append(lat_transet,dataframe[str(single_year)]['lat_3413'][index_within_bounds])
    lon_transet=np.append(lon_transet,dataframe[str(single_year)]['lon_3413'][index_within_bounds])
    '''

plt.show()

'''
#For ice layers identification for Fig. 1 in paper Slabs thickening
#Create the file to log on the information
filename_flog='C:/Users/jullienn/Documents/working_environment/iceslabs_MacFerrin/flog_may12_03_36_aggregated.txt'
f_log = open(filename_flog, "a")
f_log.write('xcoord'+','+'ycoord'+'\n')
f_log.close() #Close the file

fig1.canvas.mpl_connect('key_press_event', onclick)
plt.show()
pdb.set_trace()

#Open, display and check
xls_new_icelenses = pd.read_csv('C:/Users/jullienn/Documents/working_environment/iceslabs_MacFerrin/flog_may12_03_36_aggregated_6.csv',sep=';',decimal=',')
#Display the 2002-2003 green ice lenses identification
ax1.plot(np.array(xls_new_icelenses['xcoord']),np.array(xls_new_icelenses['ycoord']))
'''

'''
#In order to generate transect coordinates.
Coordinates_CSFS=pd.DataFrame({'lon':dataframe['2017']['lon_appended'],'lat':dataframe['2017']['lat_appended']})
Coordinates_CSFS.to_csv('C:/Users/jullienn/switchdrive/Private/research/RT1/final_dataset_2002_2018/IceSlabs_And_Coordinates/Coordinates_CSFS.csv')
'''

#Display the map
if (investigation_year==CaseStudy1):
    year_limit=2017
elif (investigation_year==CaseStudy2):
    year_limit=2014
elif (investigation_year==CaseStudy3):
    year_limit=2017
elif (investigation_year==CaseStudy4):
    year_limit=2017
elif (investigation_year==CaseStudy5):
    year_limit=2011
elif (investigation_year==CaseStudy6):
    year_limit=2017
elif (investigation_year==CaseStudy7):
    year_limit=2003
elif (investigation_year==CaseStudyFS):
    year_limit=2017
else:
    print('Year not known')
    pdb.set_trace()

#Find index of start and end of displayed transects
index_start_map=np.argmin(np.abs(np.abs(dataframe[str(year_limit)]['lon_appended'])-np.abs(start_transect)))
index_end_map=np.argmin(np.abs(np.abs(dataframe[str(year_limit)]['lon_appended'])-np.abs(end_transect)))

#Define bounds of cumulative raster map to display
x_min=dataframe[str(year_limit)]['lon_3413'][index_start_map]
x_max=dataframe[str(year_limit)]['lon_3413'][index_end_map]
y_min=dataframe[str(year_limit)]['lat_3413'][index_start_map]-3e3
y_max=dataframe[str(year_limit)]['lat_3413'][index_end_map]+3e3

#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
import rioxarray as rxr

### ------------- This is from Greenland_Hydrology_Summary.py ------------- ###
if (desired_map=='NDWI'):
    desired_year=2019
    path_NDWI='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/NDWI/'
    #Load NDWI data for display
    MapPlot = rxr.open_rasterio(path_NDWI+'NDWI_p10_'+str(desired_year)+'.vrt',
                                  masked=True).squeeze() #No need to reproject satelite image
    vlim_min=0
    vlim_max=0.3
    cmap_raster='Blues'
    
elif (desired_map=='master_map'):
    #Load hydrological master map from Tedstone and Machuguth (2022)
    path_CumHydroMap='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/master_maps/'
    #Load master_maps data for display
    MapPlot = rxr.open_rasterio(path_CumHydroMap+'master_map_GrIS_mean.vrt',
                                  masked=True).squeeze() #No need to reproject satelite image
    vlim_min=50
    vlim_max=150
    cmap_raster='viridis_r'#'magma_r'

else:
    print('Enter a correct map name!')
    pdb.set_trace()
### ------------- This is from Greenland_Hydrology_Summary.py ------------- ###

#Extract x and y coordinates of image
x_coord_MapPlot=np.asarray(MapPlot.x)
y_coord_MapPlot=np.asarray(MapPlot.y)
### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###

#Extract coordinates ofcumulative raster within Emaxs bounds
logical_x_coord_within_bounds=np.logical_and(x_coord_MapPlot>=x_min,x_coord_MapPlot<=x_max)
x_coord_within_bounds=x_coord_MapPlot[logical_x_coord_within_bounds]
logical_y_coord_within_bounds=np.logical_and(y_coord_MapPlot>=y_min,y_coord_MapPlot<=y_max)
y_coord_within_bounds=y_coord_MapPlot[logical_y_coord_within_bounds]

#Define extents based on the bounds
extent_MapPlot = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
#Display image
cbar=ax_map.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap=cmap_raster,vmin=vlim_min,vmax=vlim_max,zorder=0)

#Set xlims
ax_map.set_xlim(x_min,x_max)
ax_map.set_ylim(y_min,y_max)

if (display_Emax == "True"):
    count=0
    
    #Create a list of years holding data
    list_holding_data=[]
    for holding_data in investigation_year.keys():
        
        if (investigation_year[holding_data]!='empty'):
            list_holding_data=np.append(list_holding_data,holding_data)
    
    #Display Emax
    for single_year in range(2002,2021):
        
        #If no data before this year, continue
        if (single_year<list_holding_data[0]):
            print('No data in',str(single_year))
            continue
        
        print(single_year)
        
        #Select data of the desired year
        points_Emax_single_year=points_Emax[points_Emax.year==single_year]
        '''
        #In case yearly map of Emax point are desired to be plotted
        #Reset clean raster
        cbar=ax_map.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap='Blues',vmin=vlim_min,vmax=vlim_max,zorder=count+1) #NDWI, this is from Greenland_Hydrology_Summary.py
        #Display year
        ax_map.set_title(str(single_year))
        #Display all Emax points of this year
        ax_map.scatter(points_Emax_single_year['x'],points_Emax_single_year['y'],color='black',s=5,zorder=count+1)
        #Set xlims
        ax_map.set_xlim(x_min,x_max)
        ax_map.set_ylim(y_min,y_max)
        '''
        #Select the transect around which to perform Emax extraction
        if (single_year in list_holding_data):
            #We have a transect on this particular year, select it
            year_transect=single_year
        else:
            #We do not hate a transect on this particulat year, select the closest previous year
            #Restric the lis of years from the start to the year in question
            year_list=list_holding_data[list_holding_data<=single_year]
            year_transect=np.max(year_list).astype(int)
            print('Chosen year:', str(year_transect))
        
        #Create upper and lower line around chosen transect to extract Emax points
        index_within_bounds_transect=np.logical_and(dataframe[str(year_transect)]['lon_appended']>=start_transect,dataframe[str(year_transect)]['lon_appended']<=end_transect)
        upper_transect_lim = pd.DataFrame({'lon_3413_transect': dataframe[str(year_transect)]['lon_3413'][index_within_bounds_transect], 'lat_3413_transect': dataframe[str(year_transect)]['lat_3413'][index_within_bounds_transect]+1.1e3})#I choose 1.1e3 to include the closest 2012 Emax point
        transect_centroid = pd.DataFrame({'lon_3413_transect': dataframe[str(year_transect)]['lon_3413'][index_within_bounds_transect], 'lat_3413_transect': dataframe[str(year_transect)]['lat_3413'][index_within_bounds_transect]})
        lower_transect_lim = pd.DataFrame({'lon_3413_transect': dataframe[str(year_transect)]['lon_3413'][index_within_bounds_transect], 'lat_3413_transect': dataframe[str(year_transect)]['lat_3413'][index_within_bounds_transect]-1.1e3})#I choose 1.1e3 to include the closest 2012 Emax point
        
        '''
        ############################# TO COMMENT LATER ON #############################
        #Display upper and lower limits
        ax_map.plot(upper_transect_lim['lon_3413_transect'],upper_transect_lim['lat_3413_transect'],color='black',zorder=count+1)
        ax_map.plot(lower_transect_lim['lon_3413_transect'],lower_transect_lim['lat_3413_transect'],color='black',zorder=count+1)
        ############################# TO COMMENT LATER ON #############################
        '''
        
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
        #Create polygon patch of the polygon above
        plot_poylgon_Emax_extraction = PolygonPatch(polygon_Emax_extraction,zorder=count+1,edgecolor='red',facecolor='none')
        #Display plot_poylgon_Emax_extraction
        ax_map.add_patch(plot_poylgon_Emax_extraction)
        '''
            
        #Convert polygon of polygon_Emax_extraction into a geopandas dataframe
        polygon_Emax_extraction_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:3413', geometry=[polygon_Emax_extraction]) #from https://gis.stackexchange.com/questions/395315/shapely-coordinate-sequence-to-geodataframe
        #Intersection between points_Emax_single_year and polygon_Emax_extraction_gpd, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        Emax_extraction = gpd.sjoin(points_Emax_single_year, polygon_Emax_extraction_gpd, op='within')
        
        if (len(Emax_extraction)==0):
            '''
            #In case yearly map of Emax point are desired to be plotted
            #Save figure
            plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Section1/CS1/CS1_NDWI_Emax_'+str(single_year)+'.png',dpi=300,bbox_inches='tight')
            #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
            '''
            count=count+2
            continue
        
        #pdb.set_trace()
        '''
        #Plot the result of this selection
        ax_map.scatter(Emax_extraction['x'],Emax_extraction['y'],color='red',s=10,zorder=count+1)
        '''
        ### ------------------ This is from Emax_SlabsTickness.py ----------------- ###
        
    
        ### --------------- Inspired from Emax_Slabs_tickness.py -------------- ###
        
        ###########################################################################
        ### After a quick look at kept best Emax point for each year with the   ###
        ### method closest, I come to the conclusion that sometimes the closest ###
        ### Emax point is best, sometimes the highest Emax point is best to     ###
        ### relate the actual Emax. Hence, I keep the closest Emax point and    ###
        ### the highest Emax point each year.                                   ###
        ###########################################################################
        
        #Keep the closest Emax_extraction from the transect
        distances_Emax_extraction_transect=Emax_extraction.geometry.distance(transect_centroid_transect)#Calculate distance of each Emax points with respect to the transect centroid, this is from https://shapely.readthedocs.io/en/stable/manual.html
        closest_Emax_extraction=Emax_extraction.iloc[np.where(distances_Emax_extraction_transect==np.min(distances_Emax_extraction_transect))]#Select this closest point
        
        '''
        pdb.set_trace()
        ax_map.scatter(closest_Emax_extraction['x'],closest_Emax_extraction['y'],c='cyan',s=10,zorder=count+1)#Display this closest point
        '''
        
        #Keep the highest Emax_extraction from the transect
        highest_Emax_extraction=Emax_extraction.iloc[np.where(Emax_extraction.elev==np.max(Emax_extraction.elev))]#Select this highest point
        #if we have to points having the same highest elevation, select the easternmost one
        if (len(highest_Emax_extraction)>1):
            #Keep only the easternmost Emax point, i.e. the most positive longitute
            highest_Emax_extraction=highest_Emax_extraction.iloc[np.where(highest_Emax_extraction.x==np.max(highest_Emax_extraction.x))]
        
        '''
        pdb.set_trace()
        ax_map.scatter(highest_Emax_extraction['x'],highest_Emax_extraction['y'],c='yellow',s=10,zorder=count+1)#Display this easternmost point
        '''
        
        #Store the closest and highest points
        best_Emax_points=pd.concat([closest_Emax_extraction, highest_Emax_extraction])
        
        #If the difference in elevation between the two points is larger than 50m (to change?), we probably have two Emax points at two different hydrological features. Discard the lowest one
        if (np.diff(best_Emax_points.elev)>50):
            best_Emax_points=best_Emax_points.iloc[np.where(best_Emax_points.elev==np.max(best_Emax_points.elev))]
    
        #pdb.set_trace()
        #Display the best Emax points
        ax_map.scatter(best_Emax_points['x'],best_Emax_points['y'],s=50,zorder=200,c=my_pal[str(single_year)],edgecolors='black')#Display the best Emax points #count+1
        ### --------------- Inspired from Emax_Slabs_tickness.py -------------- ###
        
        ### In case yearly map of Emax point are desired to be plotted, comment this ###
        if (len(best_Emax_points)==0):
            count=count+2
            continue
        else:
            #Convert to coordinates into EPSG:4326
            coord_Emax=transformer_3413_to_4326.transform(np.array(best_Emax_points['x']),np.array(best_Emax_points['y']))
            
            #Plot Emax on the correct radargrams
            if (year_transect==2002):
                ax1.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2003):
                ax2.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2010):
                ax3.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2011):
                ax4.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2012):
                ax5.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2013):
                ax6.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2014):
                ax7.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2017):
                ax8.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            elif(year_transect==2018):
                ax9.scatter(coord_Emax[0],np.ones(len(coord_Emax[0]))*2,c=my_pal[str(single_year)])
            else:
                print('Should not end up there')
                pdb.set_trace()
        ### In case yearly map of Emax point are desired to be plotted, comment this ###
    
        '''
        #In case yearly map of Emax point are desired to be plotted
        #Save figure
        plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Section1/CS1/CS1_NDWI_Emax_'+str(single_year)+'.png',dpi=300,bbox_inches='tight')
        #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        '''
        count=count+2
        #pdb.set_trace()
    
#pdb.set_trace() #In case yearly map of Emax point are desired to be plotted, uncomment this

if (investigation_year==CaseStudy1):
    ax4.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax1
    ax_map.set_title('Case study 1')
    ax_top.set_title('Case study 1')

    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-47.5,-47,-48], ylocs=[66.05,66.10], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################
elif (investigation_year==CaseStudy2):
    ax9.set_ylabel('Depth [m]')
    ax9.set_yticklabels(['0','10','20'])
    ticks_through=ax9.get_xticks()
    year_ticks=2018
    ax_tick_plot=ax9
    ax_top=ax2
    '''
    ax_map.set_title('Case study 2')
    ax_top.set_title('Case study 2')
    '''
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-47.5,-47,-46.5], ylocs=[67.60,67.65], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
    #Customize lat labels
    gl.right_labels = False
    gl.top_labels = False
    ax_map.axis('off')
    ###################### From Tedstone et al., 2022 #####################
elif (investigation_year==CaseStudy3):
    ax4.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax1
    ax_map.set_title('Case study 3')
    ax_top.set_title('Case study 3')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-47.5,-47,-46.5], ylocs=[68,68.25], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################
elif (investigation_year==CaseStudy4):
    ax4.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax1
    ax_map.set_title('Case study 4')
    ax_top.set_title('Case study 4')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-66,-67], ylocs=[76.7,76.8], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################

elif (investigation_year==CaseStudy5):
    ax4.set_ylabel('Depth [m]')
    ax5.set_yticklabels(['0','10','20'])
    ticks_through=ax5.get_xticks()
    year_ticks=2012
    ax_tick_plot=ax5
    ax_top=ax2
    ax_map.set_title('Case study 5')
    ax_top.set_title('Case study 5')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-50,-60,-61], ylocs=[79.5,80], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################


elif (investigation_year==CaseStudy6):
    ax7.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax1
    ax_map.set_title('Case study 6')
    ax_top.set_title('Case study 6')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-61,-60], ylocs=[79.6,79.7], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################

elif (investigation_year==CaseStudy7):
    ax3.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax2
    ax_map.set_title('Case study 7')
    ax_top.set_title('Case study 7')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-61,-60], ylocs=[79.6,79.7], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################

elif (investigation_year==CaseStudyFS):
    ax5.set_ylabel('Depth [m]')
    ax8.set_yticklabels(['0','10','20'])
    ticks_through=ax8.get_xticks()
    year_ticks=2017
    ax_tick_plot=ax8
    ax_top=ax2
    ax_map.set_title('Case study FS')
    ax_top.set_title('Case study FS')
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_map.gridlines(draw_labels=True, xlocs=[-46,-47,-48], ylocs=[66.5,67,67.5], x_inline=False, y_inline=False,linewidth=0.5)
    ###################### From Tedstone et al., 2022 #####################

else:
    print('Wrong transect name input')

#Display colorbars. This is from FigS1.py
cbar_depth=fig1.colorbar(cb, cax=axc, aspect=5)#aspect is from https://stackoverflow.com/questions/33443334/how-to-decrease-colorbar-width-in-matplotlib
cbar_depth.set_label('Radar signal strength [dB]')

cbar_CumHydro=fig1.colorbar(cbar, cax=axc_map)
cbar_CumHydro.set_label('Hydrological occurence')
#Modify colorbar ticks according to vmin and vmax
cbar_CumHydro.set_ticks(np.arange(vlim_min,vlim_max,20)+10)
cbar_CumHydro.set_ticklabels(np.arange(vlim_min,vlim_max,20)+10-vlim_min)

plot_dist=[]
for indiv_tick in ticks_through:
    lon_diff=[]
    lon_diff=np.abs(dataframe[str(year_ticks)]['lon_appended']-indiv_tick)
    index_min=np.argmin(lon_diff)
    if (lon_diff[index_min]>0.2):
        plot_dist=np.append(plot_dist,999)
    else:
        plot_dist=np.append(plot_dist,dataframe[str(year_ticks)]['distances'][index_min]/1000-dataframe[str(year_ticks)]['distances'][np.argmin(np.abs(dataframe[str(year_ticks)]['lon_appended']-start_transect))]/1000)


if (investigation_year==CaseStudy2):
    '''
    #Display NDWI
    ### ------------- This is from Greenland_Hydrology_Summary.py ------------- ###
    desired_year=2019
    path_NDWI='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/data/NDWI/'
    #Load NDWI data for display
    MapPlot = rxr.open_rasterio(path_NDWI+'NDWI_p10_'+str(desired_year)+'.vrt',
                                  masked=True).squeeze() #No need to reproject satelite image
    vlim_min=0
    vlim_max=0.6
    cmap_raster='Blues'
    ### ------------- This is from Greenland_Hydrology_Summary.py ------------- ###

    #Extract x and y coordinates of image
    x_coord_MapPlot=np.asarray(MapPlot.x)
    y_coord_MapPlot=np.asarray(MapPlot.y)
    ### ----------------- This is from Emax_Slabs_tickness.py ----------------- ###

    #Extract coordinates ofcumulative raster within Emaxs bounds
    logical_x_coord_within_bounds=np.logical_and(x_coord_MapPlot>=x_min,x_coord_MapPlot<=x_max)
    x_coord_within_bounds=x_coord_MapPlot[logical_x_coord_within_bounds]
    logical_y_coord_within_bounds=np.logical_and(y_coord_MapPlot>=y_min,y_coord_MapPlot<=y_max)
    y_coord_within_bounds=y_coord_MapPlot[logical_y_coord_within_bounds]

    #Define extents based on the bounds
    extent_MapPlot = [np.min(x_coord_within_bounds), np.max(x_coord_within_bounds), np.min(y_coord_within_bounds), np.max(y_coord_within_bounds)]#[west limit, east limit., south limit, north limit]
    #Display image
    cbar_NDWI=ax_NDWI.imshow(MapPlot[logical_y_coord_within_bounds,logical_x_coord_within_bounds], extent=extent_MapPlot, transform=crs, origin='upper', cmap=cmap_raster,vmin=vlim_min,vmax=vlim_max,zorder=0)

    #Set xlims
    ax_NDWI.set_xlim(x_min,x_max)
    ax_NDWI.set_ylim(y_min,y_max)
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_NDWI.gridlines(draw_labels=True, xlocs=[-47.5,-47,-46.5], ylocs=[67.60,67.65], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed')
    #Customize lat labels
    gl.right_labels = False
    gl.top_labels = False
    ax_map.axis('off')
    ###################### From Tedstone et al., 2022 #####################
    
    #Display cbar
    cbar_NDWI=fig1.colorbar(cbar_NDWI, cax=axc_NDWI)
    cbar_NDWI.set_label('NDWI')
    '''
    #Display legend on CumHyro and NDWI map
    legend_elements=[]
    legend_elements.append([Line2D([0], [0], color='black', lw=2, label='Transect')][0])
    #Display legend
    ax_map.legend(handles=legend_elements,loc='lower right',ncol=5,framealpha=1)#from https://stackoverflow.com/questions/42103144/how-to-align-rows-in-matplotlib-legend-with-2-columns
    '''
    ax_NDWI.legend(handles=legend_elements,loc='lower right',ncol=5,framealpha=1)#from https://stackoverflow.com/questions/42103144/how-to-align-rows-in-matplotlib-legend-with-2-columns
    '''
    #Display panel label
    ax2.text(0.01, 0.70,'a',ha='center', va='center', transform=ax2.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    ax3.text(0.01, 0.85,'b',ha='center', va='center', transform=ax3.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    ax7.text(0.01, 0.85,'c',ha='center', va='center', transform=ax7.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    ax9.text(0.01, 0.85,'d',ha='center', va='center', transform=ax9.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    ax_map.text(0.01, 0.925,'e',ha='center', va='center', transform=ax_map.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    '''
    ax_NDWI.text(0.01, 0.925,'g',ha='center', va='center', transform=ax_NDWI.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    '''
    plt.show()

# Display scalebar with GeoPandas
ax_map.add_artist(ScaleBar(1,location='upper right',box_alpha=0,box_color=None))#.set_pad(2)
'''
ax_NDWI.add_artist(ScaleBar(1,location='upper right',box_alpha=0,box_color=None))#.set_pad(2)
'''
#Coordinates of sectors to display
coord_sectors=[#(67.620575, -47.59745),
               #(67.622106, -47.566856),
               (67.626644, -47.414368),
               (67.628561, -47.33543),
               (67.629785, -47.299504),
               (67.632129, -47.232908),
               #(67.632711, -47.216256),
               #(67.633521, -47.183796),
               (67.635528, -47.14),
               (67.636072, -47.09873)]
#Display sections on the map
for indiv_point in coord_sectors:
    #Display on radargrams
    ax2.axvline(indiv_point[1],linestyle='dashed',color='black',linewidth=1)
    ax3.axvline(indiv_point[1],linestyle='dashed',color='black',linewidth=1)
    ax7.axvline(indiv_point[1],linestyle='dashed',color='black',linewidth=1)
    ax9.axvline(indiv_point[1],linestyle='dashed',color='black',linewidth=1)

    #Display on map
    #Transform the coordinates from EPSG:3413 to EPSG:4326
    #Example from: https://pyproj4.github.io/pyproj/stable/examples.html
    points=transformer.transform(indiv_point[1],indiv_point[0])
    ax_map.axvline(points[0],zorder=3,color='black',linestyle='dashed',linewidth=1)
    '''
    ax_NDWI.axvline(points[0],zorder=3,color='black',linestyle='dashed',linewidth=1)
    '''
#Add sector label
ax_map.text(0.39, 0.05,'A',ha='center', va='center', transform=ax_map.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_map.text(0.515, 0.05,'B',ha='center', va='center', transform=ax_map.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_map.text(0.615, 0.05,'C',ha='center', va='center', transform=ax_map.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_map.text(0.685, 0.05,'D',ha='center', va='center', transform=ax_map.transAxes,weight='bold',fontsize=8,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

ax_tick_plot.xaxis.set_ticks_position('bottom') 
ax_tick_plot.set_xticklabels(np.round(plot_dist).astype(int))
ax_tick_plot.set_xlabel('Distance [km]')

'''
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
'''
plt.show()

pdb.set_trace()

#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig6/v3/Fig6abcde.png',dpi=300)#,bbox_inches='tight')

'''
#Save the figure
plt.savefig('C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/Section1/CS2/v2/CS2_NDWI_RadargramsAndEmax_HighestAndClosest_map.png',dpi=300,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''


    
    