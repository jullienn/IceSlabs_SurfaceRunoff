# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:33:47 2023

@author: jullienn
"""

def add_rectangle_zoom(axis_InsetMap,axis_plot):
    #Display rectangle around datalocation - this is from Fig1.py paper Greenland Ice Sheet Ice Slabs Expansion and Thickening  
    #This is from https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
    # Create a Rectangle patch and add the patch to the Axes
    axis_InsetMap.add_patch(patches.Rectangle(([axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][0],[axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][1]),
                                              np.abs([axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][0]-[axis_plot.get_xlim()[1],axis_plot.get_ylim()[1]][0]),
                                              np.abs([axis_plot.get_xlim()[0],axis_plot.get_ylim()[0]][1]-[axis_plot.get_xlim()[1],axis_plot.get_ylim()[1]][1]),
                                              angle=0, linewidth=1, edgecolor='black', facecolor='none'))
    return

def extract_RL_line_from_xytpd(df_xytpd_2012_in_func,df_xytpd_2019_in_func):
    
    display_RL_selection='FALSE'
    
    #Select 2012 xytpd inside the indiv box
    df_xytpd_2012_indiv_box=df_xytpd_2012_in_func[df_xytpd_2012_in_func.box_id==indiv_box_nb].copy()
    #Select 2019 xytpd inside the indiv box
    df_xytpd_2019_indiv_box=df_xytpd_2019_in_func[df_xytpd_2019_in_func.box_id==indiv_box_nb].copy()
    
    #Transform 2012 into a line - this is from Extraction_IceThicknessAndSAR_InTransects.py
    RL_tuple_2012=[tuple(row[['x','y']]) for index, row in df_xytpd_2012_indiv_box.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
    RL_line_2012=LineString(RL_tuple_2012)
    RL_line_2012_gdp=gpd.GeoDataFrame(geometry=[RL_line_2012],crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
    
    #Modify line for some specific case in 2019 - this is heritated from Extract_IceThicknessAndSAR_InSectors.py
    if (indiv_box_nb==7):
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==902,'index_Emax']=948
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==924,'index_Emax']=949
        #sort
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box.sort_values('index_Emax')
        
    if (indiv_box_nb==15):
        #We do not go down to Emax point 829 because the above category will include lower elevation places due to decreasing elevations towards the south
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box[df_xytpd_2019_indiv_box.index_Emax<=388]
    
    if (indiv_box_nb==16):
        #sort
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box.sort_values('index_Emax')
        
    if (indiv_box_nb==19):
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==715,'index_Emax']=550
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==681,'index_Emax']=549
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==748,'index_Emax']=548
        #sort
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box.sort_values('index_Emax',ascending=False)
        
    if (indiv_box_nb==27):
        #Do not consider points in the north east sector of this polygon
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box[df_xytpd_2019_indiv_box.index_Emax<=772]
        #Modify position for box 27 in 2019, from https://stackoverflow.com/questions/40427943/how-do-i-change-a-single-index-value-in-pandas-dataframe
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==637,'index_Emax']=579
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==698,'index_Emax']=580
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==668,'index_Emax']=581
        df_xytpd_2019_indiv_box.loc[df_xytpd_2019_indiv_box.index_Emax==607,'index_Emax']=582
        #sort
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box.sort_values('index_Emax')
    
    if (indiv_box_nb==32):
        #Do not consider points in the north east sector of this polygon
        df_xytpd_2019_indiv_box=df_xytpd_2019_indiv_box[df_xytpd_2019_indiv_box.index_Emax<=2064]
    
    
    #Transform 2019 into a line - this is from Extraction_IceThicknessAndSAR_InTransects.py
    if (indiv_box_nb==25):
        #For box 25, need to divide the box into two independant suite of Emax points
        #Sect 1
        df_xytpd_2019_indiv_box_sect1=df_xytpd_2019_indiv_box[df_xytpd_2019_indiv_box.index_Emax>=237]
        RL_tuple_2019_sect1=[[row['x'],row['y']] for index, row in df_xytpd_2019_indiv_box_sect1.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        #Sect 2
        df_xytpd_2019_indiv_box_sect2=df_xytpd_2019_indiv_box[df_xytpd_2019_indiv_box.index_Emax<=136]
        RL_tuple_2019_sect2=[[row['x'],row['y']] for index, row in df_xytpd_2019_indiv_box_sect2.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        
        #Create a MultiLineString - Depreciation warning with Shapely 2.0, but ok as it works
        RL_line_2019= MultiLineString([RL_tuple_2019_sect1,RL_tuple_2019_sect2])
        
        #Create gpd as an aggregation of the two independant lines
        RL_line_2019_gdp=gpd.GeoDataFrame(geometry=[RL_line_2019],crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
    
    else:
        RL_tuple_2019=[tuple(row[['x','y']]) for index, row in df_xytpd_2019_indiv_box.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        RL_line_2019=LineString(RL_tuple_2019)
        RL_line_2019_gdp=gpd.GeoDataFrame(geometry=[RL_line_2019],crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
        
    
    if (display_RL_selection=='TRUE'):
        #Create map for display
        fig = plt.figure()
        fig.set_size_inches(12, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
        #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
        gs = gridspec.GridSpec(5, 5)
        ax_box = plt.subplot(gs[0:5, 0:5],projection=crs)
        
        #Display coastlines
        ax_box.coastlines(edgecolor='black',linewidth=0.075)
        #Display 2010-2018 high end ice slabs jullien et al., 2023
        iceslabs_20102018_jullienetal2023.plot(ax=ax_box,facecolor='#ba2b2b',edgecolor='#ba2b2b')

        #Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
        GrIS_drainage_bassins.plot(ax=ax_box,facecolor='none',edgecolor='black',zorder=5)
        
        #Set x and y limits
        ax_box.set_xlim(Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb].bounds.minx.values[0], Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb].bounds.maxx.values[0])
        ax_box.set_ylim(Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb].bounds.miny.values[0], Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb].bounds.maxy.values[0])
        
        ###################### From Tedstone et al., 2022 #####################
        #from plot_map_decadal_change.py
        gl=ax_box.gridlines(draw_labels=True, xlocs=[-20,-25,-30,-35,-40,-45,-50,-55,-60,-65,-70,-75], ylocs=[60,62,64,66,68,70,72,74,76,78,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=6)
        #Customize lat labels
        gl.right_labels = False
        gl.bottom_labels = False
        ax_box.axis('off')
        ###################### From Tedstone et al., 2022 #####################
        ax_box.add_artist(ScaleBar(1))
        
        #Display box on map
        Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb].plot(ax=ax_box,color='none',edgecolor='red',zorder=4)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
        
        #Display 2012 and 2019 RL on map
        RL_line_2012_gdp.plot(ax=ax_box,color='red')
        RL_line_2019_gdp.plot(ax=ax_box,color='green')
        
        pdb.set_trace()
        plt.close()

    return RL_line_2012_gdp,RL_line_2019_gdp

     
def create_east_west_polygons(indiv_Boxes_Tedstone2022_in_func,ax_plot):
    #Duplicate box and separate it in two due to different topographic orientations!
    #Extract coords of box 22
    box_22_coords=indiv_Boxes_Tedstone2022_in_func.boundary.iloc[0].coords.xy
    #Construct point to create polygon
    point_northern_boundary_for_west_east_limit = (-625763.534,-1129439.361)#Identified by doing the intersection between a line parallel
                                                                            #to the western boundary that intersets with the northern boundary 
    point_west_east_limit_midway_a = (-490146.993,-1199636.554)#Identified by doing the intersection between a line parallel to the northern
                                                                #boundary that intersets with the line parallel to the western boundray
    point_west_east_limit_midway_b = (-476452.771,-1190689.662)#Identified by doing the intersection between the line parallel to the
                                                                #northern boundary that was used for point_west_east_limit_midway_a that
                                                                #intersets with a line parallel to the western boundary that is further away
                                                                #than the one used for point_northern_boundary_for_west_east_limit.          
    point_southern_boundary_for_west_east_limit = (-223798.139,-1321467.560)#Identified by doing the intersection between the line parallel
                                                                            #to the western boundary used for point_west_east_limit_midway_b
                                                                            #that intersets with the sounthern boundary.
    #Construct west polygon
    box_22_west_in_func = gpd.GeoDataFrame(geometry=[Polygon([(box_22_coords[0][0],box_22_coords[1][0]),
                                                              (box_22_coords[0][1],box_22_coords[1][1]),
                                                              point_northern_boundary_for_west_east_limit,
                                                              point_west_east_limit_midway_a,
                                                              point_west_east_limit_midway_b,
                                                              point_southern_boundary_for_west_east_limit])],crs="EPSG:3413")
    box_22_west_in_func.plot(ax=ax_plot,color='none',edgecolor='blue',zorder=4)
    
    #Construct east polygon
    box_22_east_in_func = gpd.GeoDataFrame(geometry=[Polygon([point_southern_boundary_for_west_east_limit,
                                                              point_west_east_limit_midway_b,
                                                              point_west_east_limit_midway_a,
                                                              point_northern_boundary_for_west_east_limit,
                                                              (box_22_coords[0][2],box_22_coords[1][2]),
                                                              (box_22_coords[0][3],box_22_coords[1][3]),])],crs="EPSG:3413")
    box_22_east_in_func.plot(ax=ax_plot,color='none',edgecolor='blue',zorder=4)

    return box_22_west_in_func,box_22_east_in_func


def create_polygon_inclusion_box21(indiv_Boxes_Tedstone2022_in_func):#,ax_plot):
    #Get rid of ice slabs to the east and south due to prominence with firn aquifers and due to different topographic orientations!
    #Extract coords of box 21
    box_21_coords=indiv_Boxes_Tedstone2022_in_func.boundary.iloc[0].coords.xy
    #Define points where intersections (see methodology below)
    point_boundary_SouthOffset_EastBoundary=(-579206.322,-1357775.425)#Intersection between eastern boundary and the line parallel to southern boundary with a certain offset
    point_boundary_SouthOffset_EastOffset=(-418061.140,-1335093.338)#Intersection between the line parallel to southern boundary with a certain offset and the line parallel to eastern boundary with a certain offset
    
    #Construct exclusion polygon
    box_21_inclusion_polygon = gpd.GeoDataFrame(geometry=[Polygon([(box_21_coords[0][0],box_21_coords[1][0]),
                                                                   point_boundary_SouthOffset_EastOffset,
                                                                   point_boundary_SouthOffset_EastBoundary,
                                                                   (box_21_coords[0][3],box_21_coords[1][3])])],crs="EPSG:3413")
    
    #Construct exclusion polygon
    box_21_exclusion_polygon = gpd.GeoDataFrame(geometry=[Polygon([(box_21_coords[0][1],box_21_coords[1][1]),
                                                                   (box_21_coords[0][2],box_21_coords[1][2]),
                                                                   point_boundary_SouthOffset_EastBoundary,
                                                                   point_boundary_SouthOffset_EastOffset,
                                                                   (box_21_coords[0][0],box_21_coords[1][0])])],crs="EPSG:3413")
        
    #Kept methodoly if similar prodecure is needed later on
    '''
    #Display one polygon edge
    ax_plot.scatter(box_21_coords[0][0],box_21_coords[1][0])
    
    #Reproduce southern boundary
    southern_boundary = gpd.GeoDataFrame(geometry=[LineString([[box_21_coords[0][1],box_21_coords[1][1]], [box_21_coords[0][2],box_21_coords[1][2]]])],crs="EPSG:3413")
    southern_boundary.plot(ax=ax_plot,color='magenta')
    
    #Create parallel offset to sounterhn boundary
    parallel_offset_southB= southern_boundary.translate(xoff=0,yoff=82500)
    parallel_offset_southB.plot(ax=ax_plot,color='magenta')
    
    #Reproduce easter boundary
    eastern_boundary = gpd.GeoDataFrame(geometry=[LineString([[box_21_coords[0][2],box_21_coords[1][2]], [box_21_coords[0][3],box_21_coords[1][3]]])],crs="EPSG:3413")
    eastern_boundary.plot(ax=ax_plot,color='magenta')
    
    #Create parallel offset to easter boundary
    parallel_offset_easternB= eastern_boundary.translate(xoff=160000,yoff=0)
    parallel_offset_easternB.plot(ax=ax_plot,color='magenta')
    
    #Extract intersection between eastern boundary and parallel_offset_southB
    point_boundary_SouthOffset_EastBoundary=eastern_boundary.intersection(parallel_offset_southB)
    ax_plot.scatter(point_boundary_SouthOffset_EastBoundary[0],point_boundary_SouthOffset_EastBoundary[1])
    
    #Extract intersection between parallel_offset_southB and parallel_offset_easternB
    point_boundary_SouthOffset_EastOffset=parallel_offset_easternB.intersection(parallel_offset_southB)
    ax_plot.scatter(point_boundary_SouthOffset_EastOffset[0],point_boundary_SouthOffset_EastOffset[1])
    '''
    return box_21_inclusion_polygon,box_21_exclusion_polygon


def extract_IceSlabs_UpperEnd(iceslabs_boundary_gpd,polygon_for_intersection_gpd_in_func,ax_plot,GrIS_DEM_in_func_extract,box_nb_in_func,slice_id_in_func):
    #Intersect slice with ice slabs boundaries
    intersection_slice_iceslabs_boundary=gpd.clip(iceslabs_boundary_gpd, polygon_for_intersection_gpd_in_func)#inspired thanks to https://gis.stackexchange.com/questions/246782/geopandas-line-polygon-intersection
                
    #If ~(no intersection), display
    if ((~intersection_slice_iceslabs_boundary[~(intersection_slice_iceslabs_boundary==None)].is_empty).astype(int).sum()>0):
        intersection_slice_iceslabs_boundary.plot(ax=ax_plot,color='magenta',zorder=15)
        '''
        intersection_slice_iceslabs_boundary.centroid.plot(ax=ax_plot,color='black')
        '''
        #Fix the CW-SW transition
        if ((box_nb_in_func=='10')&(slice_id_in_func==40)):
            #Create the line
            #First line
            first_line=intersection_slice_iceslabs_boundary[~(intersection_slice_iceslabs_boundary==None)].explode(index_parts=True).iloc[1][0].coords.xy
            point_a=(first_line[0][-1],first_line[1][-1])
            #Second line
            second_line=intersection_slice_iceslabs_boundary[~(intersection_slice_iceslabs_boundary==None)].explode(index_parts=True).iloc[2][0].coords.xy
            point_b=(second_line[0][0],second_line[1][0])
            #Display created line
            ax_plot.plot([point_a[0],point_b[0]],[point_a[1],point_b[1]])
            #Update geometry in intersection_slice_iceslabs_boundary
            intersection_slice_iceslabs_boundary = gpd.GeoDataFrame(geometry=[LineString([point_a,point_b])],crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
            #Create line centroid under the same format
            lines_centroids=intersection_slice_iceslabs_boundary.centroid
        else:
            #Keep only shapefile where there is intersection, and explode the multistring into individual line strings
            lines_centroids = intersection_slice_iceslabs_boundary[~(intersection_slice_iceslabs_boundary==None)].explode(index_parts=True).centroid# explode from https://stackoverflow.com/questions/72525894/convert-each-multilinestring-to-one-linestring-only
            #Reset index to get rid of double index from join operation
            lines_centroids.reset_index(drop=True, inplace=True)#This is from https://stackoverflow.com/questions/20107570/removing-index-column-in-pandas-when-reading-a-csv
            
        #Create a pandas datafarme
        lines_centroids_df = pd.DataFrame({"geometry":lines_centroids,
                                           "elevation":[np.nan]*len(lines_centroids)})
        
        #Loop over the centroids of each line intersection, and keep the line whose centroid is the highest
        for i, row in lines_centroids_df.iterrows(): #iterrows from https://stackoverflow.com/questions/36864690/iterate-through-a-dataframe-by-index
            #Display centroid of each line
            ax_plot.scatter(row[0].x, row[0].y,color='blue',s=100)
            #Extract elevation
            lines_centroids_df.loc[int(i),'elevation']=GrIS_DEM_in_func_extract.value_at_coords(row[0].x, row[0].y)
                
        if ((lines_centroids_df.elevation==-9999).astype(int).sum()>0):
            #pdb.set_trace()
            #If one elevation is -9999, ignore maximum ice slabs retrieval in this slice and return an empty df
            lines_centroids_df_highest=pd.DataFrame()
            print('Ice slabs distance and elevation calculations not possible, continue')
        else:
            #Select the point at the highest elevation
            lines_centroids_df_highest=lines_centroids_df.where(lines_centroids_df.elevation==lines_centroids_df.elevation.max()).copy()
            lines_centroids_df_highest=lines_centroids_df_highest[~(lines_centroids_df_highest.elevation.isna())].copy()
    else:
        #No ice slabs in this slice, return an empty df
        lines_centroids_df_highest=pd.DataFrame()
    
    return lines_centroids_df_highest


def extract_in_boxes(indiv_Boxes_Tedstone2022,poly_2012_in_func,poly_2019_in_func,iceslabs_20102018_jullienetal2023_in_func,iceslabs_20102012_jullienetal2023_in_func,GrIS_DEM_in_func,box_nb):
            
    #Create an overall summary dataframe for this box
    RL_IceSlabs_box = pd.DataFrame()
    
    #Create map for display
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure()
    fig.set_size_inches(12, 10) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    #projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
    gs = gridspec.GridSpec(5, 5)
    ax_box = plt.subplot(gs[0:5, 0:5],projection=crs)
    
    #Display coastlines
    ax_box.coastlines(edgecolor='black',linewidth=0.075)
    #Display 2010-2018 high end ice slabs jullien et al., 2023
    iceslabs_20102018_jullienetal2023_in_func.plot(ax=ax_box,facecolor='#ba2b2b',edgecolor='#ba2b2b')
    #Display 2010-2012 high end ice slabs jullien et al., 2023
    iceslabs_20102012_jullienetal2023_in_func.plot(ax=ax_box,facecolor='#6baed6',edgecolor='#6baed6')
    #Display firn aquifers Miège et al., 2016
    ax_box.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#74c476',s=1,zorder=2)
    
    #Display MVRL
    poly_2012_in_func.plot(ax=ax_box,facecolor='none',edgecolor='#dadaeb',linewidth=1,zorder=3)
    poly_2019_in_func.plot(ax=ax_box,facecolor='none',edgecolor='#54278f',linewidth=1,zorder=3)
    
    #Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
    GrIS_drainage_bassins.plot(ax=ax_box,facecolor='none',edgecolor='black',zorder=5)
    
    #Set x and y limits
    ax_box.set_xlim(indiv_Boxes_Tedstone2022.bounds.minx.values[0], indiv_Boxes_Tedstone2022.bounds.maxx.values[0])
    ax_box.set_ylim(indiv_Boxes_Tedstone2022.bounds.miny.values[0], indiv_Boxes_Tedstone2022.bounds.maxy.values[0])
    
    ###################### From Tedstone et al., 2022 #####################
    #from plot_map_decadal_change.py
    gl=ax_box.gridlines(draw_labels=True, xlocs=[-20,-25,-30,-35,-40,-45,-50,-55,-60,-65,-70,-75], ylocs=[60,62,64,66,68,70,72,74,76,78,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=6)
    #Customize lat labels
    gl.right_labels = False
    gl.bottom_labels = False
    ax_box.axis('off')
    ###################### From Tedstone et al., 2022 #####################
    ax_box.add_artist(ScaleBar(1))

    #Display box on map
    indiv_Boxes_Tedstone2022.plot(ax=ax_box,color='none',edgecolor='red',zorder=4)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
    
    #Extract coordinates of vertice
    xcoords=indiv_Boxes_Tedstone2022.boundary.iloc[0].coords.xy[0]
    ycoords=indiv_Boxes_Tedstone2022.boundary.iloc[0].coords.xy[1]
    
    #Prepare creation of 1-km wide slice
    #Offset for next box
    offset_y_right=0
    offset_y_left=0
    offset_x_right=0
    offset_x_left=0
    
    if (box_nb in list (['21'])):
        #Extract corner coordinates of box
        upper_right=(xcoords[0],ycoords[0])
        lower_right=(xcoords[1],ycoords[1])
        lower_left=(xcoords[2],ycoords[2])
        upper_left=(xcoords[3],ycoords[3])
    elif (box_nb in list (['22_west','22_east'])):
        #Extract corner coordinates of box
        upper_right=(xcoords[2],ycoords[2])
        lower_right=(xcoords[3],ycoords[3])
        lower_left=(xcoords[0],ycoords[0])
        upper_left=(xcoords[1],ycoords[1])
    elif (box_nb in list (['23'])):
        #Extract corner coordinates of box
        upper_right=(xcoords[0],ycoords[0])
        lower_right=(xcoords[1],ycoords[1])
        lower_left=(xcoords[2],ycoords[2])
        upper_left=(xcoords[3],ycoords[3])
    elif (box_nb in list (['27'])):
        #Extract corner coordinates of box
        lower_right=(xcoords[3],ycoords[3])
        lower_left=(xcoords[0],ycoords[0])
        upper_left=(xcoords[1],ycoords[1])
        upper_right=(xcoords[2],ycoords[2])
    elif (box_nb in list (['28'])):
        #Extract corner coordinates of box
        lower_right=(xcoords[1],ycoords[1])
        lower_left=(xcoords[2],ycoords[2])
        upper_left=(xcoords[3],ycoords[3])
        upper_right=(xcoords[0],ycoords[0])
    else:
        #Extract corner coordinates of box
        lower_right=(xcoords[0],ycoords[0])
        lower_left=(xcoords[1],ycoords[1])
        upper_left=(xcoords[2],ycoords[2])
        upper_right=(xcoords[3],ycoords[3])
    
    #Display to make sure correct
    ax_box.scatter(lower_right[0],lower_right[1])
    ax_box.scatter(lower_left[0],lower_left[1])
    ax_box.scatter(upper_left[0],upper_left[1])
    ax_box.scatter(upper_right[0],upper_right[1])
    
    #Define y spacing
    nb_boxes_right=np.round(np.abs((lower_right[1]-upper_right[1]))/1000).astype(int)#approximately 1-km wide boxes
    spacing_y_right=np.mean(np.diff(np.linspace(lower_right[1],upper_right[1],nb_boxes_right)))
    spacing_y_left=np.mean(np.diff(np.linspace(lower_left[1],upper_left[1],nb_boxes_right)))
    
    #Define x spacing
    spacing_x_right=(upper_right[0]-lower_right[0])/nb_boxes_right
    spacing_x_left=(upper_left[0]-lower_left[0])/nb_boxes_right
        
    #Box 21 is a particular case!
    if (box_nb == '21'):
        box_21_inclusion,box_21_exclusion = create_polygon_inclusion_box21(indiv_Boxes_Tedstone2022)
        #Display exclusion polygon
        box_21_exclusion.plot(ax=ax_box,color='grey',edgecolor='grey')
        #Display inclusion polygon
        box_21_inclusion.plot(ax=ax_box,color='none',edgecolor='blue',zorder=4)
    
    #Box 22 is a particular case!
    if (box_nb in list (['22_west','22_east'])):
        box_22_west,box_22_east = create_east_west_polygons(indiv_Boxes_Tedstone2022,ax_box)

    
    #Define conditions for while
    if (box_nb in list (['23','26','27','28','29','30','31'])):
        condition_while=(((lower_right[0]+offset_x_right)<upper_right[0])&((lower_left[0]+offset_x_left)<upper_left[0]))
    else:
        condition_while=(((lower_right[1]+offset_y_right)<upper_right[1])&((lower_left[1]+offset_y_left)<upper_left[1]))
    
    #Set slice id to 0
    slice_id=0
    
    #Loop over each slice
    while condition_while:
        print('Slice_id:',slice_id)
        
        #Create a temporary summary dataframe
        RL_IceSlabs= pd.DataFrame({'slice_id' : [np.nan],
                                   'box_id' : [np.nan],
                                   'Point_2012_RL' : [np.nan],
                                   'Point_2019_RL' : [np.nan],
                                   'Point_IceSlabsBoundary20102012' : [np.nan],
                                   'Point_IceSlabsBoundary20102018' : [np.nan],
                                   'Elevation_2012_RL' : [np.nan],
                                   'Elevation_2019_RL' : [np.nan],
                                   'Elevation_IceSlabsBoundary20102012' : [np.nan],
                                   'Elevation_IceSlabsBoundary20102018' : [np.nan],
                                   'IceSlabs20102012_Region' : [np.nan],
                                   'IceSlabs20102018_Region' : [np.nan],
                                   'RL2012Region' : [np.nan],
                                   'RL2019Region' : [np.nan],
                                   'Distance_IceSlabsBoundary20102012_2012_RL' : [np.nan],
                                   'Distance_IceSlabsBoundary20102018_2019_RL' : [np.nan],
                                   'Distance_2012_2019_RL' : [np.nan]},
                                   index=[0])
        
        #Store slice id
        RL_IceSlabs['slice_id']=int(slice_id)
        
        #1. Create slices using boxes from Tedstone and Machguth (2022)
        #Helper from this: https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
        
        #Create 1-km wide at high elevations polygon. With this method, there is an offset of ~10m the upper left corner, and ~200m on the upper right corner
        polygon_for_intersection = gpd.GeoSeries(Polygon([(lower_right[0]+offset_x_right,lower_right[1]+offset_y_right),
                                                          (lower_left[0]+offset_x_left,lower_left[1]+offset_y_left),
                                                          (lower_left[0]+offset_x_left+spacing_x_left,lower_left[1]+offset_y_left+spacing_y_left),
                                                          (lower_right[0]+offset_x_right+spacing_x_right,lower_right[1]+offset_y_right+spacing_y_right)]),crs="EPSG:3413")
        
        #Make polygon_for_intersection as a gpd
        polygon_for_intersection_gpd = gpd.GeoDataFrame(pd.DataFrame({"slide_id": [slice_id]}), geometry=polygon_for_intersection, crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
        
        #Display polygon_for_intersection
        polygon_for_intersection_gpd.plot(ax=ax_box,color='green',edgecolor='black',alpha=0.5)
        
        #For box 21, clip slice with inclusion polygon to keep inside inclusion
        if (box_nb == '21'):
            #Perform clip
            polygon_for_intersection_gpd=gpd.overlay(polygon_for_intersection_gpd,box_21_inclusion,how='intersection')
            if (polygon_for_intersection_gpd.empty == True):
                #No intersection, go to next slice
                
                #Add offsets
                offset_x_right=offset_x_right+spacing_x_right
                offset_x_left=offset_x_left+spacing_x_left
                offset_y_left=offset_y_left+spacing_y_left
                offset_y_right=offset_y_right+spacing_y_right
                               
                #Update condition for while
                condition_while=((np.round((lower_right[1]+offset_y_right),6)<np.round(upper_right[1],6))&(np.round((lower_left[1]+offset_y_left),6)<np.round(upper_left[1],6)))
                
                #Update slice id
                slice_id=slice_id+1
                
                continue
        
        #For box 22, clip slice with west or east
        if (box_nb == '22_west'):
            polygon_for_intersection_gpd=gpd.clip(polygon_for_intersection_gpd,box_22_west)
        if (box_nb == '22_east'):
            polygon_for_intersection_gpd=gpd.clip(polygon_for_intersection_gpd,box_22_east)
        
        #Display polygon_for_intersection
        polygon_for_intersection_gpd.plot(ax=ax_box,color='yellow',edgecolor='black',alpha=0.5)
        
        #2. In each slice, extract RL line and ice slabs
        #Intersect with runoff limit line
        intersection_slice_2012_RL=polygon_for_intersection_gpd.intersection(poly_2012_in_func)
        intersection_slice_2019_RL=polygon_for_intersection_gpd.intersection(poly_2019_in_func)
        
        #3. In each slice, extract cendroid coordinates of each line
        #If there is 2012 RL in this slice
        if (~intersection_slice_2012_RL.is_empty[0]):
            #Display extracted 2012 RL line
            intersection_slice_2012_RL.plot(ax=ax_box,color='green',zorder=20)
            #Display its centroid
            intersection_slice_2012_RL.centroid.plot(ax=ax_box)
            #Store RL data into RL_IceSlabs dataframe
            RL_IceSlabs['Point_2012_RL']=intersection_slice_2012_RL.centroid
            #Extract elevation of RL centroids
            RL_IceSlabs['Elevation_2012_RL']=GrIS_DEM_in_func.value_at_coords(intersection_slice_2012_RL.centroid[0].x, intersection_slice_2012_RL.centroid[0].y)

        #If there is 2019 RL in this slice
        if (~intersection_slice_2019_RL.is_empty[0]):
            #Display extracted 2019 RL line
            intersection_slice_2019_RL.plot(ax=ax_box,color='cyan',zorder=20)
            #Display its centroid
            intersection_slice_2019_RL.centroid.plot(ax=ax_box)
            #Store RL data into RL_IceSlabs dataframe
            RL_IceSlabs['Point_2019_RL']=intersection_slice_2019_RL.centroid
            #Extract elevation of RL centroids
            RL_IceSlabs['Elevation_2019_RL']=GrIS_DEM_in_func.value_at_coords(intersection_slice_2019_RL.centroid[0].x, intersection_slice_2019_RL.centroid[0].y)
        
        #If both 2012 RL and 2019 RL exist in this slice
        if ((~(intersection_slice_2019_RL.is_empty[0]))&(~(intersection_slice_2019_RL.is_empty[0]))):
            #Calculate distance between 2012 and 2019 runoff limit
            RL_IceSlabs['Distance_2012_2019_RL'] = intersection_slice_2012_RL.centroid[0].distance(intersection_slice_2019_RL.centroid[0])
        
        ### --------- Extract 2010-2018 ice slabs upper end point --------- ###
        #Make iceslabs_20102018_jullienetal2023_in_func.boundary as a gpd
        iceslabs_20102018_jullienetal2023_boundary_gpd = gpd.GeoDataFrame(geometry=iceslabs_20102018_jullienetal2023_in_func.boundary,
                                                                          crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
        
        #Extract highest 2010-2018 ice slabs point in the current slice
        highest_20102018_IceSlabs_point = extract_IceSlabs_UpperEnd(iceslabs_20102018_jullienetal2023_boundary_gpd,polygon_for_intersection_gpd,ax_box,GrIS_DEM_in_func,box_nb,slice_id)
                
        if (len(highest_20102018_IceSlabs_point)>0):
            #Store and display the kept centroid of intersection betwwen ice slabs and slice
            RL_IceSlabs['Point_IceSlabsBoundary20102018']=highest_20102018_IceSlabs_point.geometry.iloc[0]
            ax_box.scatter(highest_20102018_IceSlabs_point.geometry.iloc[0].x,highest_20102018_IceSlabs_point.geometry.iloc[0].y,color='red',s=50)
            
            #Extract elevation
            RL_IceSlabs['Elevation_IceSlabsBoundary20102018']=highest_20102018_IceSlabs_point.elevation.iloc[0]
            
            #4. Calculate distances between centroids
            if (~intersection_slice_2019_RL.is_empty[0]):
                RL_IceSlabs['Distance_IceSlabsBoundary20102018_2019_RL'] = highest_20102018_IceSlabs_point.geometry.iloc[0].distance(intersection_slice_2019_RL.centroid[0])
        ### --------- Extract 2010-2018 ice slabs upper end point --------- ###

        ### --------- Extract 2010-2012 ice slabs upper end point --------- ###
        #Make iceslabs_20102012_jullienetal2023_in_func.boundary as a gpd
        iceslabs_20102012_jullienetal2023_boundary_gpd = gpd.GeoDataFrame(geometry=iceslabs_20102012_jullienetal2023_in_func.boundary,
                                                                          crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
        #Extract highest 2010-2012 ice slabs point in the curren slice
        highest_20102012_IceSlabs_point = extract_IceSlabs_UpperEnd(iceslabs_20102012_jullienetal2023_boundary_gpd,polygon_for_intersection_gpd,ax_box,GrIS_DEM_in_func,box_nb,slice_id)
        
        if (len(highest_20102012_IceSlabs_point)>0):
            #Store and display the kept centroid of intersection betwwen ice slabs and slice
            RL_IceSlabs['Point_IceSlabsBoundary20102012']=highest_20102012_IceSlabs_point.geometry.iloc[0]
            ax_box.scatter(highest_20102012_IceSlabs_point.geometry.iloc[0].x,highest_20102012_IceSlabs_point.geometry.iloc[0].y,color='orange',s=50)
            
            #Extract elevation
            RL_IceSlabs['Elevation_IceSlabsBoundary20102012']=highest_20102012_IceSlabs_point.elevation.iloc[0]
            
            #4. Calculate distances between centroids
            if (~intersection_slice_2012_RL.is_empty[0]):
                RL_IceSlabs['Distance_IceSlabsBoundary20102012_2012_RL'] = highest_20102012_IceSlabs_point.geometry.iloc[0].distance(intersection_slice_2012_RL.centroid[0])
        ### --------- Extract 2010-2012 ice slabs upper end point --------- ###
        
        ### Where ice slabs extent regional transition (CW VS SW), might need to include some filtering to get rid of falsely too high ice slabs extent at the low end? It should not be the case, but check that!
        
        #Concat to have a summary dataframe
        RL_IceSlabs_box=pd.concat([RL_IceSlabs_box,RL_IceSlabs])
        
        #Add offsets
        offset_x_right=offset_x_right+spacing_x_right
        offset_x_left=offset_x_left+spacing_x_left
        offset_y_left=offset_y_left+spacing_y_left
        offset_y_right=offset_y_right+spacing_y_right
                       
        #Update condition for while
        if (box_nb in list (['23','26','27','28','29','30','31'])):
            condition_while=((np.round((lower_right[0]+offset_x_right),6)<np.round(upper_right[0],6))&(np.round((lower_left[0]+offset_x_left),6)<np.round(upper_left[0],6)))
        else:
            condition_while=((np.round((lower_right[1]+offset_y_right),6)<np.round(upper_right[1],6))&(np.round((lower_left[1]+offset_y_left),6)<np.round(upper_left[1],6)))
        
        #Update slice id
        slice_id=slice_id+1
    
    pdb.set_trace()
    #Save figure
    plt.savefig(path_switchdrive+'RT3/figures/Fig1/IceSlabs_and_RL_extraction/ExtractionSlabs_and_RL_box_'+box_nb+'_cleanedxytpdV3_final.png',dpi=500,bbox_inches='tight')
    #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    plt.close()
    
    ### Save data as csv files ###
    # The should be no index other than 0
    if ((len(RL_IceSlabs_box.index.unique())>1)&(RL_IceSlabs_box.index.unique()[0]!=0)):
        print('Issue with pandas indexing, STOP!')
        pdb.set_trace()
    
    #Store the box id
    if (box_nb == '22_west'):
        box_save='2201'
    elif (box_nb == '22_east'):
        box_save='2202'
    else:
        box_save=box_nb
    
    #Save box_id
    RL_IceSlabs_box['box_id'] = [int(box_save)]*len(RL_IceSlabs_box)
    #Reindex RL_IceSlabs_summary
    RL_IceSlabs_box.slice_id=RL_IceSlabs_box.slice_id.astype(int)
    RL_IceSlabs_box=RL_IceSlabs_box.set_index("slice_id",drop=True)
        
    ### --- Extract region for ice slabs and RL independantly from each other --- ###
    #1. Extract ice slabs region  
    if (box_nb[0:2]=='22'):
        #Special case for box 22
        if (box_nb=='22_west'):
            region_to_store='NW'
        elif (box_nb=='22_east'):
            region_to_store='NO'
        else:
            print('Region not recognized, STOP')
            pdb.set_trace()
        
        ### Store 2010-2012 ice slabs region ###
        #Store index
        index_to_fill=RL_IceSlabs_box[~(RL_IceSlabs_box['Elevation_IceSlabsBoundary20102012'].isna())].index.to_numpy()
        #Create df
        temp_region=pd.DataFrame(data={'SUBREGION1':[region_to_store]*len(index_to_fill)}, index=index_to_fill)
        #Store into main df
        RL_IceSlabs_box['IceSlabs20102012_Region']=temp_region.SUBREGION1
        ### Store 2010-2012 ice slabs region ###

        ### Store 2010-2018 ice slabs region ###
        #Store index
        index_to_fill=RL_IceSlabs_box[~(RL_IceSlabs_box['Elevation_IceSlabsBoundary20102018'].isna())].index.to_numpy()
        #Create df
        temp_region=pd.DataFrame(data={'SUBREGION1':[region_to_store]*len(index_to_fill)}, index=index_to_fill)
        #Store into main df
        RL_IceSlabs_box['IceSlabs20102018_Region']=temp_region.SUBREGION1
        ### Store 2010-2018 ice slabs region ###
        
    else:
        #Identify 2010-2012 ice slabs region
        RL_IceSlabs_box_IceSlabss20102012Points=RL_IceSlabs_box.copy()
        RL_IceSlabs_box_IceSlabss20102012Points=gpd.GeoDataFrame(RL_IceSlabs_box_IceSlabss20102012Points, geometry=RL_IceSlabs_box_IceSlabss20102012Points.Point_IceSlabsBoundary20102012, crs="EPSG:3413")#Transform into geopandas dataframe, and intersection with GrIS_drainage_bassins which is from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        RL_IceSlabs_box_IceSlabs20102012Points_Regions = gpd.sjoin(RL_IceSlabs_box_IceSlabss20102012Points, GrIS_drainage_bassins, predicate='within')   
        RL_IceSlabs_box['IceSlabs20102012_Region']=RL_IceSlabs_box_IceSlabs20102012Points_Regions.SUBREGION1
        
        #Identify 2010-2018 ice slabs region
        RL_IceSlabs_box_IceSlabss20102018Points=RL_IceSlabs_box.copy()
        RL_IceSlabs_box_IceSlabss20102018Points=gpd.GeoDataFrame(RL_IceSlabs_box_IceSlabss20102018Points, geometry=RL_IceSlabs_box_IceSlabss20102018Points.Point_IceSlabsBoundary20102018, crs="EPSG:3413")#Transform into geopandas dataframe, and intersection with GrIS_drainage_bassins which is from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon
        RL_IceSlabs_box_IceSlabs20102018Points_Regions = gpd.sjoin(RL_IceSlabs_box_IceSlabss20102018Points, GrIS_drainage_bassins, predicate='within')   
        RL_IceSlabs_box['IceSlabs20102018_Region']=RL_IceSlabs_box_IceSlabs20102018Points_Regions.SUBREGION1
    
    #2. Extract 2012 RL region
    RL_IceSlabs_box_2012RLPoints=RL_IceSlabs_box.copy()
    RL_IceSlabs_box_2012RLPoints=gpd.GeoDataFrame(RL_IceSlabs_box_2012RLPoints, geometry=RL_IceSlabs_box_2012RLPoints.Point_2012_RL, crs="EPSG:3413")
    RL_IceSlabs_box_2012RLPoints_Regions = gpd.sjoin(RL_IceSlabs_box_2012RLPoints, GrIS_drainage_bassins, predicate='within')   
    RL_IceSlabs_box['RL2012Region']=RL_IceSlabs_box_2012RLPoints_Regions.SUBREGION1
    
    #3. Extract 2019 RL region
    RL_IceSlabs_box_2019RLPoints=RL_IceSlabs_box.copy()
    RL_IceSlabs_box_2019RLPoints=gpd.GeoDataFrame(RL_IceSlabs_box_2019RLPoints, geometry=RL_IceSlabs_box_2019RLPoints.Point_2019_RL, crs="EPSG:3413")
    RL_IceSlabs_box_2019RLPoints_Regions = gpd.sjoin(RL_IceSlabs_box_2019RLPoints, GrIS_drainage_bassins, predicate='within')   
    RL_IceSlabs_box['RL2019Region']=RL_IceSlabs_box_2019RLPoints_Regions.SUBREGION1    
    ### --- Extract region for ice slabs and RL independantly from each other --- ###
    
    #Export RL_IceSlabs_box into a single csv file
    RL_IceSlabs_box.to_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/whole/RL_IceSlabs_box_'+box_save+'_cleanedxytpdV3.csv')
    ### Save data as csv files ###
    
    return RL_IceSlabs_box


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
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import geoutils as gu
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as patches

extract_data_in_boxes='FALSE'

#Define paths
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'
path_rignotetal2016_GrIS=path_switchdrive+'backup_Aglaja/working_environment/greenland_topo_data/'
path_jullienetal2023=path_switchdrive+'RT1/final_dataset_2002_2018/'

path_data=path_switchdrive+'RT3/data/'
path_local='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/'

#Define palette for time , this is From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'SW': "#9e9ac8", 'CW': "#9e9ac8", 'NW': "#9e9ac8", 'NO': "#9e9ac8", 'NE': "#9e9ac8"}
pal_year= {2012 : "#6baed6", 2019 : "#fcbba1"}

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

#Load GrIS DEM (load in-memory inplicitly)
GrIS_DEM = gu.Raster(path_rignotetal2016_GrIS+'elevations/greenland_dem_mosaic_100m_v3.0.tif')

#load 2012-2018 ice slabs high end from Jullien et al., (2023)
iceslabs_20102012_jullienetal2023=gpd.read_file(path_jullienetal2023+'/shapefiles/iceslabs_jullien_highend_2010_11_12.shp')
'''
#load 2010-2018 ice slabs high end from Jullien et al., (2023)
iceslabs_20102018_jullienetal2023=gpd.read_file(path_jullienetal2023+'/shapefiles/iceslabs_jullien_highend_20102018.shp')
'''
#load 2017-2018 ice slabs high end but name it as iceslabs_20102018_jullienetal2023 to avoid changes in variable names
iceslabs_20102018_jullienetal2023=gpd.read_file('C:/Users/jullienn/switchdrive/Private/research/RT3/data/IceSlabsExtentHighEndJullien_20172018/iceslabs_jullien_highend_20172018.shp')

'''### REV1
#Load MVRL in 2012, 2012, 2016, 2019
poly_2010=gpd.read_file('X:/RT3_jullien/TedstoneAndMachguth2022/runoff_limit_polys/poly_2010.shp')
poly_2012=gpd.read_file('X:/RT3_jullien/TedstoneAndMachguth2022/runoff_limit_polys/poly_2012.shp')
poly_2016=gpd.read_file('X:/RT3_jullien/TedstoneAndMachguth2022/runoff_limit_polys/poly_2016.shp')
poly_2019=gpd.read_file('X:/RT3_jullien/TedstoneAndMachguth2022/runoff_limit_polys/poly_2019.shp')
REV1 ###'''
### -------------------------- Load shapefiles --------------------------- ###

### ---------------------------- Load xytpd ------------------------------ ###
'''
df_xytpd_all=pd.read_csv(path_switchdrive+'RT3/data/Emax/xytpd.csv',delimiter=',',decimal='.')
'''
df_xytpd_all=pd.read_csv(path_switchdrive+'RT3/data/Emax/xytpd_NDWI_cleaned_2019_v3.csv',delimiter=',',decimal='.')
### ---------------------------- Load xytpd ------------------------------ ###

'''### REV1
### -------------------------- Load CumHydro ----------------------------- ###
#Open and display satelite image behind map - This is from Fig4andS6andS7.py from paper 'Greenland Ice slabs Expansion and Thicknening' 
#This section of displaying sat data was coding using tips from
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-raster-data-python/raster-data-processing/reproject-raster/
#https://towardsdatascience.com/visualizing-satellite-data-using-matplotlib-and-cartopy-8274acb07b84
CumHydro = rxr.open_rasterio('X:/RT3_jullien/TedstoneAndMachguth2022/master_maps/'+'master_map_GrIS_mean.vrt',
                             masked=True).squeeze() #No need to reproject satelite image
#Extract x and y coordinates of satellite image
x_coord_CumHydro=np.asarray(CumHydro.x)
y_coord_CumHydro=np.asarray(CumHydro.y)
### -------------------------- Load CumHydro ----------------------------- ###
REV1 ###'''

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
nogo_polygon=np.concatenate((np.arange(1,3+1),np.arange(20,20+1),np.arange(32,53+1)))

###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################

### ----------- Ice slabs upper end and RL 2012/2019 extraction ----------- ###
#Transform xytpd dataframe into geopandas dataframe for distance calculation
df_xytpd_all_gpd = gpd.GeoDataFrame(df_xytpd_all, geometry=gpd.GeoSeries.from_xy(df_xytpd_all['x'], df_xytpd_all['y'], crs="EPSG:3413"))
#Intersection between df_xytpd_all_gpd and GrIS drainage bassins, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
df_xytpd_all_gpd = gpd.sjoin(df_xytpd_all_gpd, GrIS_drainage_bassins, predicate='within')
#Drop index_right
df_xytpd_all_gpd=df_xytpd_all_gpd.drop(columns=["index_right"])
#Rename index to index_Emax (this is heritated from Extract_IceThicknessAndSAR_InSectors.py)
df_xytpd_all_gpd=df_xytpd_all_gpd.rename(columns={"index":"index_Emax"})

#Calculate the difference in elevation between xytpd in 2012 VS 2019 in each slice_id
df_xytpd_2012=df_xytpd_all_gpd[df_xytpd_all_gpd.year==2012].copy()
df_xytpd_2019=df_xytpd_all_gpd[df_xytpd_all_gpd.year==2019].copy()

#pdb.set_trace()

if (extract_data_in_boxes == 'TRUE'):
    #Generate dataset
    RL_IceSlabs_summary=pd.DataFrame()

    for indiv_box_nb in Boxes_Tedstone2022[~Boxes_Tedstone2022.FID.isin(nogo_polygon)].FID:
        print(indiv_box_nb)
        '''
        if (indiv_box_nb == 5):
            pdb.set_trace()
        else:
            continue
        '''
        #Extract the RL lines
        RL_line_2012_indiv_box,RL_line_2019_indiv_box = extract_RL_line_from_xytpd(df_xytpd_2012,df_xytpd_2019)
                
        #Store 2012 data as points coordinates rather than line to save
        RL2012_coordinates=pd.DataFrame()
        RL2012_coordinates['x']=np.array(RL_line_2012_indiv_box.geometry.iloc[0].coords.xy[0])
        RL2012_coordinates['y']=np.array(RL_line_2012_indiv_box.geometry.iloc[0].coords.xy[1])
        
        #Save the 2012 RL line
        RL2012_coordinates.to_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/RL_2012/RL_line_2012_box_'+str(indiv_box_nb)+'_cleanedxytpdV3.csv')
        
        #Store 2019 data as points coordinates rather than line to save
        if (indiv_box_nb==25):
            #Box 25 in 2019 has a multistring, explode multistring into line strings and save
            RL_line_2019_indiv_box_exploded = RL_line_2019_indiv_box.explode(index_parts=False)
            
            #Initialise count for data save
            count=0
            for indiv_line in RL_line_2019_indiv_box_exploded.geometry:            
                #Store data as points coordinates rather than line to save
                RL2019_coordinates=pd.DataFrame()
                RL2019_coordinates['x']=np.array(indiv_line.coords.xy[0])
                RL2019_coordinates['y']=np.array(indiv_line.coords.xy[1])
                
                #Save the RL lines
                RL2019_coordinates.to_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/RL_2019/RL_line_2019_box_'+str(indiv_box_nb)+'_'+str(count)+'_cleanedxytpdV3.csv')
                #update count
                count=count+1
        else:
            RL2019_coordinates=pd.DataFrame()
            RL2019_coordinates['x']=np.array(RL_line_2019_indiv_box.geometry.iloc[0].coords.xy[0])
            RL2019_coordinates['y']=np.array(RL_line_2019_indiv_box.geometry.iloc[0].coords.xy[1])
            
            #Save the RL lines
            RL2019_coordinates.to_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/RL_2019/RL_line_2019_box_'+str(indiv_box_nb)+'_cleanedxytpdV3.csv')
            
        #Perform extraction: generate figure and datasets
        if (indiv_box_nb==22):            
            #Perform analysis in the west side of the box
            RL_IceSlabs_summary_single_box=extract_in_boxes(Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb],RL_line_2012_indiv_box,RL_line_2019_indiv_box,iceslabs_20102018_jullienetal2023,iceslabs_20102012_jullienetal2023,GrIS_DEM,'22_west')
            #Perform analysis in the east side of the box
            RL_IceSlabs_summary_single_box=extract_in_boxes(Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb],RL_line_2012_indiv_box,RL_line_2019_indiv_box,iceslabs_20102018_jullienetal2023,iceslabs_20102012_jullienetal2023,GrIS_DEM,'22_east')
        else:
            RL_IceSlabs_summary_single_box=extract_in_boxes(Boxes_Tedstone2022[Boxes_Tedstone2022.FID==indiv_box_nb],RL_line_2012_indiv_box,RL_line_2019_indiv_box,iceslabs_20102018_jullienetal2023,iceslabs_20102012_jullienetal2023,GrIS_DEM,str(indiv_box_nb))
        
        #Concatenate data
        RL_IceSlabs_summary=pd.concat([RL_IceSlabs_summary,RL_IceSlabs_summary_single_box])
else:
    #Data already generated, open and display plot
    print('Dataset already generated, load data and display on figure')
        
    #Create empty dataframe
    RL_2012=pd.DataFrame()
    RL_2019=pd.DataFrame()
    RL_lines_2012_gdp=pd.DataFrame()
    RL_lines_2019_gdp=pd.DataFrame()
    RL_IceSlabs_gdp=pd.DataFrame()
        
    for indiv_box_nb in Boxes_Tedstone2022[~Boxes_Tedstone2022.FID.isin(nogo_polygon)].FID:        
        if (indiv_box_nb==15):
            #Do not load data from box 15 because cannot reliably assess to which drainage area the runoff limit retrievals are from
            print('Do not consider data in box 15, continue')
            continue
        
        print(indiv_box_nb)
        #Load RL and IceSlabs distance and elevations extraction datasets
        if (indiv_box_nb==22):
            #2 files to load
            indiv_RL_IceSlabs_2201 = pd.read_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/whole/RL_IceSlabs_box_'+str(indiv_box_nb)+'01_cleanedxytpdV3.csv')
            indiv_RL_IceSlabs_2202 = pd.read_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/whole/RL_IceSlabs_box_'+str(indiv_box_nb)+'02_cleanedxytpdV3.csv')
            #Concatenate data
            RL_IceSlabs_gdp=pd.concat([RL_IceSlabs_gdp,indiv_RL_IceSlabs_2201,indiv_RL_IceSlabs_2202])            
        else:
            indiv_RL_IceSlabs = pd.read_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/whole/RL_IceSlabs_box_'+str(indiv_box_nb)+'_cleanedxytpdV3.csv')
            #Concatenate data
            RL_IceSlabs_gdp=pd.concat([RL_IceSlabs_gdp,indiv_RL_IceSlabs])
        
        
        #Load 2012 runoff limit lines
        indiv_2012_RL = pd.read_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/RL_2012/RL_line_2012_box_'+str(indiv_box_nb)+'_cleanedxytpdV3.csv')
        #Add box number to the 2012 dataframe
        indiv_2012_RL['box_nb']=indiv_box_nb
        #Concatenate 2012 data
        RL_2012=pd.concat([RL_2012,indiv_2012_RL])
        
        #Load 2019 runoff limit lines
        if (indiv_box_nb == 25):            
            indiv_2019_RL=pd.DataFrame()
            #There are two lines to load in this box
            for i in range(0,2):
                #Load runoff limit lines
                indiv_2019_RL_temp = pd.read_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/RL_2019/RL_line_2019_box_'+str(indiv_box_nb)+'_'+str(i)+'_cleanedxytpdV3.csv')
                #Add box number to the dataframe
                indiv_2019_RL_temp['box_nb']=float(str(indiv_box_nb)+str(i))/10
                #Concatenate
                indiv_2019_RL=pd.concat([indiv_2019_RL,indiv_2019_RL_temp])
        else:
            #Load 2019 runoff limit lines
            indiv_2019_RL = pd.read_csv(path_switchdrive+'RT3/data/outputs/IceSlabs_and_RL_extraction/RL_2019/RL_line_2019_box_'+str(indiv_box_nb)+'_cleanedxytpdV3.csv')
            #Add box number to the 2019 dataframe
            indiv_2019_RL['box_nb']=indiv_box_nb

        #Concatenate 2019 data
        RL_2019=pd.concat([RL_2019,indiv_2019_RL])
                
        #Create LineStrings        
        #Transform 2012 into a line - this is from Extraction_IceThicknessAndSAR_InTransects.py
        RL_tuple_2012=[tuple(row[['x','y']]) for index, row in indiv_2012_RL.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        RL_line_2012=LineString(RL_tuple_2012)
        RL_line_2012_gdp=gpd.GeoDataFrame(pd.DataFrame({'box_nb':[indiv_box_nb]}),geometry=[RL_line_2012],crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
        RL_lines_2012_gdp=pd.concat([RL_lines_2012_gdp,RL_line_2012_gdp])
        #Transform 2019 into a line - this is from Extraction_IceThicknessAndSAR_InTransects.py
        RL_tuple_2019=[tuple(row[['x','y']]) for index, row in indiv_2019_RL.iterrows()]#from https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/ and https://stackoverflow.com/questions/37515659/returning-a-list-of-x-and-y-coordinate-tuples
        RL_line_2019=LineString(RL_tuple_2019)
        RL_line_2019_gdp=gpd.GeoDataFrame(pd.DataFrame({'box_nb':[indiv_box_nb]}),geometry=[RL_line_2019],crs="EPSG:3413")#This is from https://gis.stackexchange.com/questions/294206/%d0%a1reating-polygon-from-coordinates-in-geopandas
        RL_lines_2019_gdp=pd.concat([RL_lines_2019_gdp,RL_line_2019_gdp])
        
    #Sort by boxes
    RL_2012.sort_values(by=['box_nb','Unnamed: 0'],inplace=True)
    RL_2019.sort_values(by=['box_nb','Unnamed: 0'],inplace=True)

    #Set index in the geopandas dataframes as being the box number
    RL_lines_2012_gdp.set_index("box_nb",inplace=True)
    RL_lines_2019_gdp.set_index("box_nb",inplace=True)

    #Reset index in the dataframes
    RL_2012.reset_index(inplace=True)
    RL_2019.reset_index(inplace=True)
    
    #Drop index and Unnamed colums
    RL_2012.drop(columns=['index','Unnamed: 0'],inplace=True)
    RL_2019.drop(columns=['index','Unnamed: 0'],inplace=True)
    
### ----------- Ice slabs upper end and RL 2012/2019 extraction ----------- ###

#Prepare plot
#Set fontsize plot
plt.rcParams.update({'font.size': 8})
fig = plt.figure()
fig.set_size_inches(7.09, 10.15) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
gs = gridspec.GridSpec(60, 12)
gs.update(hspace=1)
gs.update(wspace=0.1)
ax_north = plt.subplot(gs[0:12, 0:12],projection=crs)
ax_NW = plt.subplot(gs[12:48, 0:4],projection=crs)
ax_CW = plt.subplot(gs[12:48, 4:8],projection=crs)
ax_SW = plt.subplot(gs[12:48, 8:12],projection=crs)
ax_colorbar = plt.subplot(gs[48:49, 0:8])
ax_inset_map = plt.subplot(gs[48:60, 8:12],projection=crs)
axsummary_elev = plt.subplot(gs[53:60, 1:8])

#Draw plot of GrIS map
#Load and display Greenland coast shapefile
GreenlandCoast=gpd.read_file('C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/Greenland_coast/Greenland_coast.shp') 
GreenlandCoast.plot(ax=ax_inset_map,color='#CEB481', edgecolor='grey',linewidth=0.1)
GrIS_drainage_bassins.plot(ax=ax_inset_map,color='white', edgecolor='black',linewidth=0.075)

#Display region name
ax_inset_map.text(NO_rignotetal.centroid.x-150000,NO_rignotetal.centroid.y-100000,np.asarray(NO_rignotetal.SUBREGION1)[0])
ax_inset_map.text(NE_rignotetal.centroid.x-200000,NE_rignotetal.centroid.y+20000,np.asarray(NE_rignotetal.SUBREGION1)[0])
ax_inset_map.text(SE_rignotetal.centroid.x-100000,SE_rignotetal.centroid.y,np.asarray(SE_rignotetal.SUBREGION1)[0])
ax_inset_map.text(SW_rignotetal.centroid.x-130000,SW_rignotetal.centroid.y-200000,np.asarray(SW_rignotetal.SUBREGION1)[0])
ax_inset_map.text(CW_rignotetal.centroid.x-50000,CW_rignotetal.centroid.y-100000,np.asarray(CW_rignotetal.SUBREGION1)[0])
ax_inset_map.text(NW_rignotetal.centroid.x,NW_rignotetal.centroid.y-150000,np.asarray(NW_rignotetal.SUBREGION1)[0])
ax_inset_map.axis('off')
ax_inset_map.set_xlim(-693308, 912076)
ax_inset_map.set_ylim(-3440844, -541728)

#Display GrIS
GrIS_drainage_bassins.plot(ax=ax_north,color='white', edgecolor='black',linewidth=0.075,zorder=0)
GrIS_drainage_bassins.plot(ax=ax_SW,color='white', edgecolor='black',linewidth=0.075,zorder=0)
GrIS_drainage_bassins.plot(ax=ax_NW,color='white', edgecolor='black',linewidth=0.075,zorder=0)
GrIS_drainage_bassins.plot(ax=ax_CW,color='white', edgecolor='black',linewidth=0.075,zorder=0)

### MoA plot ###
import xarray

#Load the MoA count 2001-2011
MoAgtT_count_2001_2011 = xarray.open_dataset("C:/Users/jullienn/switchdrive/Private/research/RT3/data/jullien_slabhydrology_MoA/jullien_moa/MARv.3.14_MoAgtT_count_2001_2011.nc")
#Load the 2012 MoA shapefile
MoA_gte_2012=gpd.read_file('C:/Users/jullienn/switchdrive/Private/research/RT3/data/jullien_slabhydrology_MoA/jullien_moa/MoA_gte_0.7_2012.shp') 

#Define extent
extent_MoAgtT_count_2001_2011 = [np.min(MoAgtT_count_2001_2011.x.data), np.max(MoAgtT_count_2001_2011.x.data), np.min(MoAgtT_count_2001_2011.y.data), np.max(MoAgtT_count_2001_2011.y.data)]#[west limit, east limit., south limit, north limit]

#Display map
cmap = sns.color_palette("flare", as_cmap=True)
ax_north.imshow(MoAgtT_count_2001_2011.__xarray_dataarray_variable__.data,
                extent=extent_MoAgtT_count_2001_2011,origin='lower',cmap=cmap,zorder=1,alpha=0.75)
MoA_gte_2012[MoA_gte_2012.raster_val==0].plot(ax=ax_north,edgecolor='green', color='none', linewidth=1,zorder=5)

cbar_MoA=ax_SW.imshow(MoAgtT_count_2001_2011.__xarray_dataarray_variable__.data,
                      extent=extent_MoAgtT_count_2001_2011,origin='lower',cmap=cmap,zorder=1,alpha=0.75)
MoA_gte_2012[MoA_gte_2012.raster_val==0].plot(ax=ax_SW,edgecolor='green', color='none', linewidth=1,zorder=5)

ax_NW.imshow(MoAgtT_count_2001_2011.__xarray_dataarray_variable__.data,
             extent=extent_MoAgtT_count_2001_2011,origin='lower',cmap=cmap,zorder=1,alpha=0.75)
MoA_gte_2012[MoA_gte_2012.raster_val==0].plot(ax=ax_NW,edgecolor='green', color='none', linewidth=1,zorder=5)

ax_CW.imshow(MoAgtT_count_2001_2011.__xarray_dataarray_variable__.data,
             extent=extent_MoAgtT_count_2001_2011,origin='lower',cmap=cmap,zorder=1,alpha=0.75)
MoA_gte_2012[MoA_gte_2012.raster_val==0].plot(ax=ax_CW,edgecolor='green', color='none', linewidth=1,zorder=5)
#Display also on inset map
MoA_gte_2012[MoA_gte_2012.raster_val==0].plot(ax=ax_inset_map,edgecolor='green', color='none', linewidth=1,zorder=1)

#Display cbar excess melt
cbar_MoA=fig.colorbar(cbar_MoA, cax=ax_colorbar,orientation='horizontal',ticklocation='bottom')#Inspired from https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
cbar_MoA.set_label('N. Yrs MoA > 0.7 (2001-2011)')
### MoA plot ###

#Load and display Greenland inverted coast shapefile
GreenlandInvertedCoast=gpd.read_file('C:/Users/jullienn/switchdrive/Private/research/backup_Aglaja/working_environment/greenland_topo_data/Greenland_coast/Inverted_Greenland_coast.shp') 
#Display coastlines
GreenlandInvertedCoast.plot(ax=ax_north,color='#CEB481', edgecolor='grey',linewidth=0.1,zorder=2)
GreenlandInvertedCoast.plot(ax=ax_SW,color='#CEB481', edgecolor='grey',linewidth=0.1,zorder=2)
GreenlandInvertedCoast.plot(ax=ax_NW,color='#CEB481', edgecolor='grey',linewidth=0.1,zorder=2)
GreenlandInvertedCoast.plot(ax=ax_CW,color='#CEB481', edgecolor='grey',linewidth=0.1,zorder=2)

'''
#Display 2010-2018 high end ice slabs jullien et al., 2023
iceslabs_20102018_jullienetal2023.plot(ax=ax_north,facecolor='#ba2b2b',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_SW,facecolor='#ba2b2b',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_NW,facecolor='#ba2b2b',edgecolor='#ba2b2b')
iceslabs_20102018_jullienetal2023.plot(ax=ax_CW,facecolor='#ba2b2b',edgecolor='#ba2b2b')
'''
#Display 2010-2012 high end ice slabs jullien et al., 2023
iceslabs_20102012_jullienetal2023.plot(ax=ax_north,facecolor='none',edgecolor='white',zorder=4,linewidth=2)
iceslabs_20102012_jullienetal2023.plot(ax=ax_SW,facecolor='none',edgecolor='white',zorder=4,linewidth=2)
iceslabs_20102012_jullienetal2023.plot(ax=ax_NW,facecolor='none',edgecolor='white',zorder=4,linewidth=2)
iceslabs_20102012_jullienetal2023.plot(ax=ax_CW,facecolor='none',edgecolor='white',zorder=4,linewidth=2)
''' REV 1
#Display 2010-2012 high end ice slabs jullien et al., 2023
iceslabs_20102012_jullienetal2023.plot(ax=ax_north,facecolor='#045a8d',edgecolor='#045a8d'
iceslabs_20102012_jullienetal2023.plot(ax=ax_SW,facecolor='#045a8d',edgecolor='#045a8d')
iceslabs_20102012_jullienetal2023.plot(ax=ax_NW,facecolor='#045a8d',edgecolor='#045a8d')
iceslabs_20102012_jullienetal2023.plot(ax=ax_CW,facecolor='#045a8d',edgecolor='#045a8d')
REV1 '''

### -------------------------- Load and display 2010-2012 AR flightlines --------------------------- ###
flighlines_2010 = pd.read_csv(path_jullienetal2023+'flightlines/'+'2010_Greenland_P3.csv',usecols=['LAT','LON'])
flighlines_2011 = pd.read_csv(path_jullienetal2023+'flightlines/'+'2011_Greenland_P3.csv',usecols=['LAT','LON'])
flighlines_2012 = pd.read_csv(path_jullienetal2023+'flightlines/'+'2012_Greenland_P3.csv',usecols=['LAT','LON'])
#Transform coordinates from WGS84 to EPSG:3413
points=transformer.transform(np.asarray(flighlines_2010["LON"]),np.asarray(flighlines_2010["LAT"]))
flighlines_2010['lon_3413']=points[0]
flighlines_2010['lat_3413']=points[1]

points=transformer.transform(np.asarray(flighlines_2011["LON"]),np.asarray(flighlines_2011["LAT"]))
flighlines_2011['lon_3413']=points[0]
flighlines_2011['lat_3413']=points[1]

points=transformer.transform(np.asarray(flighlines_2012["LON"]),np.asarray(flighlines_2012["LAT"]))
flighlines_2012['lon_3413']=points[0]
flighlines_2012['lat_3413']=points[1]

#Make flightlines geopandas dataframes
gdf_flighlines_2010 = gpd.GeoDataFrame(flighlines_2010, geometry=gpd.points_from_xy(flighlines_2010.lon_3413, flighlines_2010.lat_3413), crs="EPSG:3413")
gdf_flighlines_2011 = gpd.GeoDataFrame(flighlines_2011, geometry=gpd.points_from_xy(flighlines_2011.lon_3413, flighlines_2011.lat_3413), crs="EPSG:3413")
gdf_flighlines_2012 = gpd.GeoDataFrame(flighlines_2012, geometry=gpd.points_from_xy(flighlines_2012.lon_3413, flighlines_2012.lat_3413), crs="EPSG:3413")

#Clip flightlines to Greenland mask
gdf_flighlines_2010_GrIS = gpd.sjoin(gdf_flighlines_2010, GrIS_mask, how='left', predicate='within')
gdf_flighlines_2011_GrIS = gpd.sjoin(gdf_flighlines_2011, GrIS_mask, how='left', predicate='within')
gdf_flighlines_2012_GrIS = gpd.sjoin(gdf_flighlines_2012, GrIS_mask, how='left', predicate='within')

#Drop nans to keep only flightlines which intersect with GrIS mask
gdf_flighlines_2010_GrIS.dropna(inplace=True)
gdf_flighlines_2011_GrIS.dropna(inplace=True)
gdf_flighlines_2012_GrIS.dropna(inplace=True)

#Display flightlines
ax_north.scatter(gdf_flighlines_2010_GrIS.lon_3413,gdf_flighlines_2010_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_north.scatter(gdf_flighlines_2011_GrIS.lon_3413,gdf_flighlines_2011_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_north.scatter(gdf_flighlines_2012_GrIS.lon_3413,gdf_flighlines_2012_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)

ax_SW.scatter(gdf_flighlines_2010_GrIS.lon_3413,gdf_flighlines_2010_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_SW.scatter(gdf_flighlines_2011_GrIS.lon_3413,gdf_flighlines_2011_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_SW.scatter(gdf_flighlines_2012_GrIS.lon_3413,gdf_flighlines_2012_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)

ax_NW.scatter(gdf_flighlines_2010_GrIS.lon_3413,gdf_flighlines_2010_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_NW.scatter(gdf_flighlines_2011_GrIS.lon_3413,gdf_flighlines_2011_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_NW.scatter(gdf_flighlines_2012_GrIS.lon_3413,gdf_flighlines_2012_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)

ax_CW.scatter(gdf_flighlines_2010_GrIS.lon_3413,gdf_flighlines_2010_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_CW.scatter(gdf_flighlines_2011_GrIS.lon_3413,gdf_flighlines_2011_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
ax_CW.scatter(gdf_flighlines_2012_GrIS.lon_3413,gdf_flighlines_2012_GrIS.lat_3413,s=0.1,color='#bdbdbd',edgecolor='none',zorder=3)
### -------------------------- Load and display 2010-2012 AR flightlines --------------------------- ###

#Treat the follwing boxes together
list_together_2012=np.array([[4,6],[7,9],[10,13],[14,14],[16,19],[21,21],[22,27],[28,30],[31,31]])

#loop over box to apply a rolling window 2012
for i in range(0,len(list_together_2012)):
    print(list_together_2012[i,0])
    print(list_together_2012[i,1])
    RL_2012_continuum = RL_2012[np.logical_and(RL_2012.box_nb>=list_together_2012[i,0],RL_2012.box_nb<=list_together_2012[i,1])]
    roll_window_2012 = RL_2012_continuum.rolling(5,center=True).mean()
    #Do not loose data! Where NaN due to rolling window at extreme ends, include these data!
    roll_window_2012[roll_window_2012.isna()]=RL_2012_continuum[roll_window_2012.isna()]
    #plot
    ax_north.plot(roll_window_2012.x,roll_window_2012.y,zorder=5,color=pal_year[2012])
    ax_SW.plot(roll_window_2012.x,roll_window_2012.y,zorder=5,color=pal_year[2012])
    ax_NW.plot(roll_window_2012.x,roll_window_2012.y,zorder=5,color=pal_year[2012])
    ax_CW.plot(roll_window_2012.x,roll_window_2012.y,zorder=5,color=pal_year[2012])

'''
#Treat the follwing boxes together
list_together_2019=np.array([[4,14],[16,19],[21,21],[22,27],[28,31]])

#loop over box to apply a rolling window 2019
for i in range(0,len(list_together_2019)):
    print(list_together_2019[i,0])
    print(list_together_2019[i,1])
    RL_2019_continuum = RL_2019[np.logical_and(RL_2019.box_nb>=list_together_2019[i,0],RL_2019.box_nb<=list_together_2019[i,1])]
    
    #2019 must be cut at box 25
    if ((list_together_2019[i,0]>=22)&(list_together_2019[i,0]<=27)):
        #Where to break
        break_end_index = RL_2019_continuum[RL_2019_continuum.box_nb==25].index[-1]
        break_start_index = RL_2019_continuum[RL_2019_continuum.box_nb==25.1].index[0]
        
        #Define continuums
        RL_2019_continuum_1=RL_2019_continuum.loc[0:break_end_index]
        RL_2019_continuum_2=RL_2019_continuum.loc[break_start_index:]
        
        #Perform roll
        roll_window_2019_continuum_1 = RL_2019_continuum_1.rolling(5,center=True).mean()
        roll_window_2019_continuum_2 = RL_2019_continuum_2.rolling(5,center=True).mean()

        #Do not loose data! Where NaN due to rolling window at extreme ends, include these data!
        roll_window_2019_continuum_1[roll_window_2019_continuum_1.isna()]=RL_2019_continuum_1[roll_window_2019_continuum_1.isna()]
        roll_window_2019_continuum_2[roll_window_2019_continuum_2.isna()]=RL_2019_continuum_2[roll_window_2019_continuum_2.isna()]

        #plot
        ax_north.plot(roll_window_2019_continuum_1.x,roll_window_2019_continuum_1.y,zorder=10,color=pal_year[2019])
        ax_north.plot(roll_window_2019_continuum_2.x,roll_window_2019_continuum_2.y,zorder=10,color=pal_year[2019])
        ax_SW.plot(roll_window_2019_continuum_1.x,roll_window_2019_continuum_1.y,zorder=10,color=pal_year[2019])
        ax_SW.plot(roll_window_2019_continuum_2.x,roll_window_2019_continuum_2.y,zorder=10,color=pal_year[2019])
        ax_NW.plot(roll_window_2019_continuum_1.x,roll_window_2019_continuum_1.y,zorder=10,color=pal_year[2019])
        ax_NW.plot(roll_window_2019_continuum_2.x,roll_window_2019_continuum_2.y,zorder=10,color=pal_year[2019])
        ax_CW.plot(roll_window_2019_continuum_1.x,roll_window_2019_continuum_1.y,zorder=10,color=pal_year[2019])
        ax_CW.plot(roll_window_2019_continuum_2.x,roll_window_2019_continuum_2.y,zorder=10,color=pal_year[2019])
    else:
        roll_window_2019 = RL_2019_continuum.rolling(5,center=True).mean()
        #Do not loose data! Where NaN due to rolling window at extreme ends, include these data!
        roll_window_2019[roll_window_2019.isna()]=RL_2019_continuum[roll_window_2019.isna()]
        #plot
        ax_north.plot(roll_window_2019.x,roll_window_2019.y,zorder=10,color=pal_year[2019])
        ax_SW.plot(roll_window_2019.x,roll_window_2019.y,zorder=10,color=pal_year[2019])
        ax_NW.plot(roll_window_2019.x,roll_window_2019.y,zorder=10,color=pal_year[2019])
        ax_CW.plot(roll_window_2019.x,roll_window_2019.y,zorder=10,color=pal_year[2019])
'''

#Display boxes not processed
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_north,color = "white",edgecolor="grey",hatch= "xxxx",zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_SW,color = "white",edgecolor="grey",hatch= "xxxx",zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_NW,color = "white",edgecolor="grey",hatch= "xxxx",zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID.isin(nogo_polygon)].overlay(GrIS_mask, how='intersection').plot(ax=ax_CW,color = "white",edgecolor="grey",hatch= "xxxx",zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
#Display exclusion box in box 21
box_21_inclusion,box_21_exclusion = create_polygon_inclusion_box21(Boxes_Tedstone2022[Boxes_Tedstone2022.FID==21])
box_21_exclusion.overlay(GrIS_mask, how='intersection').plot(ax=ax_north,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
box_21_exclusion.overlay(GrIS_mask, how='intersection').plot(ax=ax_SW,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
box_21_exclusion.overlay(GrIS_mask, how='intersection').plot(ax=ax_NW,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
box_21_exclusion.overlay(GrIS_mask, how='intersection').plot(ax=ax_CW,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
#Display exclusion box 15!
Boxes_Tedstone2022[Boxes_Tedstone2022.FID==15].overlay(GrIS_mask, how='intersection').plot(ax=ax_north,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID==15].overlay(GrIS_mask, how='intersection').plot(ax=ax_SW,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID==15].overlay(GrIS_mask, how='intersection').plot(ax=ax_NW,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
Boxes_Tedstone2022[Boxes_Tedstone2022.FID==15].overlay(GrIS_mask, how='intersection').plot(ax=ax_CW,color='#d9d9d9',edgecolor='none',zorder=6)#overlay from https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
#Display Rignot and Mouginot regions edges to make sure projection is correct - it looks correct
GrIS_drainage_bassins.plot(ax=ax_north,facecolor='none',edgecolor='black',zorder=7,linewidth=0.5)
GrIS_drainage_bassins.plot(ax=ax_SW,facecolor='none',edgecolor='black',zorder=7,linewidth=0.5)
GrIS_drainage_bassins.plot(ax=ax_NW,facecolor='none',edgecolor='black',zorder=7,linewidth=0.5)
GrIS_drainage_bassins.plot(ax=ax_CW,facecolor='none',edgecolor='black',zorder=7,linewidth=0.5)

'''
#Display firn aquifers Miège et al., 2016
ax_north.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=1,zorder=4)
ax_SW.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=1,zorder=4)
ax_NW.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=1,zorder=4)
ax_CW.scatter(df_firn_aquifer_all['lon_3413'],df_firn_aquifer_all['lat_3413'],c='#238b45',s=1,zorder=4)
'''

ax_inset_map.set_xlim(-642397, 1005201)
ax_inset_map.set_ylim(-3366273, -784280)
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
gl=ax_inset_map.gridlines(draw_labels=True, xlocs=[-20,-30,-40,-50,-60,-70], ylocs=[60,65,70,75,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=6)
#Customize lat labels
gl.left_labels = False
gl.top_labels = False
ax_inset_map.axis('off')
#ax8map.legend(loc='upper right')
###################### From Tedstone et al., 2022 #####################

ax_north.set_xlim(-605557.1790513115, 464264.83348328667)
ax_north.set_ylim(-1224724.0325818455, -884876.6847590464)              
gl=ax_north.gridlines(draw_labels=True, xlocs=[-30,-50,-70], ylocs=[78,80], x_inline=False, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
gl.bottom_labels = False

ax_NW.set_xlim(-327640.39779065247, -218961.34771576658)
ax_NW.set_ylim(-1874638.6166935905, -1494261.94143149)            
gl=ax_NW.gridlines(draw_labels=True, xlocs=[-52,-53,-54,-55,-56,-57,-58,], ylocs=[73,74,75,76], x_inline=True, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
#gl.bottom_labels = False

ax_SW.set_xlim(-171994.14895702014, -65379.43459850631)
ax_SW.set_ylim(-2755750.7046506493, -2374228.6276486944)
gl=ax_SW.gridlines(draw_labels=True, xlocs=[-49,-48,-47,], ylocs=[65,66,67,68], x_inline=True, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
#gl.bottom_labels = False

ax_CW.set_ylim(-2375526.002225974, -2022535.0326401703)
ax_CW.set_xlim(-183736.7996412225, -62636.10288859956)
gl=ax_CW.gridlines(draw_labels=True, xlocs=[-47,-48,-49,-50], ylocs=[69,70,71,72], x_inline=True, y_inline=False,linewidth=0.5,linestyle='dashed',zorder=8)
gl.right_labels = False
#gl.bottom_labels = False

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [#Patch(facecolor='#ba2b2b',edgecolor='none',label='2017-2018 ice slabs'),
                   Line2D([0], [0], color=pal_year[2012], lw=2, label='2012 runoff limit'),
                   Line2D([0], [0], color='green', lw=2, label='2012 MoA >= 0.7'),
                   Patch(facecolor='none',edgecolor='white',label='2010-2012 ice slabs',linewidth=2),
                   Line2D([0], [0], color='#bdbdbd', lw=1, label='OIB AR flight-lines'),
                   Patch(facecolor='white',edgecolor='grey',hatch= "xxxx",label='Ignored areas')]
ax_north.legend(handles=legend_elements,loc='lower center',fontsize=8,framealpha=0.6, bbox_to_anchor=(0.65, 0)).set_zorder(15)
plt.show()

# Display scalebar with GeoPandas
ax_north.add_artist(ScaleBar(1,location='upper left',box_alpha=0)).set_pad(1.5)
ax_NW.add_artist(ScaleBar(1,location='lower left',box_alpha=0))
ax_SW.add_artist(ScaleBar(1,location='lower right',box_alpha=0))
ax_CW.add_artist(ScaleBar(1,location='lower left',box_alpha=0))

#Display boxes zoom
add_rectangle_zoom(ax_inset_map,ax_north)
add_rectangle_zoom(ax_inset_map,ax_NW)
add_rectangle_zoom(ax_inset_map,ax_CW)
add_rectangle_zoom(ax_inset_map,ax_SW)

#Display panel label
ax_north.text(0.02, 0.95,'a',ha='center', va='center', transform=ax_north.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_NW.text(0.05, 0.975,'b',ha='center', va='center', transform=ax_NW.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_CW.text(0.05, 0.975,'c',ha='center', va='center', transform=ax_CW.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_SW.text(0.05, 0.975,'d',ha='center', va='center', transform=ax_SW.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(-0.1, 0.95,'f',ha='center', va='center', transform=ax_inset_map.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#Display panel label on inset map
ax_inset_map.text(ax_north.get_xlim()[1]+120000, (ax_north.get_ylim()[0]+ax_north.get_ylim()[1])/2,'a',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(ax_NW.get_xlim()[0]-120000, (ax_NW.get_ylim()[0]+ax_NW.get_ylim()[1])/2,'b',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(ax_CW.get_xlim()[0]-120000, (ax_CW.get_ylim()[0]+ax_CW.get_ylim()[1])/2,'c',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
ax_inset_map.text(ax_SW.get_xlim()[0]-120000, (ax_SW.get_ylim()[0]+ax_SW.get_ylim()[1])/2,'d',ha='center', va='center', fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot


### !!! There is a -9999 elevation in 2012 in box 5

#Elevation difference, and signed distances between 2010-2012 Ice Slabs boundary and 2012 RL. Positive means ice slabs upper end higher than RL
RL_IceSlabs_gdp['Elev_diff_IceSlabs20102012_2012RL']=RL_IceSlabs_gdp.Elevation_IceSlabsBoundary20102012-RL_IceSlabs_gdp.Elevation_2012_RL
RL_IceSlabs_gdp['Signed_dist_IceSlabs20102012_2012RL']=RL_IceSlabs_gdp.Distance_IceSlabsBoundary20102012_2012_RL*np.sign(RL_IceSlabs_gdp.Elev_diff_IceSlabs20102012_2012RL)

#Elevation difference, and signed distances between 2010-2018 Ice Slabs boundary and 2019 RL. Positive means ice slabs upper end higher than RL
RL_IceSlabs_gdp['Elev_diff_IceSlabs20102018_2019RL']=RL_IceSlabs_gdp.Elevation_IceSlabsBoundary20102018-RL_IceSlabs_gdp.Elevation_2019_RL
RL_IceSlabs_gdp['Signed_dist_IceSlabs20102018_2019RL']=RL_IceSlabs_gdp.Distance_IceSlabsBoundary20102018_2019_RL*np.sign(RL_IceSlabs_gdp.Elev_diff_IceSlabs20102018_2019RL)

#Concatenate data to do a violin plot
df_plot_IceSlabs20102012_2012RL=RL_IceSlabs_gdp[['Elev_diff_IceSlabs20102012_2012RL','Signed_dist_IceSlabs20102012_2012RL','RL2012Region']].copy()
df_plot_IceSlabs20102012_2012RL['Year']=[2012]*len(df_plot_IceSlabs20102012_2012RL)
df_plot_IceSlabs20102012_2012RL.rename(columns={"Elev_diff_IceSlabs20102012_2012RL": "Elev_diff_IceSlabs_RL", 'Signed_dist_IceSlabs20102012_2012RL': 'Signed_dist_IceSlabs_RL', "RL2012Region": "Region"},inplace=True)

df_plot_IceSlabs20102018_2019RL=RL_IceSlabs_gdp[['Elev_diff_IceSlabs20102018_2019RL','Signed_dist_IceSlabs20102018_2019RL','RL2019Region']].copy()
df_plot_IceSlabs20102018_2019RL['Year']=[2019]*len(df_plot_IceSlabs20102018_2019RL)
df_plot_IceSlabs20102018_2019RL.rename(columns={"Elev_diff_IceSlabs20102018_2019RL": "Elev_diff_IceSlabs_RL", 'Signed_dist_IceSlabs20102018_2019RL': 'Signed_dist_IceSlabs_RL', "RL2019Region": "Region"},inplace=True)

#Display quantiles in the difference regions of signed distances and elevation difference.
print('---- df_plot_IceSlabs20102012_2012RL ---- ')
print('- Regions')
print('signed_dist_diff:')
print('    ',df_plot_IceSlabs20102012_2012RL.groupby(["Region"]).quantile([0.25,0.5,0.75]).Signed_dist_IceSlabs_RL)
print(' ')
print('elev_diff:')
print('    ',df_plot_IceSlabs20102012_2012RL.groupby(["Region"]).quantile([0.25,0.5,0.75]).Elev_diff_IceSlabs_RL)
print(' ')
print('Count:')
print('    ',df_plot_IceSlabs20102012_2012RL.groupby(["Region"]).count())

#Display quantiles of signed distances and elevation difference without differentiating the regions.
print('- Total')
print('signed_dist_diff:')
print('    ',df_plot_IceSlabs20102012_2012RL.quantile([0.25,0.5,0.75]).Signed_dist_IceSlabs_RL)
print(' ')
print('elev_diff:')
print('    ',df_plot_IceSlabs20102012_2012RL.quantile([0.25,0.5,0.75]).Elev_diff_IceSlabs_RL)
print(' ')
print('Count:')
print('    ',df_plot_IceSlabs20102012_2012RL.count())


print('---- df_plot_IceSlabs20102018_2019RL ---- ')
print('- Regions')
print('signed_dist_diff:')
print('    ',df_plot_IceSlabs20102018_2019RL.groupby(["Region"]).quantile([0.25,0.5,0.75]).Signed_dist_IceSlabs_RL)
print(' ')
print('elev_diff:')
print('    ',df_plot_IceSlabs20102018_2019RL.groupby(["Region"]).quantile([0.25,0.5,0.75]).Elev_diff_IceSlabs_RL)
print(' ')
print('Count:')
print('    ',df_plot_IceSlabs20102018_2019RL.groupby(["Region"]).count())

#Display quantiles of signed distances and elevation difference without differentiating the regions.
print('- Total')
print('signed_dist_diff:')
print('    ',df_plot_IceSlabs20102018_2019RL.quantile([0.25,0.5,0.75]).Signed_dist_IceSlabs_RL)
print(' ')
print('elev_diff:')
print('    ',df_plot_IceSlabs20102018_2019RL.quantile([0.25,0.5,0.75]).Elev_diff_IceSlabs_RL)
print(' ')
print('Count:')
print('    ',df_plot_IceSlabs20102018_2019RL.count())

''' ### REV 1
#Concatenate data to display panels f and g
df_plot_IceSlabs_RL = pd.concat([df_plot_IceSlabs20102012_2012RL,df_plot_IceSlabs20102018_2019RL])
### REV 1'''
#Concatenate data to display panels f and g
df_plot_IceSlabs_RL = df_plot_IceSlabs20102012_2012RL.copy()
#Reset index of df_plot_IceSlabs_RL
df_plot_IceSlabs_RL.reset_index(inplace=True)
#Transform signed distances from m to km
df_plot_IceSlabs_RL.Signed_dist_IceSlabs_RL=df_plot_IceSlabs_RL.Signed_dist_IceSlabs_RL/1000

#Display elevation difference
pal_year= {2012 : "grey", 2019 : "#fcbba1"}
sns.boxplot(data=df_plot_IceSlabs_RL,x="Elev_diff_IceSlabs_RL",y="Region",hue='Year',ax=axsummary_elev,orient='h',fliersize=0,palette=pal_year)
axsummary_elev.set_xlabel('Elevation difference [m]')
axsummary_elev.set_ylabel('Region',labelpad=10)
axsummary_elev.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False)
axsummary_elev.set_xlim(-310,310)
axsummary_elev.text(0.025, 0.9,'e',ha='center', va='center', transform=axsummary_elev.transAxes,weight='bold',fontsize=12,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#This is from https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
axsummary_elev.set_axisbelow(True)
axsummary_elev.grid(color='gray', linestyle='dashed',linewidth=0.5)

pdb.set_trace()

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig1/v11/Fig1_cleanedxytpdV3.png',dpi=1000,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''

#Display signed distances
#Prepare plot
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(8.55, 4.09))
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
gs = gridspec.GridSpec(5, 5)
axsummary_signed_dist = plt.subplot(gs[0:5, 0:5])

#Display violing plot of signed distances differences
sns.boxplot(data=df_plot_IceSlabs_RL,x="Signed_dist_IceSlabs_RL",y="Region",hue='Year',ax=axsummary_signed_dist,orient='h',palette=pal_year)

#Custom plot
axsummary_signed_dist.set_xlabel('Distance difference [km]')
axsummary_signed_dist.set_ylabel(' ')
axsummary_signed_dist.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=False, right=True, labelright=True)
axsummary_signed_dist.yaxis.set_label_position('right')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
axsummary_signed_dist.set_xlim(-50,50)
axsummary_signed_dist.text(0.03, 0.95,'b',ha='center', va='center', transform=axsummary_signed_dist.transAxes,weight='bold',fontsize=15,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

#This is from https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
axsummary_signed_dist.set_axisbelow(True)
axsummary_signed_dist.grid(color='gray', linestyle='dashed')

pdb.set_trace()

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig1/v6/Fig1Supp_SignedDist_cleanedxytpdV3.png',dpi=1000,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''

### Calculate differences between 2019 and 2012 runoff limits ###
#Elevation difference between 2019 RL and 2012 RL 
RL_IceSlabs_gdp["Elev_diff_2019RL_2012RL"]=RL_IceSlabs_gdp.Elevation_2019_RL-RL_IceSlabs_gdp.Elevation_2012_RL
#Signed distances difference between 2019 RL and 2012 RL 
RL_IceSlabs_gdp["Signed_Distance_2019_2012_RL"]=RL_IceSlabs_gdp.Distance_2012_2019_RL*np.sign(RL_IceSlabs_gdp.Elev_diff_2019RL_2012RL)

print('---- Elev_diff_2019RL_2012RL ---- ')
print('    ',np.round(RL_IceSlabs_gdp.groupby(["RL2019Region"]).quantile([0.25,0.5,0.75]).Elev_diff_2019RL_2012RL))
print(' ')
print('---- Signed dist_2019RL_2012RL ---- ')
print('    ',np.round(RL_IceSlabs_gdp.groupby(["RL2019Region"]).quantile([0.25,0.5,0.75]).Signed_Distance_2019_2012_RL))
print(' ')
print('Count:')
print('    ',np.round(RL_IceSlabs_gdp.groupby(["RL2019Region"]).count().Elev_diff_2019RL_2012RL))

RL_IceSlabs_gdp.Signed_Distance_2019_2012_RL=RL_IceSlabs_gdp.Signed_Distance_2019_2012_RL/1000
#Set fontsize plot
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
fig.set_size_inches(12, 5) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
gs = gridspec.GridSpec(4, 8)
gs.update(wspace=1)
axsummary_elev_RL = plt.subplot(gs[0:4, 0:4])
axsummary_signed_dist_RL = plt.subplot(gs[0:4, 4:48])
sns.boxplot(data=RL_IceSlabs_gdp,x="Elev_diff_2019RL_2012RL",y="RL2019Region",ax=axsummary_elev_RL,orient='h',color='grey')
sns.boxplot(data=RL_IceSlabs_gdp,x="Signed_Distance_2019_2012_RL",y="RL2019Region",ax=axsummary_signed_dist_RL,orient='h',color='grey')
#Custom plot
axsummary_elev_RL.set_xlabel('Elevation difference [m]')
axsummary_elev_RL.set_ylabel('Region',labelpad=10)
axsummary_elev_RL.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False)
axsummary_elev_RL.set_xlim(-200,200)
axsummary_elev_RL.grid()
axsummary_signed_dist_RL.set_xlabel('Distance difference [km]')
axsummary_signed_dist_RL.set_ylabel(' ')
axsummary_signed_dist_RL.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=False, right=True, labelright=True)
axsummary_signed_dist.yaxis.set_label_position('right')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
axsummary_signed_dist_RL.set_xlim(-30,30)
axsummary_signed_dist_RL.grid()
axsummary_elev_RL.text(0.035, 0.95,'a',ha='center', va='center', transform=axsummary_elev_RL.transAxes,weight='bold',fontsize=15,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
axsummary_signed_dist_RL.text(0.035, 0.95,'b',ha='center', va='center', transform=axsummary_signed_dist_RL.transAxes,weight='bold',fontsize=15,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig_RL20192019_diff/v1/Fig_RL20192019_diff.png',dpi=1000,bbox_inches='tight')
#bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
'''




pdb.set_trace()


#Display boxes summary
display_summary(elevation_differences,'Elevation',10)
display_summary(distance_differences,'Distance',100)
display_summary(signed_distances,'Signed distance',100)


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
    '''
    #Save figure
    plt.savefig(path_local+'MVRL/Difference_elevation_2012_2019_box_'+str(indiv_box)+'_cleanedV3.png',dpi=300,bbox_inches='tight')
    #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    '''
    #Possibly filter out outliers? (e.g. 300 m difference is very likely one). Do not do it,maybe consider later on
    plt.close()

print('--- End of code ---')

