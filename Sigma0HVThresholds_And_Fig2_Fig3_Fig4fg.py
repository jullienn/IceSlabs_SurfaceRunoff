# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:01:33 2022

@author: jullienn
"""

#The fuction plot_histo is from Emax_SlabsThickness.py
def plot_histo(ax_plot,iceslabs_above,iceslabs_within,iceslabs_below,region):
    if (region == 'GrIS'):
        ax_plot.hist(iceslabs_above['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_below['20m_ice_content_m'],color='green',label='Below',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_below['20m_ice_content_m'],0.5),linestyle='--',color='green')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.nanquantile(iceslabs_below['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='green')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Display sample size
        print(region)
        print('-> Sample size:')
        print('   Above: ',len(iceslabs_above))
        print('   Within: ',len(iceslabs_within))
        print('   Below: ',len(iceslabs_below))
        print('-> Coefficient of variation:')
        print('   Above: ',np.round(iceslabs_above['20m_ice_content_m'].std()/iceslabs_above['20m_ice_content_m'].mean(),4))
        print('   Within: ',np.round(iceslabs_within['20m_ice_content_m'].std()/iceslabs_within['20m_ice_content_m'].mean(),4))
        print('   Below: ',np.round(iceslabs_below['20m_ice_content_m'].std()/iceslabs_below['20m_ice_content_m'].mean(),4))
        print('-> MAD/median:')
        print('   Above: ',np.round(stats.median_abs_deviation(iceslabs_above['20m_ice_content_m'],nan_policy='omit')/iceslabs_above['20m_ice_content_m'].median(),2))
        print('   Within: ',np.round(stats.median_abs_deviation(iceslabs_within['20m_ice_content_m'],nan_policy='omit')/iceslabs_within['20m_ice_content_m'].median(),2))
        print('   Below: ',np.round(stats.median_abs_deviation(iceslabs_below['20m_ice_content_m'],nan_policy='omit')/iceslabs_below['20m_ice_content_m'].median(),2))        
        print('\n')        
    else:
        ax_plot.hist(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],color='blue',label='Above',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],color='red',label='Within',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.hist(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],color='green',label='Below',alpha=0.5,bins=np.arange(0,17),density=True)
        ax_plot.text(0.075, 0.9,region,zorder=10, ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        #Dislay median values
        ax_plot.axvline(x=np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='blue')
        ax_plot.text(0.75, 0.25,'med:'+str(np.round(np.nanquantile(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='blue')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='red')
        ax_plot.text(0.75, 0.5,'med:'+str(np.round(np.nanquantile(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='red')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
        ax_plot.axvline(x=np.nanquantile(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],0.5),linestyle='--',color='green')
        ax_plot.text(0.75, 0.05,'med:'+str(np.round(np.nanquantile(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],0.5),1))+'m',ha='center', va='center', transform=ax_plot.transAxes,fontsize=15,weight='bold',color='green')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
                
        #Display sample size
        print(region)
        print('-> Sample size:')
        print('   Above: ',len(iceslabs_above[iceslabs_above['key_shp']==region]))
        print('   Within: ',len(iceslabs_within[iceslabs_within['key_shp']==region]))
        print('   Below: ',len(iceslabs_below[iceslabs_below['key_shp']==region]))
        print('-> Coefficient of variation:')
        print('   Above: ',np.round(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'].std()/iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'].mean(),4))
        print('   Within: ',np.round(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'].std()/iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'].mean(),4))
        print('   Below: ',np.round(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'].std()/iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'].mean(),4))
        print('-> MAD/median:')
        print('   Above: ',np.round(stats.median_abs_deviation(iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'],nan_policy='omit')/iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'].median(),2))
        print('   Within: ',np.round(stats.median_abs_deviation(iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'],nan_policy='omit')/iceslabs_within[iceslabs_within['key_shp']==region]['20m_ice_content_m'].median(),2))
        print('   Below: ',np.round(stats.median_abs_deviation(iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'],nan_policy='omit')/iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'].median(),2))
        print('-> Perform Welsch t-test above VS below:')
                
        above_to_test=iceslabs_above[iceslabs_above['key_shp']==region]['20m_ice_content_m'].copy()
        above_to_test=above_to_test[~above_to_test.isna()]
        
        below_to_test=iceslabs_below[iceslabs_below['key_shp']==region]['20m_ice_content_m'].copy()
        below_to_test=below_to_test[~below_to_test.isna()]
        
        #Perform a Welsch's t test when we have no normality, no equal variance
        #Perform a Yuen's t test when we have no normality, no equal variance, and tailed distribution - https://www.youtube.com/watch?v=D_dZyUpgGkI
        #According to scipy doc, "Trimming is recommended if the underlying distribution is long-tailed or contaminated with outliers"
        #Should I perform a Yuen's t-test? If yes, ass in the ttest_ind() the arguemtn trim = 0.x, where x represents the percentage of data in the extremetiy of the distribution to be excluded
        print('  ',stats.ttest_ind(above_to_test,below_to_test,equal_var=False))#, trim=.1))#If negative t statistic return, this means the mean of above is less than the mean of below. If p < alpha (alpha being the significance level), then the difference between the two distributions is significantly different at the alpha level.
        print('\n')
    
    #Set x lims
    ax_plot.set_xlim(-0.5,20)

    if (region == 'NW'):
        ax_plot.legend()
    
    if (region in list(['NO','NE','GrIS'])):
        ax_plot.yaxis.tick_right()#This is from Fig4andS6andS7.py from paper 'Greenland Ice Slabs Expansion and Thickening'
    
    if (region in list(['NW','NO','CW','NE'])):
        ax_plot.set_xticklabels([])

    return


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def deg_n(x, a, n):
    return a * np.power(x,n)

def inverse_x(x, a, b, c):
    return a+b/(c*x)


def keep_sectorial(df_input,indiv_trackname_tokeep):    
    df_output=df_input[df_input.Track_name==indiv_trackname_tokeep].copy()
    return df_output

def sector_association(indiv_df_SAR_IceThickness,indiv_df_sectors,sector):
            
    #Perform match between indiv_df_SAR_IceThickness and indiv_IceThickness_sectors 
    indiv_df_SAR_IceThickness_sector = indiv_df_SAR_IceThickness.merge(indiv_df_sectors, how="left",on=['lat','lon'],suffixes=('','_droite'))
        
    #drop useless columns
    indiv_df_SAR_IceThickness_sector=indiv_df_SAR_IceThickness_sector.drop(columns=['Unnamed: 0_droite', 'Track_name_droite', 'Tracenumber',
                                                          'alongtrack_distance_m', '20m_ice_content_m_droite',
                                                          'likelihood_droite', 'lat_3413_droite', 'lon_3413_droite',
                                                          'key_shp_droite', 'elevation_droite', 'year_droite', 'geometry',
                                                          'index_right_polygon', 'FID', 'rev_subs', 'index_right_droite'])
    
    #Get rid of data points which do not intersect with sector
    indiv_df_SAR_IceThickness_sector_noNaN=indiv_df_SAR_IceThickness_sector[~indiv_df_SAR_IceThickness_sector.type.isna()]
    
    if (len(indiv_df_SAR_IceThickness_sector_noNaN)>0):
        #Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
        indiv_upsampled_SAR_and_IceSlabs_sector=indiv_df_SAR_IceThickness_sector_noNaN.groupby('index_right').mean()
        #Add column of sector
        indiv_upsampled_SAR_and_IceSlabs_sector['sector']=[sector]*len(indiv_upsampled_SAR_and_IceSlabs_sector)
    else:
        indiv_upsampled_SAR_and_IceSlabs_sector=indiv_df_SAR_IceThickness_sector_noNaN
    
    return indiv_upsampled_SAR_and_IceSlabs_sector


def hist_regions(df_to_plot,region_to_plot,ax_region):
    ax_region.hist(df_to_plot[df_to_plot.sector=='Below']['SAR'],density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='green',label='Below')
    ax_region.hist(df_to_plot[df_to_plot.sector=='Above']['SAR'],density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='blue',label='Above')
    ax_region.set_xlim(-20,-2)
    ax_region.set_xlabel('Signal strength [dB]')
    ax_region.set_ylabel('Density')
    ax_region.text(-19.8,0.2,region_to_plot)
    
    
    ### Should I test for normality, and if normal then perform student t-test???
    #Display sample size
    print(region_to_plot)
    print('-> Median:')
    print('   Above: ',np.round(df_to_plot[df_to_plot.sector=='Above']['SAR'].median(),2))
    print('   Within: ',np.round(df_to_plot[df_to_plot.sector=='Within']['SAR'].median(),2))
    print('   Below: ',np.round(df_to_plot[df_to_plot.sector=='Below']['SAR'].median(),2))
    print('-> Coefficient of variation:')
    print('   Above: ',np.round(df_to_plot[df_to_plot.sector=='Above']['SAR'].std()/df_to_plot[df_to_plot.sector=='Above']['SAR'].mean(),4))
    print('   Within: ',np.round(df_to_plot[df_to_plot.sector=='Within']['SAR'].std()/df_to_plot[df_to_plot.sector=='Within']['SAR'].mean(),4))
    print('   Below: ',np.round(df_to_plot[df_to_plot.sector=='Below']['SAR'].std()/df_to_plot[df_to_plot.sector=='Below']['SAR'].mean(),4))
    print('-> MAD/median:')
    print('   Above: ',np.round(stats.median_abs_deviation(df_to_plot[df_to_plot.sector=='Above']['SAR'],nan_policy='omit')/df_to_plot[df_to_plot.sector=='Above']['SAR'].median(),2))
    print('   Within: ',np.round(stats.median_abs_deviation(df_to_plot[df_to_plot.sector=='Within']['SAR'],nan_policy='omit')/df_to_plot[df_to_plot.sector=='Within']['SAR'].median(),2))
    print('   Below: ',np.round(stats.median_abs_deviation(df_to_plot[df_to_plot.sector=='Below']['SAR'],nan_policy='omit')/df_to_plot[df_to_plot.sector=='Below']['SAR'].median(),2))
    print('-> Sample size:')
    print('   Above: ',len(df_to_plot[df_to_plot.sector=='Above']['SAR']))
    print('   Within: ',len(df_to_plot[df_to_plot.sector=='Within']['SAR']))
    print('   Below: ',len(df_to_plot[df_to_plot.sector=='Below']['SAR']))
    print('-> Perform Welsch t-test above VS below:')
    
    above_to_test=df_to_plot[df_to_plot.sector=='Above']['SAR'].copy()
    above_to_test=above_to_test[~above_to_test.isna()]
    
    below_to_test=df_to_plot[df_to_plot.sector=='Below']['SAR'].copy()
    below_to_test=below_to_test[~below_to_test.isna()]
    
    #Perform a Welsch's t test when we have no normality, no equal variance
    print('  ',stats.ttest_ind(above_to_test,below_to_test,equal_var=False))#If negative t statistic return, this means the mean of above is less than the mean of below. If p < alpha (alpha being the significance level), then the difference between the two distributions is significantly different at the alpha level.
    print('\n')
    
    return

    
def regional_normalisation(df_to_normalize,region):
    #x'=(x-min(x))/(max(x)-min(x))
    #○Note that I did choose to normalise using quantile 0.01 and quantile 0.99 to avoid normalisation by outliers
    #Select regional df
    df_to_normalize_regional = df_to_normalize[df_to_normalize.SUBREGION1==region].copy()
    #normalise
    df_to_normalize_regional['normalized_raster']=(df_to_normalize_regional.raster_values-df_to_normalize_regional.raster_values.quantile(0.01))/(df_to_normalize_regional.raster_values.quantile(0.99)-df_to_normalize_regional.raster_values.quantile(0.01))
    #Where outside of [0,1] range, set to min and max
    df_to_normalize_regional.loc[df_to_normalize_regional["normalized_raster"]>1,"normalized_raster"]=1
    df_to_normalize_regional.loc[df_to_normalize_regional["normalized_raster"]<0,"normalized_raster"]=0

    return df_to_normalize_regional


def extract_regional_normalised_SAR(df_for_normalization,SAR_to_normalize,region):
    #x'=(x-min(x))/(max(x)-min(x))
    #○Note that I did choose to normalise using quantile 0.01 and quantile 0.99 to avoid normalisation by outliers
    #Select regional df
    df_for_normalization_regional = df_for_normalization[df_for_normalization.SUBREGION1==region].copy()
    #normalise    
    normalised_regional_SAR=(SAR_to_normalize[region]['SAR']-df_for_normalization_regional.raster_values.quantile(0.01))/(df_for_normalization_regional.raster_values.quantile(0.99)-df_for_normalization_regional.raster_values.quantile(0.01))
    
    return normalised_regional_SAR


def display_2d_histogram(df_to_display,FS_display,method,thresholds_dictionnary_in_func):
    
    pdb.set_trace()
    
    #Create empty datafrme to store regional nornalised SAR dataframes
    df_to_display_normalised=pd.DataFrame()
    
    #Display 2D histogram
    fig_heatmap, ((ax_SW, ax_CW, ax_NW), (ax_NO, ax_NE, ax_GrIS)) = plt.subplots(2, 3)
    fig_heatmap.set_size_inches(12.47, 7) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    
    for region in list(['SW','CW','NW','NO','NE']):
        if (region == 'SW'):
            ax_plot=ax_SW
        elif (region == 'CW'):
            ax_plot=ax_CW
        elif (region == 'NW'):
            ax_plot=ax_NW
        elif (region == 'NO'):
            ax_plot=ax_NO
        elif (region == 'NE'):
            ax_plot=ax_NE
        else:
            print('Region not found!')
            pdb.set_trace()
        
        #Display 2d hist
        cbar_region=ax_plot.hist2d(df_to_display[df_to_display.SUBREGION1==region]['raster_values'],
                                   df_to_display[df_to_display.SUBREGION1==region]['20m_ice_content_m'],
                                   bins=[np.arange(-16,-3,0.5),np.arange(0,16.5,0.5)],cmap='magma_r',cmin=1)
        
        #The maximum of occurrence in a single cell is the maximum occurrence without considering < 1 m and > 16 m thick ice slabs
        occurrence_matrix = cbar_region[0][:,np.logical_and(cbar_region[2]>=1,cbar_region[2]<15.5)[0:-1]]
        
        #Set a maximum to the colorbar   
        cbar_region[3].set_clim(vmin=np.quantile(np.arange(np.nanmin(occurrence_matrix),np.nanmax(occurrence_matrix),1),0.05), vmax=np.nanmax(occurrence_matrix))#from https://stackoverflow.com/questions/15282189/setting-matplotlib-colorbar-range
        #For ice thickness at SAR threshold extraction, set quantile to 0.5
        
        #Display colorbar
        fig_heatmap.colorbar(cbar_region[3], ax=ax_plot,label='Count') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot

        #Display runoff thresholds
        ax_plot.axvline(x=thresholds_dictionnary_in_func[region]['SAR'][0],linestyle='dashed',color='#005AB5')
        ax_plot.axvline(x=thresholds_dictionnary_in_func[region]['SAR'][1],linestyle='dashed',color='#DC3220')
        ax_plot.set_ylim(0,16)
        ax_plot.set_xlim(-16,-3.5)
        ax_plot.set_title(region)
        
        #Normalise SAR per region!
        df_to_display_normalised_region = regional_normalisation(df_to_display,region)
        #Concatenate regional normalised dataframes
        df_to_display_normalised=pd.concat([df_to_display_normalised,df_to_display_normalised_region])
        
        #Identify the corresponding value to which the SAR thresholds correspond to in the normalised world
        thresholds_dictionnary_in_func[region]['normalised_SAR']=extract_regional_normalised_SAR(df_to_display,thresholds_dictionnary_in_func,region)

    '''
    ### Display manual fit function to data ###
    ax_SW.plot(np.arange(-16,-3,0.05), exponential_func(np.arange(-16,-3,0.05), 0.25,-0.44,-5),color='blue',label='y = 0.25*exp(-0.44*x)-5')
    ax_CW.plot(np.arange(-16,-3,0.05), exponential_func(np.arange(-16,-3,0.05), 1.8,-0.24,-7),color='blue',label='y = 1.8*exp(-0.24*x)-7')
    ax_NW.plot(np.arange(-16,-3,0.05), exponential_func(np.arange(-16,-3,0.05), 1.8,-0.225,-6),color='blue',label='y = 1.8*exp(-0.225*x)-6')
    ax_NO.plot(np.arange(-16,-3,0.05), exponential_func(np.arange(-16,-3,0.05), 1.6,-0.26,-7),color='blue',label='y = 1.6*exp(-0.26*x)-7')
    ax_NE.plot(np.arange(-10,-3,0.05), exponential_func(np.arange(-10,-3,0.05), 2,-0.3,-12),color='blue',label='y = 2*exp(-0.3*x)-12')
    ### Display manual fit function to data ###
    '''
    ### Display ice thickness estimates needed to trigger runoff ###
    ax_SW.plot([-16,thresholds_dictionnary_in_func['SW']['SAR'][1]],
                [3.5,3.5],linestyle='dashed',color='black')
    ax_SW.plot([-16,thresholds_dictionnary_in_func['SW']['SAR'][1]],
                [8.5,8.5],linestyle='dashed',color='black')
    
    ax_CW.plot([-16,thresholds_dictionnary_in_func['CW']['SAR'][1]],
                [3,3],linestyle='dashed',color='black')
    ax_CW.plot([-16,thresholds_dictionnary_in_func['CW']['SAR'][1]],
                [6,6],linestyle='dashed',color='black')
    
    ax_NW.plot([-16,thresholds_dictionnary_in_func['NW']['SAR'][1]],
                [3.5,3.5],linestyle='dashed',color='black')
    ax_NW.plot([-16,thresholds_dictionnary_in_func['NW']['SAR'][1]],
                [8.5,8.5],linestyle='dashed',color='black')
    
    ax_NO.plot([-16,thresholds_dictionnary_in_func['NO']['SAR'][1]],
                [2,2],linestyle='dashed',color='black')
    ax_NO.plot([-16,thresholds_dictionnary_in_func['NO']['SAR'][1]],
                [5,5],linestyle='dashed',color='black')
    
    ax_NE.plot([-16,thresholds_dictionnary_in_func['NE']['SAR'][1]],
                [3.5,3.5],linestyle='dashed',color='black')
    ax_NE.plot([-16,thresholds_dictionnary_in_func['NE']['SAR'][1]],
                [4.5,4.5],linestyle='dashed',color='black')
    ### Display ice thickness estimates needed to trigger runoff ###

    #Display GrIS normalised 2d histogram
    cbar_GrIS=ax_GrIS.hist2d(df_to_display_normalised['normalized_raster'],
                             df_to_display_normalised['20m_ice_content_m'],
                             bins=[len(np.linspace(0,1,len(np.arange(-16,-3,0.5)))),np.arange(0,16.5,0.5)],cmap='magma_r',cmin=1)
    
    #The maximum of occurrence in a single cell is the maximum occurrence without considering < 1 m and > 16 m thick ice slabs
    occurrence_matrix = cbar_GrIS[0][:,np.logical_and(cbar_GrIS[2]>=1,cbar_GrIS[2]<15.5)[0:-1]]
        
    #Set a maximum to the colorbar 
    cbar_GrIS[3].set_clim(vmin=np.quantile(np.arange(np.nanmin(occurrence_matrix),np.nanmax(occurrence_matrix),1),0.05), vmax=np.nanmax(occurrence_matrix))#from https://stackoverflow.com/questions/15282189/setting-matplotlib-colorbar-range
    
    ax_GrIS.set_ylim(0,16)
    ax_GrIS.set_xlim(0,1)
    ax_GrIS.set_title('All')
    fig_heatmap.colorbar(cbar_GrIS[3], ax=ax_GrIS,label='Count') #this is from https://stackoverflow.com/questions/42387471/how-to-add-a-colorbar-for-a-hist2d-plot

    '''
    ### Display manual fit function to data ###
    #sort df_to_display_normalised
    df_to_display_normalised.sort_values(by=['normalized_raster'],inplace=True)
    #prepare data for fit        
    xdata = np.array(df_to_display_normalised['normalized_raster'])
    ydata = np.array(df_to_display_normalised['20m_ice_content_m'])  
    
    ax_GrIS.plot(xdata, exponential_func(xdata, 30,-2.5,-3),color='blue',label='y = 2*exp(-0.3*x)-12')
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Manual fit')]
    ax_GrIS.legend(handles=legend_elements,loc='best',fontsize=10,framealpha=0.5).set_zorder(7)
    ### Display manual fit function to data ###
    '''
    
    ### Finalise plot ###
    #Set labels
    ax_SW.set_xlabel('$\sigma_{0}^{HV}$ [dB]')
    ax_CW.set_xlabel('$\sigma_{0}^{HV}$ [dB]')
    ax_NW.set_xlabel('$\sigma_{0}^{HV}$ [dB]')
    ax_NO.set_xlabel('$\sigma_{0}^{HV}$ [dB]')
    ax_NE.set_xlabel('$\sigma_{0}^{HV}$ [dB]')
    ax_GrIS.set_xlabel('Normalised $\sigma_{0}^{HV}$ [-]')

    ax_SW.set_ylabel('Ice slab thickness [m]')
    ax_CW.set_ylabel('Ice slab thickness [m]')
    ax_NW.set_ylabel('Ice slab thickness [m]')
    ax_NO.set_ylabel('Ice slab thickness [m]')
    ax_NE.set_ylabel('Ice slab thickness [m]')
    ax_GrIS.set_ylabel('Ice slab thickness [m]')
    '''
    #Display firn cores ice content and SAR on SW plot
    ax_SW.scatter(FS_display['SAR'],FS_display['10m_ice_content_%']/10,c='black',marker='x')
    legend_elements = [Line2D([0], [0], color='black', marker='x',linestyle='none', label='Firn station')]
    ax_SW.legend(handles=legend_elements,loc='lower left',fontsize=10,framealpha=0.5).set_zorder(7)
    '''
    #Custom legend myself for ax_SW - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    legend_elements = [Line2D([0], [0], color='#DC3220', lw=2 ,linestyle='dashed', label='Upper threshold'),
                       Line2D([0], [0], color='#005AB5', lw=2 ,linestyle='dashed', label='Lower threshold'),
                       Line2D([0], [0], color='black', lw=2 ,linestyle='dashed', label='Ice thickness retrieval')]
    ax_NE.legend(handles=legend_elements,loc='upper left',bbox_to_anchor=(0, 0.85), fontsize=10,framealpha=0.5).set_zorder(7)
    
    #Add panel labels
    ax_SW.text(0.04, 0.925,'a',ha='center', va='center', transform=ax_SW.transAxes,weight='bold',fontsize=15,color='black',zorder=10)#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    ax_CW.text(0.04, 0.925,'b',ha='center', va='center', transform=ax_CW.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
    ax_NW.text(0.04, 0.925,'c',ha='center', va='center', transform=ax_NW.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
    ax_NO.text(0.04, 0.925,'d',ha='center', va='center', transform=ax_NO.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
    ax_NE.text(0.04, 0.925,'e',ha='center', va='center', transform=ax_NE.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
    ax_GrIS.text(0.04, 0.935,' ',ha='center', va='center', transform=ax_GrIS.transAxes,weight='bold',fontsize=8,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),zorder=10)
    ax_GrIS.text(0.04, 0.925,'f',ha='center', va='center', transform=ax_GrIS.transAxes,weight='bold',fontsize=15,color='black',zorder=10)
    ### Finalise plot ###
    
    fig_heatmap.tight_layout()
    
    pdb.set_trace()
    '''
    #Save figure
    plt.savefig(path_switchdrive+'RT3/figures/Fig4/v3/Fig4.png',dpi=300,bbox_inches='tight')
    #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    '''
    '''
    #sort df_to_display_normalised
    df_to_display_normalised.sort_values(by=['normalized_raster'],inplace=True)
    #prepare data for fit        
    xdata = np.array(df_to_display_normalised['normalized_raster'])
    ydata = np.array(df_to_display_normalised['20m_ice_content_m'])  
    popt, pcov = curve_fit(deg_n, xdata, ydata,p0=[1.5,-2.5])#,bounds=([0,-1,-4],[2,1,2]))
    ax_GrIS.plot(xdata, deg_n(xdata, *popt))#,'b',label='automatic fit: y = %5.3f*exp(-%5.3f*x)+%5.3f' % tuple(popt))#, 'r-'
    '''
    
    '''
    #Display region normalised SAR thresholds
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['SW']['normalised_SAR'][0],linestyle='dashed',color='green')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['CW']['normalised_SAR'][0],linestyle='dashed',color='green')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['NW']['normalised_SAR'][0],linestyle='dashed',color='green')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['NO']['normalised_SAR'][0],linestyle='dashed',color='green')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['NE']['normalised_SAR'][0],linestyle='dashed',color='green')

    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['SW']['normalised_SAR'][1],linestyle='dashed',color='red')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['CW']['normalised_SAR'][1],linestyle='dashed',color='red')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['NW']['normalised_SAR'][1],linestyle='dashed',color='red')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['NO']['normalised_SAR'][1],linestyle='dashed',color='red')
    ax_GrIS.axvline(x=thresholds_dictionnary_in_func['NE']['normalised_SAR'][1],linestyle='dashed',color='red')
    '''
    
    return

def aquitard_identification(df_for_aquitard,region,LowCutoff,HighCutoff):
        
    #Select region
    df_for_aquitard_region=df_for_aquitard[df_for_aquitard.SUBREGION1==region].copy()
    #below low cutoff = aquitard
    df_for_aquitard_region.loc[df_for_aquitard_region.raster_values<LowCutoff,'aquitard']='1'
    #above high cutoff = non aquitard
    df_for_aquitard_region.loc[df_for_aquitard_region.raster_values>HighCutoff,'aquitard']='0'
    #in betwwen the two cutoss, half aquitard
    df_for_aquitard_region.loc[df_for_aquitard_region.aquitard.isna(),'aquitard']='0.5'
    
    #Display sample size, coeff of variation, and if the difference between below and above is statistically different
    print(region)
    print('-> Sample size:')
    print('   ',region,': ',len(df_for_aquitard_region))
    print('      0: ',len(df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']))
    #print('      0.5: ',len(df_for_aquitard_region[df_for_aquitard_region.aquitard=='0.5']))
    print('      1: ',len(df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']))
    
    print('-> Median:')
    print('      0:',np.round(df_for_aquitard_region[df_for_aquitard_region.aquitard=="0"]['20m_ice_content_m'].median(),1))
    print('      1:',np.round(df_for_aquitard_region[df_for_aquitard_region.aquitard=="1"]['20m_ice_content_m'].median(),1))
    
    print('-> MAD:')
    print('      0: ',np.round(stats.median_abs_deviation(df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']['20m_ice_content_m'],nan_policy='omit'),2))
    print('      1: ',np.round(stats.median_abs_deviation(df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']['20m_ice_content_m'],nan_policy='omit'),2))

    
    print('-> Coefficient of variation:')
    print('      0: ',np.round(df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']['20m_ice_content_m'].std()/df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']['20m_ice_content_m'].mean(),4))
    #print('      0.5: ',np.round(df_for_aquitard_region[df_for_aquitard_region.aquitard=='0.5']['20m_ice_content_m'].std()/df_for_aquitard_region[df_for_aquitard_region.aquitard=='0.5']['20m_ice_content_m'].mean(),4))
    print('      1: ',np.round(df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']['20m_ice_content_m'].std()/df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']['20m_ice_content_m'].mean(),4))
    
    print('-> MAD/median:')
    print('      0: ',np.round(stats.median_abs_deviation(df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']['20m_ice_content_m'],nan_policy='omit')/df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']['20m_ice_content_m'].median(),2))
    print('      1: ',np.round(stats.median_abs_deviation(df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']['20m_ice_content_m'],nan_policy='omit')/df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']['20m_ice_content_m'].median(),2))

    print('-> Test to distritution normality:')
    print('   - null hypothesis: x comes from a normal distribution')
    print('      0: ',stats.normaltest(df_for_aquitard_region[df_for_aquitard_region.aquitard=="0"]['20m_ice_content_m']))
    print('      1: ',stats.normaltest(df_for_aquitard_region[df_for_aquitard_region.aquitard=="1"]['20m_ice_content_m']))

    print('-> Perform Welsch t-test Aquitard VS Non_aquitard:')     
    aquitard_to_test=df_for_aquitard_region[df_for_aquitard_region.aquitard=='1']['20m_ice_content_m'].copy()
    aquitard_to_test=aquitard_to_test[~aquitard_to_test.isna()]
    NonAquitard_to_test=df_for_aquitard_region[df_for_aquitard_region.aquitard=='0']['20m_ice_content_m'].copy()
    NonAquitard_to_test=NonAquitard_to_test[~NonAquitard_to_test.isna()]
    
    #Perform a Welsch's t test when we have no normality, no equal variance
    #Perform a Yuen's t test when we have no normality, no equal variance, and tailed distribution - https://www.youtube.com/watch?v=D_dZyUpgGkI
    #According to scipy doc, "Trimming is recommended if the underlying distribution is long-tailed or contaminated with outliers"
    #Should I perform a Yuen's t-test? If yes, ass in the ttest_ind() the arguemtn trim = 0.x, where x represents the percentage of data in the extremetiy of the distribution to be excluded
    print('  ',stats.ttest_ind(aquitard_to_test,NonAquitard_to_test,equal_var=False))#, trim=.1))#If negative t statistic return, this means the mean of above is less than the mean of below. If p < alpha (alpha being the significance level), then the difference between the two distributions is significantly different at the alpha level.
    print('\n')
    
    return df_for_aquitard_region.copy()


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
import os
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy import stats

#Choose if we want to generate Fig. 4 or not
generating_Fig_SAR_IceThickness = 'TRUE' #Can be TRUE or FALSE

composite='TRUE'
radius=500
SAR_quantiles_extraction='FALSE'#If it is desired to extract the SAR quantiles in the different sectors of different regions

#Define projection
###################### From Tedstone et al., 2022 #####################
#from plot_map_decadal_change.py
# Define the CartoPy CRS object.
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
# This can be converted into a `proj4` string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
###################### From Tedstone et al., 2022 #####################


#Define palette for time , this is From Fig3.py from paper 'Greenland Ice slabs Expansion and Thicknening'
#This is from https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn
my_pal = {'Within': "#DC3220", 'Above': "#6DC11C", 'Below': "#005AB5"}

#Generate boxplot and distributions using 2012, 2016 and 2019 as one population
path_data='C:/Users/jullienn/Documents/working_environment/IceSlabs_SurfaceRunoff/SAR_and_IceThickness/'
path_switchdrive='C:/Users/jullienn/switchdrive/Private/research/'

#Load IMBIE drainage bassins
path_rignotetal2016_GrIS_drainage_bassins=path_switchdrive+'/backup_Aglaja/working_environment/greenland_topo_data/GRE_Basins_IMBIE2_v1.3/'
GrIS_drainage_bassins=gpd.read_file(path_rignotetal2016_GrIS_drainage_bassins+'GRE_Basins_IMBIE2_v1.3_EPSG_3413.shp')

###############################################################################
###         Load Ice Slabs Thickness data in the different sectors          ###
###############################################################################
#Define empty dataframe
IceThickness_above=pd.DataFrame()
IceThickness_in_between=pd.DataFrame()
IceThickness_within=pd.DataFrame()
IceThickness_below=pd.DataFrame()

for indiv_box in range(4,32):
    print(indiv_box)
    #open above
    try:
        above = pd.read_csv(path_data+'SAR_sectors/above/IceSlabs_above_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(above)>0):
            #Append data
            IceThickness_above=pd.concat([IceThickness_above,above])
    except FileNotFoundError:
        print('No above')
    
    #open InBetween
    try:
        in_between = pd.read_csv(path_data+'SAR_sectors/in_between/IceSlabs_in_between_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(in_between)>0):
            #Append data
            IceThickness_in_between=pd.concat([IceThickness_in_between,in_between])
    except FileNotFoundError:
        print('No in_between')
    
    #open within
    try:
        within = pd.read_csv(path_data+'SAR_sectors/within/IceSlabs_within_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(within)>0):
            #Append data
            IceThickness_within=pd.concat([IceThickness_within,within])
    except FileNotFoundError:
        print('No within')
    
    #open below
    try:
        below = pd.read_csv(path_data+'SAR_sectors/below/IceSlabs_below_box_'+str(indiv_box)+'_year_2019.csv')
        if (len(below)>0):
            #Append data
            IceThickness_below=pd.concat([IceThickness_below,below])
    except FileNotFoundError:
        print('No below')
###############################################################################
###         Load Ice Slabs Thickness data in the different sectors          ###
###############################################################################

###############################################################################
###         Plot Ice Slabs Thickness data in the different sectors          ###
###############################################################################
######################## Plot with 0m thick ice slabs #########################
#Display ice slabs distributions as a function of the regions - This is fromEmax_SLabsThickness.py
#Prepare plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(15, 10)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
axNW = plt.subplot(gs[0:5, 0:5])
axCW = plt.subplot(gs[5:10, 0:5])
axSW = plt.subplot(gs[10:15, 0:5])
axNO = plt.subplot(gs[0:5, 5:10])
axNE = plt.subplot(gs[5:10, 5:10])
axGrIS = plt.subplot(gs[10:15, 5:10])

#Plot histograms
plot_histo(axNW,IceThickness_above,IceThickness_within,IceThickness_below,'NW')
plot_histo(axCW,IceThickness_above,IceThickness_within,IceThickness_below,'CW')
plot_histo(axSW,IceThickness_above,IceThickness_within,IceThickness_below,'SW')
plot_histo(axNO,IceThickness_above,IceThickness_within,IceThickness_below,'NO')
plot_histo(axNE,IceThickness_above,IceThickness_within,IceThickness_below,'NE')
plot_histo(axGrIS,IceThickness_above,IceThickness_within,IceThickness_below,'GrIS')

#Finalise plot
axSW.set_xlabel('Ice Thickness [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2019 - 2 years running slabs')
plt.show()
'''
#Save the figure
plt.savefig(path_data+'SAR_sectors/Composite2019_Histo_IceSlabsThickness_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_with0mslabs.png',dpi=500)
'''

#Display as boxplots
#Aggregate data together
IceThickness_above['type']=['Above']*len(IceThickness_above)
IceThickness_in_between['type']=['In_Between']*len(IceThickness_in_between)
IceThickness_within['type']=['Within']*len(IceThickness_within)
IceThickness_below['type']=['Below']*len(IceThickness_below)
IceThickness_all_sectors=pd.concat([IceThickness_above,IceThickness_within,IceThickness_below])

IceThickness_all_sectors_GrIS=IceThickness_all_sectors.copy(deep=True)
IceThickness_all_sectors_GrIS['key_shp']=['GrIS']*len(IceThickness_all_sectors)
IceThickness_all_sectors_region_GrIS=pd.concat([IceThickness_all_sectors,IceThickness_all_sectors_GrIS])

#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.boxplot(data=IceThickness_all_sectors_region_GrIS, x="20m_ice_content_m", y="key_shp",hue="type",orient="h",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_ylabel('')
ax_regions_GrIS.set_xlabel('Ice Thickness [m]')
ax_regions_GrIS.set_xlim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle('2019 - 2 years running slabs')
'''
#Save the figure
plt.savefig(path_data+'SAR_sectors/Composite2019_Boxplot_IceSlabsThickness_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_with0mslabs.png',dpi=500)
'''
######################## Plot with 0m thick ice slabs #########################

#pdb.set_trace()

####################### Plot without 0m thick ice slabs #######################
#Display ice slabs distributions as a function of the regions without 0m thick ice slabs
#Prepare plot
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(15, 10)
#projection set up from https://stackoverflow.com/questions/33942233/how-do-i-change-matplotlibs-subplot-projection-of-an-existing-axis
axNW = plt.subplot(gs[0:5, 0:5])
axCW = plt.subplot(gs[5:10, 0:5])
axSW = plt.subplot(gs[10:15, 0:5])
axNO = plt.subplot(gs[0:5, 5:10])
axNE = plt.subplot(gs[5:10, 5:10])
axGrIS = plt.subplot(gs[10:15, 5:10])

#Plot histograms
plot_histo(axNW,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'NW')
plot_histo(axCW,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'CW')
plot_histo(axSW,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'SW')
plot_histo(axNO,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'NO')
plot_histo(axNE,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'NE')
plot_histo(axGrIS,
           IceThickness_above[IceThickness_above['20m_ice_content_m']>0],
           IceThickness_within[IceThickness_within['20m_ice_content_m']>0],
           IceThickness_below[IceThickness_below['20m_ice_content_m']>0],
           'GrIS')

#Finalise plot
axSW.set_xlabel('Ice content [m]')
axSW.set_ylabel('Density [ ]')
fig.suptitle('2019 - 2 years running slabs - 0m thick slabs excluded')
plt.show()
'''
#Save the figure
plt.savefig(path_data+'SAR_sectors/Composite2019_Histo_IceSlabsThickness_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_without0mslabs.png',dpi=500)
'''
#Display
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 6)
ax_regions_GrIS = plt.subplot(gs[0:10, 0:6])
box_plot_regions_GrIS=sns.boxplot(data=IceThickness_all_sectors_region_GrIS[IceThickness_all_sectors_region_GrIS['20m_ice_content_m']>0], x="20m_ice_content_m", y="key_shp",hue="type",orient="h",ax=ax_regions_GrIS,palette=my_pal)#, kde=True)
ax_regions_GrIS.set_ylabel('')
ax_regions_GrIS.set_xlabel('Ice content [m]')
ax_regions_GrIS.set_xlim(-0.5,20)
ax_regions_GrIS.legend(loc='lower right')
fig.suptitle('2019 - 2 years running slabs - 0m thick slabs excluded')
'''
#Save the figure
plt.savefig(path_data+'SAR_sectors/Composite2019_Boxplot_IceSlabsThickness_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_without0mslabs.png',dpi=500)
'''
####################### Plot without 0m thick ice slabs #######################

###############################################################################
###         Plot Ice Slabs Thickness data in the different sectors          ###
###############################################################################

#pdb.set_trace()

###############################################################################
###                                   SAR                                   ###
###############################################################################
############################# Sectors - 2019 MVRL #############################

if (SAR_quantiles_extraction == 'TRUE'):
    
    #Create dataframe to associate each box with its region
    box_and_region=pd.DataFrame(data={'box_nb': np.arange(1,32),'region': np.nan})
    box_and_region['region'].iloc[0:9]='SW'
    box_and_region['region'].iloc[9]='shared'
    box_and_region['region'].iloc[10:14]='CW'
    box_and_region['region'].iloc[14]='shared'
    box_and_region['region'].iloc[15:21]='NW'
    box_and_region['region'].iloc[21]='shared'
    box_and_region['region'].iloc[22:28]='NO'
    box_and_region['region'].iloc[28]='shared'
    box_and_region['region'].iloc[29:32]='NE'
    
    above_all=pd.DataFrame()
    within_all=pd.DataFrame()
    below_all=pd.DataFrame()
    in_between_all=pd.DataFrame()
    
    #Load SAR csv files
    for indiv_box in range(4,32):
        print(indiv_box)
        #open above
        try:
            above = pd.read_csv(path_data+'SAR_sectors/above/SAR_above_box_'+str(indiv_box)+'_year_2019.csv')
            #drop index column
            above=above.drop(columns=["Unnamed: 0"])
            above['sector']=pd.Series(['Above']*len(above))
            above['box_nb']=pd.Series([indiv_box]*len(above))
            above['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(above))
            #Append data
            above_all=pd.concat([above_all,above])
        except FileNotFoundError:
            print('No above')
        
        #open in_between
        try:
            in_between = pd.read_csv(path_data+'SAR_sectors/in_between/SAR_in_between_box_'+str(indiv_box)+'_year_2019.csv')
            #drop index column
            in_between=in_between.drop(columns=["Unnamed: 0"])
            in_between['sector']=pd.Series(['InBetween']*len(in_between))
            in_between['box_nb']=pd.Series([indiv_box]*len(in_between))
            in_between['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(in_between))
            #Append data
            in_between_all=pd.concat([in_between_all,in_between])
        except FileNotFoundError:
            print('No in_between')
            
        #open within
        try:
            within = pd.read_csv(path_data+'SAR_sectors/within/SAR_within_box_'+str(indiv_box)+'_year_2019.csv')
            #drop index column
            within=within.drop(columns=["Unnamed: 0"])
            within['sector']=pd.Series(['Within']*len(within))
            within['box_nb']=pd.Series([indiv_box]*len(within))
            within['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(within))
            #Append data
            within_all=pd.concat([within_all,within])
        except FileNotFoundError:
            print('No within')
        
        #open below
        try:
            below = pd.read_csv(path_data+'SAR_sectors/below/SAR_below_box_'+str(indiv_box)+'_year_2019.csv')
            #drop index column
            below=below.drop(columns=["Unnamed: 0"])
            below['sector']=pd.Series(['Below']*len(below))
            below['box_nb']=pd.Series([indiv_box]*len(below))
            below['region']=pd.Series([box_and_region.iloc[indiv_box-1]['region']]*len(below))
            #Append data
            below_all=pd.concat([below_all,below])
        except FileNotFoundError:
            print('No below')
    
    pdb.set_trace()
    ### For boxes which share different regions, perform intersection with GrIS drainage bassins ###
    #Reunite all the sectors into one single dataframe
    SAR_all_sectors=pd.concat([above_all,in_between_all,within_all,below_all])
    #Reset index to have a single index per data point
    SAR_all_sectors=SAR_all_sectors.reset_index(drop=True)
    #Identify index which have the 'shared' region
    index_shared=SAR_all_sectors[SAR_all_sectors.region=='shared'].index
    #Select data in boxes sharing 2 regions
    SAR_all_sectors_shared=SAR_all_sectors.loc[index_shared].copy()
    #Transform SAR_all_sectors_shared into a geopandas dataframe
    SAR_all_sectors_shared_gdp = gpd.GeoDataFrame(SAR_all_sectors_shared,
                                                  geometry=gpd.GeoSeries.from_xy(SAR_all_sectors_shared['x_coord_SAR'],
                                                                                 SAR_all_sectors_shared['y_coord_SAR'],
                                                                                 crs='EPSG:3413'))
    #Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
    SAR_all_sectors_shared_gdp_with_regions = gpd.sjoin(SAR_all_sectors_shared_gdp, GrIS_drainage_bassins, predicate='within')
    #Drop index of drainage bassins and the 'region' colum storing 'shared'
    SAR_all_sectors_shared_gdp_with_regions=SAR_all_sectors_shared_gdp_with_regions.drop(columns=["region","index_right"])
    SAR_all_sectors_shared_gdp_with_regions=SAR_all_sectors_shared_gdp_with_regions.rename(columns={"SUBREGION1":"region"})
    #fill in the SAR_all_sectors dataframe the identified regions
    SAR_all_sectors.loc[index_shared,'region']=SAR_all_sectors_shared_gdp_with_regions.region
    
    pdb.set_trace()
    
    '''
    #Make sure identifcation went well - yes, it performs perfectly! Rechecked on May 31, all good :)
    fig = plt.figure()
    gs = gridspec.GridSpec(10, 6)
    ax_region_check = plt.subplot(gs[0:10, 0:6],projection=crs)
    
    GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='SW'].plot(ax=ax_region_check,color='red',alpha=0.2)
    GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='CW'].plot(ax=ax_region_check,color='magenta',alpha=0.2)
    GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NW'].plot(ax=ax_region_check,color='green',alpha=0.2)
    GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NO'].plot(ax=ax_region_check,color='blue',alpha=0.2)
    GrIS_drainage_bassins[GrIS_drainage_bassins.SUBREGION1=='NE'].plot(ax=ax_region_check,color='cyan',alpha=0.2)
    
    ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='SW'].x_coord_SAR,
                            SAR_all_sectors[SAR_all_sectors.region=='SW'].y_coord_SAR,
                            color='red') #Perfect!
    ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='CW'].x_coord_SAR,
                            SAR_all_sectors[SAR_all_sectors.region=='CW'].y_coord_SAR,
                            color='magenta') #Perfect!
    ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='NW'].x_coord_SAR,
                            SAR_all_sectors[SAR_all_sectors.region=='NW'].y_coord_SAR,
                            color='green')
    ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='NO'].x_coord_SAR,
                            SAR_all_sectors[SAR_all_sectors.region=='NO'].y_coord_SAR,
                            color='blue') #Perfect!
    ax_region_check.scatter(SAR_all_sectors[SAR_all_sectors.region=='NE'].x_coord_SAR,
                            SAR_all_sectors[SAR_all_sectors.region=='NE'].y_coord_SAR,
                            color='cyan')   
    '''
    
    #Display figure distribution
    fig, (ax_distrib) = plt.subplots()      
    ax_distrib.hist(SAR_all_sectors[SAR_all_sectors.sector=='Below'].SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='green',label='Below')
    #ax_distrib.hist(in_between_all.SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='yellow',label='In Between')
    #ax_distrib.hist(within_all.SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='red',label='Within')
    ax_distrib.hist(SAR_all_sectors[SAR_all_sectors.sector=='Above'].SAR,density=True,alpha=0.5,bins=np.arange(-21,1,0.5),color='blue',label='Above')
    ax_distrib.set_xlim(-20,-2)
    ax_distrib.set_xlabel('Signal strength [dB]')
    ax_distrib.set_ylabel('Density')
    ax_distrib.legend()
    ax_distrib.set_title('GrIS-wide')
    '''
    #Save the figure
    plt.savefig(path_data+'SAR_sectors/Composite2019_HistoGrIS_SAR_cleanedxytpdV3.png',dpi=500)
    '''
    
    print('--- SAR ---')
    #Display figure distribution in different regions
    fig, (ax_distrib_SW,ax_distrib_CW,ax_distrib_NW,ax_distrib_NO,ax_distrib_NE) = plt.subplots(5,1)  
    hist_regions(SAR_all_sectors[SAR_all_sectors.region=='SW'],'SW',ax_distrib_SW)
    hist_regions(SAR_all_sectors[SAR_all_sectors.region=='CW'],'CW',ax_distrib_CW)
    hist_regions(SAR_all_sectors[SAR_all_sectors.region=='NW'],'NW',ax_distrib_NW)
    hist_regions(SAR_all_sectors[SAR_all_sectors.region=='NO'],'NO',ax_distrib_NO)
    hist_regions(SAR_all_sectors[SAR_all_sectors.region=='NE'],'NE',ax_distrib_NE)
    ax_distrib_SW.legend()
    fig.suptitle('Regional separation')
    '''
    #Save the figure
    plt.savefig(path_data+'SAR_sectors/Composite2019_HistoRegions_SAR_cleanedxytpdV3.png',dpi=500)
    '''
    #Display boxplot
    #GrIS-wide
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(10, 6)
    ax_SAR = plt.subplot(gs[0:10, 0:6])
    sns.boxplot(data=SAR_all_sectors[np.logical_or((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.sector=='Below'))], y="sector", x="SAR",ax=ax_SAR)#, kde=True)
    ax_SAR.set_xlabel('Signal strength [dB]')
    ax_SAR.set_ylabel('Category')
    ax_SAR.set_title('GrIS-wide')
    '''
    #Save the figure
    plt.savefig(path_data+'SAR_sectors/Composite2019_BoxplotGrIS_SAR_cleanedxytpdV3.png',dpi=500)
    '''
    
    #Regions
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(10, 6)
    ax_SAR = plt.subplot(gs[0:10, 0:6])
    sns.boxplot(data=SAR_all_sectors[np.logical_or((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.sector=='Below'))], y="region", x="SAR",hue="sector",ax=ax_SAR)#, kde=True)
    ax_SAR.set_xlabel('Signal strength [dB]')
    ax_SAR.set_ylabel('Category')
    ax_SAR.set_title('GrIS-wide')
    '''
    #Save the figure
    plt.savefig(path_data+'SAR_sectors/Composite2019_BoxplotRegions_SAR_cleanedxytpdV3.png',dpi=500)
    '''
    #Violin plot
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(10, 6)
    ax_SAR = plt.subplot(gs[0:10, 0:6])
    sns.violinplot(data=pd.DataFrame(SAR_all_sectors[np.logical_or((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.sector=='Below'))].to_dict()),
                   y="region", x="SAR",hue="sector",ax=ax_SAR,palette=my_pal)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
    ax_SAR.set_xlabel('Signal strength [dB]')
    ax_SAR.set_ylabel('Region')
    ax_SAR.set_title('GrIS-wide')
    '''
    #Save the figure
    plt.savefig(path_data+'SAR_sectors/Composite2019_ViolinPlotRegions_SAR_AboveBelow_cleanedxytpdV3.png',dpi=500)
    '''
    
    #Display above, within and below violin plot!
    df_except_InBetween=SAR_all_sectors.drop(SAR_all_sectors[SAR_all_sectors.sector=='InBetween'].index.to_numpy()).copy()
    #Violin plot
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(10, 6)
    ax_SAR = plt.subplot(gs[0:10, 0:6])
    sns.violinplot(data=pd.DataFrame(df_except_InBetween.to_dict()),
                   y="region", x="SAR",hue="sector",ax=ax_SAR,palette=my_pal)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
    ax_SAR.set_xlabel('Signal strength [dB]')
    ax_SAR.set_ylabel('Region')
    ax_SAR.set_title('GrIS-wide')
    '''
    #Save the figure
    plt.savefig(path_data+'SAR_sectors/Composite2019_ViolinPlotRegions_SAR_cleanedxytpdV3.png',dpi=500)
    '''
    ############################# Sectors - 2019 MVRL #############################
    pdb.set_trace()
    
    #Display SAR sectorial summary statistics
    print('--- SW ---')
    print('- Above')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.region=='SW'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Within')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Within'),(SAR_all_sectors.region=='SW'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Below')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Below'),(SAR_all_sectors.region=='SW'))].SAR.quantile([0.25,0.5,0.75]))
    
    print('--- CW ---')
    print('- Above')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.region=='CW'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Within')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Within'),(SAR_all_sectors.region=='CW'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Below')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Below'),(SAR_all_sectors.region=='CW'))].SAR.quantile([0.25,0.5,0.75]))
    
    print('--- NW ---')
    print('- Above')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.region=='NW'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Within')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Within'),(SAR_all_sectors.region=='NW'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Below')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Below'),(SAR_all_sectors.region=='NW'))].SAR.quantile([0.25,0.5,0.75]))
    
    print('--- NO ---')
    print('- Above')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.region=='NO'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Within')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Within'),(SAR_all_sectors.region=='NO'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Below')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Below'),(SAR_all_sectors.region=='NO'))].SAR.quantile([0.25,0.5,0.75]))
    
    print('--- NE ---')
    print('- Above')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Above'),(SAR_all_sectors.region=='NE'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Within')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Within'),(SAR_all_sectors.region=='NE'))].SAR.quantile([0.25,0.5,0.75]))
    print('- Below')
    print(SAR_all_sectors[np.logical_and((SAR_all_sectors.sector=='Below'),(SAR_all_sectors.region=='NE'))].SAR.quantile([0.25,0.5,0.75]))
    
    pdb.set_trace()
    
    #Display ice thickness and SAR for each regions for each sector as violin plot aside each other
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(5, 10)
    gs.update(hspace=0)
    gs.update(wspace=0.1)
    ax_SAR = plt.subplot(gs[0:5,5:10])
    sns.violinplot(data=pd.DataFrame(df_except_InBetween.to_dict()), x="SAR", y="region",hue="sector",orient="h",scale="width",ax=ax_SAR,palette=my_pal,cut=0)#, kde=True)
    ax_SAR.set_xlabel('Signal strength [dB]',labelpad=10)
    ax_SAR.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=False, right=True, labelright=True)
    ax_SAR.yaxis.set_label_position('right')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
    ax_SAR.get_legend().remove()#from https://stackoverflow.com/questions/5735208/remove-the-legend-on-a-matplotlib-figure
    ax_SAR.set_ylabel('Region',labelpad=10)
    ax_SAR.grid(linestyle='dashed')
    ax_SAR.text(0.03, 0.97,'b',ha='center', va='center', transform=ax_SAR.transAxes,weight='bold',fontsize=20,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    
    #Display actually only one plot
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(7,10))
    gs = gridspec.GridSpec(5, 10)
    ax_ice_thickness = plt.subplot(gs[0:5, 0:10])
    sns.violinplot(data=IceThickness_all_sectors, x="20m_ice_content_m", y="key_shp",hue="type",orient="h",scale="width",ax=ax_ice_thickness,palette=my_pal,cut=0)#, kde=True)
    ax_ice_thickness.set_xlabel('Ice slab thickness [m]',labelpad=10)
    ax_ice_thickness.set_ylabel('Region',labelpad=10)
    ax_ice_thickness.grid(linestyle='dashed')
    #ax_ice_thickness.text(0.03, 0.97,'a',ha='center', va='center', transform=ax_ice_thickness.transAxes,weight='bold',fontsize=20,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot
    
    #Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
    legend_elements = [Patch(facecolor=my_pal['Above'],edgecolor='black',label='Above'),
                       Patch(facecolor=my_pal['Within'],edgecolor='black',label='At'),
                       Patch(facecolor=my_pal['Below'],edgecolor='black',label='Below')]
    ax_ice_thickness.legend(handles=legend_elements,loc='lower right',fontsize=15,framealpha=0.8).set_zorder(7)
    
    '''
    #Save the figure
    plt.savefig(path_switchdrive+'RT3/figures/Fig2/v3/Fig2.png',dpi=300,bbox_inches='tight')
    #bbox_inches is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
    '''

'''
Results of the quantiles in the different regions for the different sectors
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

#Create a threshold dictionnary
thresholds_dictionnary={'SW':{'SAR':[-9.110144,-8.638897],'SAR_normalised':[]},
                        'CW':{'SAR':[-7.82364,-7.038067],'SAR_normalised':[]},
                        'NW':{'SAR':[-8.982688,-8.266375],'SAR_normalised':[]},
                        'NO':{'SAR':[-7.194321,-6.104299],'SAR_normalised':[]},
                        'NE':{'SAR':[-6.329797,-5.715943],'SAR_normalised':[]}
                        }

###############################################################################
###                                   SAR                                   ###
###############################################################################

pdb.set_trace()

###############################################################################
###                          SAR and Ice Thickness                          ###
###############################################################################

########### Load ice slabs with SAR dataset and identify the sector ###########
#Path to data
path_SAR_And_IceThickness=path_data+'csv/NotClipped_With0mSlabs/'
#List all the files in the folder
list_composite=os.listdir(path_SAR_And_IceThickness) #this is inspired from https://pynative.com/python-list-files-in-a-directory/

#Load ice thickness and SAR at FS
FS_pd=pd.DataFrame(data={'Station': ['FS2', 'FS4', 'FS5'], '10m_ice_content_%': [95.06, 56.50, 38.44], 'SAR': [-11.37, -6.58, -5.42]})

#Define empty dataframe
upsampled_SAR_and_IceSlabs=pd.DataFrame()
upsampled_SAR_and_IceSlabs_above=pd.DataFrame()
upsampled_SAR_and_IceSlabs_in_between=pd.DataFrame()
upsampled_SAR_and_IceSlabs_within=pd.DataFrame()
upsampled_SAR_and_IceSlabs_below=pd.DataFrame()

#The join betwwen the ice slabs and SAR dataset with the ice slabs with sector dataset is now performed using both datasets having as native ice slabs data the one containing 0 m thick slabs!

#Loop over all the files
for indiv_file in list_composite:
    print(indiv_file)
    
    #Open the individual file
    indiv_csv=pd.read_csv(path_SAR_And_IceThickness+indiv_file)
    
    if (generating_Fig_SAR_IceThickness == 'TRUE'):
        #If ice content is larger than 16 m thick, set it to 16 m
        indiv_csv.loc[indiv_csv["20m_ice_content_m"]>16,"20m_ice_content_m"]=16
        
    ### ALL ###
    #Upsample data: where index_right is identical (i.e. for each SAR cell), keep a single value of radar signal and average the ice content
    indiv_upsampled_SAR_and_IceSlabs=indiv_csv.groupby('index_right').mean()  
    #Append the data to each other
    upsampled_SAR_and_IceSlabs=pd.concat([upsampled_SAR_and_IceSlabs,indiv_upsampled_SAR_and_IceSlabs])
    ### ALL ###
    
    ### SECTORS ###
    #In the sectorial dataframes, keep only data corresponding to the current TrackName
    indiv_IceThickness_above=keep_sectorial(IceThickness_above,indiv_csv.Track_name.unique()[0])
    indiv_IceThickness_in_between=keep_sectorial(IceThickness_in_between,indiv_csv.Track_name.unique()[0])
    indiv_IceThickness_within=keep_sectorial(IceThickness_within,indiv_csv.Track_name.unique()[0])
    indiv_IceThickness_below=keep_sectorial(IceThickness_below,indiv_csv.Track_name.unique()[0])
    
    #Prepare figure to display
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 5)
    ax_check_csv_sectors = plt.subplot(gs[0:5, 0:5],projection=crs)
    ax_check_csv_sectors.scatter(indiv_csv.lon_3413,indiv_csv.lat_3413,s=5,color='black')
    ax_check_csv_sectors.scatter(indiv_IceThickness_above.lon_3413,indiv_IceThickness_above.lat_3413,s=1,color='blue')
    ax_check_csv_sectors.scatter(indiv_IceThickness_in_between.lon_3413,indiv_IceThickness_in_between.lat_3413,s=1,color='yellow')
    ax_check_csv_sectors.scatter(indiv_IceThickness_within.lon_3413,indiv_IceThickness_within.lat_3413,s=1,color='red')
    ax_check_csv_sectors.scatter(indiv_IceThickness_below.lon_3413,indiv_IceThickness_below.lat_3413,s=1,color='green')
        
    #Associate the sector to the dataframe where ice thickness and SAR data are present by joining the two following dataframes
    #indiv_csv is the dataframe holding ice content and SAR signal NOT upsampled, but no info on the sector
    #indiv_IceThickness_above/in_between/within/below are the dataframe holding the ice content in the sector NOT upsampled, but no info on SAR.
    indiv_upsampled_SAR_and_IceSlabs_above=sector_association(indiv_csv,indiv_IceThickness_above,'above')
    indiv_upsampled_SAR_and_IceSlabs_in_between=sector_association(indiv_csv,indiv_IceThickness_in_between,'InBetween')
    indiv_upsampled_SAR_and_IceSlabs_within=sector_association(indiv_csv,indiv_IceThickness_within,'within')
    indiv_upsampled_SAR_and_IceSlabs_below=sector_association(indiv_csv,indiv_IceThickness_below,'below')
    
    #Append data to obtain one dataframe per sector
    if (len(indiv_upsampled_SAR_and_IceSlabs_above)>0):
        upsampled_SAR_and_IceSlabs_above=pd.concat([upsampled_SAR_and_IceSlabs_above,indiv_upsampled_SAR_and_IceSlabs_above])
    
    if (len(indiv_upsampled_SAR_and_IceSlabs_in_between)>0):
        upsampled_SAR_and_IceSlabs_in_between=pd.concat([upsampled_SAR_and_IceSlabs_in_between,indiv_upsampled_SAR_and_IceSlabs_in_between])
        
    if (len(indiv_upsampled_SAR_and_IceSlabs_within)>0):
        upsampled_SAR_and_IceSlabs_within=pd.concat([upsampled_SAR_and_IceSlabs_within,indiv_upsampled_SAR_and_IceSlabs_within])
        
    if (len(indiv_upsampled_SAR_and_IceSlabs_below)>0):
        upsampled_SAR_and_IceSlabs_below=pd.concat([upsampled_SAR_and_IceSlabs_below,indiv_upsampled_SAR_and_IceSlabs_below])
    #pdb.set_trace()
    
    plt.close()
    ### SECTORS ###
########### Load ice slabs with SAR dataset and identify the sector ###########

################### Relationship using data in sectors only ###################
#Append data to each other
upsampled_SAR_and_IceSlabs_allsectors=pd.concat([upsampled_SAR_and_IceSlabs_above,upsampled_SAR_and_IceSlabs_in_between,upsampled_SAR_and_IceSlabs_within,upsampled_SAR_and_IceSlabs_below])
#Get rid of NaNs
upsampled_SAR_and_IceSlabs_allsectors_temp=upsampled_SAR_and_IceSlabs_allsectors[~upsampled_SAR_and_IceSlabs_allsectors.raster_values.isna()].copy()
upsampled_SAR_and_IceSlabs_allsectors_NoNaN=upsampled_SAR_and_IceSlabs_allsectors_temp[~upsampled_SAR_and_IceSlabs_allsectors_temp["20m_ice_content_m"].isna()].copy()

#Transform upsampled_SAR_and_IceSlabs_allsectors_NoNaN as a geopandas dataframe
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp = gpd.GeoDataFrame(upsampled_SAR_and_IceSlabs_allsectors_NoNaN,
                                                                   geometry=gpd.GeoSeries.from_xy(upsampled_SAR_and_IceSlabs_allsectors_NoNaN['lon_3413'],
                                                                                                  upsampled_SAR_and_IceSlabs_allsectors_NoNaN['lat_3413'],
                                                                                                  crs='EPSG:3413'))

#Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions = gpd.sjoin(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp, GrIS_drainage_bassins, predicate='within')
#Display histograms
pdb.set_trace()
display_2d_histogram(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,FS_pd,'sectors',thresholds_dictionnary)
#Conclusion: I think this is hard to derive a relationship between SAR and ice thickness using data in sectors

'''
#Save the figure
plt.savefig(path_data+'Sectors2019_Hist2D_IceSlabsThickness_SAR_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_with0mslabs.png',dpi=500)
'''

pdb.set_trace()

#Perform the comparison aquitard VS non-aquitard in all the sectors
#Apply thresholds to differentiate between efficient aquitard VS non-efficient aquitard places - let's choose quantile 0.75 of below as low cutoff, quantile 0.75 of within for high cutoff
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_SW=aquitard_identification(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,'SW',-9.110144,-8.638897)
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_CW=aquitard_identification(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,'CW',-7.82364,-7.038067)
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_NW=aquitard_identification(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,'NW',-8.982688,-8.266375)
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_NO=aquitard_identification(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,'NO',-7.194321,-6.104299)
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_NE=aquitard_identification(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_with_regions,'NE',-6.329797,-5.715943)
#Aggregate regional dataframes
upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_GrIS=pd.concat([upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_SW,
                                                               upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_CW,
                                                               upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_NW,
                                                               upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_NO,
                                                               upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_NE])

#Get rid of 0.5 aquitard and reindex
final_df_SAR_IceThickness=upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_GrIS[~(upsampled_SAR_and_IceSlabs_allsectors_NoNaN_gdp_GrIS.aquitard=='0.5')].copy()
final_df_SAR_IceThickness['new_index']=np.arange(0,len(final_df_SAR_IceThickness))
final_df_SAR_IceThickness=final_df_SAR_IceThickness.set_index('new_index')

pal_zones = {'0': "#e7f1ff", '0.5': "#db4500", '1': "#0a4aaa"}
# Display SAR violin plot for lateral runoff and retention zones
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(8.27,4.55))#Nature pdf size = (8.27,9)
gs = gridspec.GridSpec(50, 10)
gs.update(hspace=0)
gs.update(wspace=0.1)
ax_filtered_regions_SAR = plt.subplot(gs[0:47, 5:10])
sns.violinplot(data=final_df_SAR_IceThickness,
               y="SUBREGION1", x="raster_values",hue="aquitard",ax=ax_filtered_regions_SAR,scale="width",palette=pal_zones,cut=0)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
ax_filtered_regions_SAR.set_xlabel('$\sigma_{0}^{HV}$ [dB]',labelpad=10)
ax_filtered_regions_SAR.set_ylabel('')
ax_filtered_regions_SAR.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=False, right=True, labelright=True)
ax_filtered_regions_SAR.yaxis.set_label_position('right')#from https://stackoverflow.com/questions/14406214/moving-x-axis-to-the-top-of-a-plot-in-matplotlib
ax_filtered_regions_SAR.grid(linestyle='dashed')
ax_filtered_regions_SAR.get_legend().remove()#from https://stackoverflow.com/questions/5735208/remove-the-legend-on-a-matplotlib-figure
ax_filtered_regions_SAR.text(0.03, 0.95,'g',ha='center', va='center', transform=ax_filtered_regions_SAR.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot


# Display ice thickness violin plot for lateral runoff and retention zones
ax_filtered_regions_IceThickness = plt.subplot(gs[0:47,0:5])
sns.violinplot(data=final_df_SAR_IceThickness,
               y="SUBREGION1", x="20m_ice_content_m",hue="aquitard",ax=ax_filtered_regions_IceThickness,scale="width",palette=pal_zones,cut=0)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
ax_filtered_regions_IceThickness.set_xlabel('Ice slab thickness [m]',labelpad=10)
ax_filtered_regions_IceThickness.set_ylabel('Region',labelpad=10)
ax_filtered_regions_IceThickness.grid(linestyle='dashed')
ax_filtered_regions_IceThickness.get_legend().remove()

#Custom legend myself for ax2 - this is from Fig1.py from paper 'Greenland ice slabs expansion and thickening'        
legend_elements = [Patch(facecolor='#0a4aaa',edgecolor='black',label='Areas supporting runoff'),
                   Patch(facecolor='#e7f1ff',edgecolor='black',label='Retention areas')]
ax_filtered_regions_SAR.legend(handles=legend_elements,loc='lower left',fontsize=10,framealpha=0.8,bbox_to_anchor=(0, 0.135)).set_zorder(7)
ax_filtered_regions_IceThickness.text(0.03, 0.95,'f',ha='center', va='center', transform=ax_filtered_regions_IceThickness.transAxes,weight='bold',fontsize=12,color='black')#This is from https://pretagteam.com/question/putting-text-in-top-left-corner-of-matplotlib-plot

pdb.set_trace()

'''
#Save the figure
plt.savefig(path_switchdrive+'RT3/figures/Fig3/v4/Fig3_fg.png',dpi=300)
'''


fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(5, 5)
ax_hist = plt.subplot(gs[0:5, 0:5])
sns.histplot(final_df_SAR_IceThickness, x="20m_ice_content_m",hue="aquitard",stat='density',kde=True,ax=ax_hist)


#Display the ice thickness distributions
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(10, 10)
ax_aquitard_above = plt.subplot(gs[0:5, 0:5])
ax_aquitard_InBetween = plt.subplot(gs[0:5, 5:10])
ax_aquitard_within = plt.subplot(gs[5:10, 0:5])
ax_aquitard_below = plt.subplot(gs[5:10, 5:10])
gs.update(wspace=1)
gs.update(hspace=1)
sns.violinplot(data=final_df_SAR_IceThickness[final_df_SAR_IceThickness.sector=='above'],
               y="SUBREGION1", x="20m_ice_content_m",hue="aquitard",ax=ax_aquitard_above)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
sns.violinplot(data=final_df_SAR_IceThickness[final_df_SAR_IceThickness.sector=='InBetween'],
               y="SUBREGION1", x="20m_ice_content_m",hue="aquitard",ax=ax_aquitard_InBetween)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
sns.violinplot(data=final_df_SAR_IceThickness[final_df_SAR_IceThickness.sector=='within'],
               y="SUBREGION1", x="20m_ice_content_m",hue="aquitard",ax=ax_aquitard_within)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo
sns.violinplot(data=final_df_SAR_IceThickness[final_df_SAR_IceThickness.sector=='below'],
               y="SUBREGION1", x="20m_ice_content_m",hue="aquitard",ax=ax_aquitard_below)#, kde=True)#Making the display possible using sns.violinplot by helper from https://stackoverflow.com/questions/52284034/categorical-plotting-with-seaborn-raises-valueerror-object-arrays-are-not-suppo

ax_aquitard_above.set_title('Above')
ax_aquitard_InBetween.set_title('InBetween')
ax_aquitard_within.set_title('Within')
ax_aquitard_below.set_title('Below')

ax_aquitard_within.set_xlabel('Ice Thickness [m]')
ax_aquitard_within.set_ylabel('Region')

ax_aquitard_above.set_xlabel('')
ax_aquitard_InBetween.set_xlabel('')
ax_aquitard_below.set_xlabel('')
ax_aquitard_above.set_ylabel('')
ax_aquitard_InBetween.set_ylabel('')
ax_aquitard_below.set_ylabel('')
################### Relationship using data in sectors only ###################

pdb.set_trace()

#CONTINUE WORKING HERE ON THE RELATIONSHIP FOR THE WHOLE DATASET!

############### Relationship using the whole ice slabs dataset ###############
#Create a unique index for each line, and set this new vector as the index in dataframe
upsampled_SAR_and_IceSlabs['index_unique']=np.arange(0,len(upsampled_SAR_and_IceSlabs))
upsampled_SAR_and_IceSlabs=upsampled_SAR_and_IceSlabs.set_index('index_unique')

#Get rid of NaNs
upsampled_SAR_and_IceSlabs_temp=upsampled_SAR_and_IceSlabs[~upsampled_SAR_and_IceSlabs.raster_values.isna()].copy()
upsampled_SAR_and_IceSlabs_NoNaN=upsampled_SAR_and_IceSlabs_temp[~(upsampled_SAR_and_IceSlabs_temp["20m_ice_content_m"].isna())].copy()

#Display some descriptive statistics
upsampled_SAR_and_IceSlabs_NoNaN.describe()['raster_values']
upsampled_SAR_and_IceSlabs_NoNaN.describe()['20m_ice_content_m']

#7. Display the composite relationship using all the files
#Prepare plot
fig = plt.figure()
fig.set_size_inches(14, 8) # set figure's size manually to your full screen (32x18), this is from https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
gs = gridspec.GridSpec(10, 10)
gs.update(wspace=2)
gs.update(hspace=1)
ax_SAR = plt.subplot(gs[0:2, 0:8])
ax_scatter = plt.subplot(gs[2:10, 0:8])
ax_IceContent = plt.subplot(gs[2:10, 8:10])

#Display
upsampled_SAR_and_IceSlabs_NoNaN.plot.scatter(x='raster_values',y='20m_ice_content_m',ax=ax_scatter)
ax_scatter.set_xlim(-18,1)
ax_scatter.set_ylim(-0.5,20.5)
ax_scatter.set_xlabel('SAR [dB]')
ax_scatter.set_ylabel('Ice content [m]')

ax_IceContent.hist(upsampled_SAR_and_IceSlabs_NoNaN['20m_ice_content_m'],
                   bins=np.arange(np.min(upsampled_SAR_and_IceSlabs_NoNaN['20m_ice_content_m']),np.max(upsampled_SAR_and_IceSlabs_NoNaN['20m_ice_content_m'])),
                   density=True,orientation='horizontal')
ax_IceContent.set_xlabel('Density [ ]')
ax_IceContent.set_ylim(-0.5,20.5)

ax_SAR.hist(upsampled_SAR_and_IceSlabs_NoNaN['raster_values'],
            bins=np.arange(np.min(upsampled_SAR_and_IceSlabs_NoNaN['raster_values']),np.max(upsampled_SAR_and_IceSlabs_NoNaN['raster_values'])),
            density=True)
ax_SAR.set_xlim(-18,1)
ax_SAR.set_ylabel('Density [ ]')
fig.suptitle('Ice content and SAR')

#Transform upsampled_SAR_and_IceSlabs as a geopandas dataframe
upsampled_SAR_and_IceSlabs_NoNaN_gdp = gpd.GeoDataFrame(upsampled_SAR_and_IceSlabs_NoNaN,
                                                  geometry=gpd.GeoSeries.from_xy(upsampled_SAR_and_IceSlabs_NoNaN['lon_3413'],
                                                                                 upsampled_SAR_and_IceSlabs_NoNaN['lat_3413'],
                                                                                 crs='EPSG:3413'))

#Intersection between dataframe and poylgon, from https://gis.stackexchange.com/questions/346550/accelerating-geopandas-for-selecting-points-inside-polygon        
upsampled_SAR_and_IceSlabs_NoNaN_gdp_with_regions = gpd.sjoin(upsampled_SAR_and_IceSlabs_NoNaN_gdp, GrIS_drainage_bassins, predicate='within')

#Display histograms
display_2d_histogram(upsampled_SAR_and_IceSlabs_NoNaN_gdp_with_regions,FS_pd,'complete_dataset')
'''
#Save the figure
plt.savefig(path_data+'Composite2019_Hist2D_IceSlabsThickness_SAR_2YearsRunSlabs_radius_'+str(radius)+'m_cleanedxytpdV3_with0mslabs.png',dpi=500)
'''
############### Relationship using the whole ice slabs dataset ###############

###############################################################################
###                          SAR and Ice Thickness                          ###
###############################################################################




print('--- End of code ---')
