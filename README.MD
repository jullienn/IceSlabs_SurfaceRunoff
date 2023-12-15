# Ice slabs thickening drives surface runoff expansion from the Greenland Ice Sheet’s percolation zone

## Introduction

All the codes used for data preparation, processing, analysis and figure generation are present in this folder.
These codes were written by Nicolas Jullien between 2022 and 2023.
The way the codes and folders are now organized in this code repository were not tested, and some parts may fail. Furthermore, path adaptations are mandatory before running any code.
Some codes (that we may or may not have modified) were directly taken from MacFerrin et al., (2019), and Jullien et al., (2023). Some sections in other codes use functions developped by MacFerrin et al., (2019), some others, and Jullien et al., (2023) that may or not have been modified by us.
Feel free to contact the author for any questions (nicolas.jullien@unifr.ch).

The datasets used in this paper, as well as exported data are archived in the data repository related to the paper, accessible at https://zenodo.org/records/10376906

## Codes
### Data processing and figures

* 'AreasSupportingRunoff_And_Fig4abcde.py': Creating the raster files of Areas Supporting Runoff by applying the sigma0HV threshods to the winter composite maps, and generating Fig 4abcde.
* 'Fig6_bcdef.py': Generates Fig 6bcdef.
* 'Extraction_IceThicknessAndSigma0HV_InRLzones.py': Extracting the ice slab thickness and sigma0HV in the different zones in the vicinity of the runoff limit ('below', 'at', 'in-between', 'above' zones).
* 'Extraction_CumHydroAndSigma0HV_InTransects.py': Extracting the occurrence of surface hydrology and sigma0HV at each ice slabs thickness sampling point.
* 'RunoffLimit_Check_Filtering.py': Displaying the cleaned runoff limit retrievals with NDWI map to check filtering went well.
* 'Fig1.py': Performing data processing associated with the generation of Fig. 1, performing the generation of Fig. 1 generation, and performing the analysis for Supp Info 1.
* 'Fig5_And_Fig6agh.py': Generation of Fig. 5, Fig 6a, Fig 6gh.
* 'Sigma0HVThresholds_And_Fig2_And_Fig3_And_Fig4fg.py': Determine the sigma0HV thresholds used to map the areas supporting runoff after performing the aggregation of sigma0HV in the different zones in the vicinity of the runoff limit. Also, performing the generation of Fig2, Fig3, Fig4fg.
* 'Sigma0HV_And_FirnCores.py': Performing the data processing associated with Supp Info 2.


### Codes not used in this study, but kept for continuitprocessing and figures
* 'Sigma0HVUpperLimitExtraction.py': This code ot used in the paper, kept for the future)


* 'Greenland_Hydrology_Summary.py': Displaying a context map of Greenland hydrology summary - not used in paper, but kept for the future.











### Figures

All the figures can be regenerated using the corresponding code.


## License
This code repository is under the GNU General Public License v3.0. If it is used in publication (academic or otherwise), we request the following paper be cited:
N. Jullien, A. J. Tedstone, H. Machguth, "Ice slabs thickening drives surface runoff expansion from the Greenland Ice Sheet’s percolation zone,". Submitted to Nature Communications.