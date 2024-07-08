# TO BE MODIFIED - Ice slabs thickening drives surface runoff expansion from the Greenland Ice Sheet’s percolation zone

## Introduction

All the codes used for data preparation, processing, analysis and figure generation are present in this folder.
These codes were written by Nicolas Jullien between 2022 and 2023.
The way the codes and folders are now organized in this code repository were not tested, and some parts may fail. Furthermore, path adaptations are mandatory before running any code.
Some sections in several codes use functions developped by Jullien et al., (2023), some others by MacFerrin et al., (2019) that may or not have been modified by us.
Feel free to contact the author for any questions (nicolas.jullien@unifr.ch).

The datasets used in this paper, as well as exported data are archived in the data repository related to the paper, accessible at https://zenodo.org/records/10376906

## Codes
### Data processing and figures

* 'ExtractionIceThickness_InRLzones_andFig1.py': Extracting the ice slab thickness (and sigma0HV, out of date) in the different zones in the vicinity of the runoff limit ('below', 'at', 'in-between', 'above' zones). Generating Fig. 1.
* 'Extraction_CumHydroAndSigma0HV_InTransects.py': Extracting the occurrence of surface hydrology (and sigma0HV, out of date) at each ice slabs thickness sampling point.
* 'Fig2.py': Performing data processing associated with the generation of Fig. 2.
* 'Fig3_and_FigS2_and_Sigma0HVThresholds.py': Generation of Fig. 3 and Fig S2. There is codes sections where is performed the determination of the sigma0HV thresholds used to map the areas supporting runoff after performing the aggregation of sigma0HV in the different zones in the vicinity of the runoff limit (legacy code, out of date).
* 'Fig4_and_Fig5agh_and_Fig6.py': Generation of Fig. 4, Fig. 5agh, Fig. 6. 
* 'Fig5bcdef.py': Generates Fig. 5bcdef.
* 'FigS2.py': Generating Fig. S1.
* 'RunoffLimit_Check_Filtering.py': Displaying the cleaned runoff limit retrievals with NDWI map to check filtering went well.

## License
This code repository is under the GNU General Public License v3.0. If it is used in publication (academic or otherwise), we request the following paper be cited:
N. Jullien, A. J. Tedstone, H. Machguth, "Ice slabs thickening drives surface runoff expansion from the Greenland Ice Sheet’s percolation zone,". Submitted to Nature Communications.