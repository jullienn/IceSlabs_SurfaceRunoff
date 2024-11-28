# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calculate Melt-over-Accumulation ratio for Jullien et al. Ice slabs thickness and surface hydrology paper
#
# A.T., November 2024

# %% trusted=true
import xarray as xr
import rioxarray
from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import Client
import os
import matplotlib.pyplot as plt
import geoutils as gu
import numpy as np
import seaborn as sns

cluster = MyCluster()
cluster.scale(jobs=4)
client = Client(cluster)

# %% trusted=true
pth_mar = '/home/geoscience/nobackup_cassandra/MARv.HorstRCMStudy_20240624'
moa_threshold = 0.7


# %% trusted=true
def reorganise_mar(xd):
    xd = xd.rename({'X14_163':'x', 'Y19_288':'y'})
    xd['x'] = xd['x'] * 1000
    xd['y'] = xd['y'] * 1000
    return xd


# %% trusted=true
# Load MAR mask etc
mar_eg = xr.open_dataset(os.path.join(pth_mar, 'ICE.2020.01-12.m60.nc'))
mar_eg = reorganise_mar(mar_eg)

# %% trusted=true
# Load MAR accumulation and melt data
accum = xr.open_mfdataset(os.path.join(pth_mar, '*.nc'), preprocess=lambda x:x.SF)
accum = accum.rio.write_crs('epsg:3413')
accum = reorganise_mar(accum).sel(TIME=slice('2000-01-01','2012-12-31'))
accum = accum.SF.squeeze()
accum = accum.where(mar_eg.MSK > 50)

melt = xr.open_mfdataset(os.path.join(pth_mar, '*.nc'), preprocess=lambda x:x.ME)
melt = melt.rio.write_crs('epsg:3413')
melt = reorganise_mar(melt).sel(TIME=slice('2000-01-01','2012-12-31'))
melt = melt.ME.squeeze()
melt = melt.where(mar_eg.MSK > 50)

# %% trusted=true
# Calculate hydrological year sums
melt_annual = melt.resample(TIME='1A-SEP').sum()
accum_annual = accum.resample(TIME='1A-SEP').sum()

# %% trusted=true
# Calculate ratio
moa = melt_annual / accum_annual
# Save ratio
moa.to_netcdf('/flash/tedstona/jullien_moa/MARv.3.14_MoA_2000_2012.nc')

# %% trusted=true
# Count number of years preceding 2012 when MoA > criterion was met
moa_pre = moa.sel(TIME=slice('2001-09-30','2011-09-30'))
moa_pre_count = moa_pre.where(moa_pre >= moa_threshold).count(dim='TIME').compute()
moa_pre_count.to_netcdf('/flash/tedstona/jullien_moa/MARv.3.14_MoAgtT_count_2001_2011.nc')

# %% trusted=true
# Identify MoA > threshold in 2012
moa_2012_t = moa.sel(TIME='2012-09-30')
moa_2012_t = moa_2012_t >= moa_threshold
moa_2012_t = moa_2012_t.where(mar_eg.MSK > 50)
moa_2012_t.rio.to_raster('/flash/tedstona/jullien_moa/moa_2012_t.tif')

# %% trusted=true
# Polygonise the 2012 MoA
moa_2012_gr = gu.Raster('/flash/tedstona/jullien_moa/moa_2012_t.tif')
polys = moa_2012_gr.polygonize()
polys.save('/flash/tedstona/jullien_moa/MoA_gte_{moa}_2012.shp'.format(moa=moa_threshold))

# %% trusted=true
# Plot the results
plt.figure()
cmap = sns.color_palette("flare", as_cmap=True)
moa_pre_count = moa_pre_count.where(mar_eg.MSK > 0.5)
moa_pre_count.plot(cmap=cmap, cbar_kwargs={'label':'N. Yrs MoA > 0.7 (2001-2011)'})
polys.show(edgecolor='white', color='none', linewidth=0.5)
plt.title('')

# %%
