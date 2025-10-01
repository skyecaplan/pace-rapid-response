"""
Setup the function calls
Meng Gao, Sep 25, 2025

Need specify proper directories
Note that earthaccess download, use a defult folder of ./data
"""

import earthaccess
import requests

import os
import glob
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
from pathlib import Path
from matplotlib import rcParams

from tools.detection_html_all_MK import *
from tools.detection_plot_map_MK import *
from tools.detection_download_MK import *

import os
import requests
import subprocess
from matplotlib.colors import LogNorm


    
def setup_data(tspan):
    """
    setup folders, and download l2 data
    tspan: time range
    """
    day1 = tspan[0]+'_'+tspan[1]
    ## cannot change, default for download tool
    data_path = './data/'+day1
    os.makedirs(data_path, exist_ok=True)

    l1c_path = './data_l1c/'+day1
    os.makedirs(l1c_path, exist_ok=True)

    plot_path = './plot/'+day1
    os.makedirs(plot_path, exist_ok=True)

    html_path = './html/'
    os.makedirs(html_path, exist_ok=True)

    print(data_path, l1c_path, plot_path, html_path)
    
    return data_path, l1c_path, plot_path, html_path

def calc_chl_anomaly_MK(filelist_l3, filelist_l3_previous):
    """
    calculate the 30-day chl anomaly of filelist_l3
    """

    dataset = xr.open_mfdataset(filelist_l3_previous, combine='nested', concat_dim='time')
    window_mean = dataset['chlor_a'].mean("time")
    print(dataset)

    return dataset 
    # chl_all = []
    # for i1 in range(len(filelist_l3)):
    #     file1 = filelist_l3[i1]
    #     print(file1)
    
    #     datatree = xr.open_datatree(file1)
    #     dataset = xr.merge(datatree.to_dict().values())
        
    #     chl = dataset["chlor_a"].values
    #     chl_all.append(chl)
    
    # chl_all = np.array(chl_all)
    # chl_mean = np.nanmean(chl_all, axis=0)
    
    # return chl_mean

def L3_quickplot_dataarray_MK(dataarray, title=None, cmap=cmocean.cm.haline, clabel=None, vmin=None, vmax=None, log_scale=True, output_path=None):
    """
    Generalized function to plot a variable from an xarray.DataArray.

    Parameters:
    - dataset: xarray.Dataset containing the variable to plot.
    - var_str: Name of the variable in the dataset to plot.
    - bbox: Tuple of (min_lon, min_lat, max_lon, max_lat) for plot limits.
    - vmin: Minimum value for color normalization (optional).
    - vmax: Maximum value for color normalization (optional).
    - log_scale: Boolean to toggle between LogNorm (True) and linear scale (False).
    - output_path: Optional path to save the figure. If None, the plot is not saved.

    Returns:
    - fig: The matplotlib figure object.
    - ax: The matplotlib axis object.
    - plot: The xarray plot object.
    - cbar: The matplotlib colorbar object.
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    ax.coastlines()

    # Dynamically calculate vmin and vmax if not provided
    if vmin is None:
        #vmin = dataarray.min().item()
        vmin = np.nanmin(dataarray.values)
    if vmax is None:
        #vmax = dataarray.max().item()
        vmax = np.nanmax(dataarray.values)

    # Determine normalization based on log_scale
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    # Plot the data
    plot = dataarray.plot(
        x="lon",
        y="lat",
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="neither",
        robust=False,
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax 
    )

    # Add and customize the colorbar
    cbar = plt.colorbar(plot, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label(clabel)  # Use the variable name as the label

    # # Set plot limits
    # ax.set_xlim(bbox[0], bbox[2])
    # ax.set_ylim(bbox[1], bbox[3])

    ax.set_title(title)

    #plt.show()

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)

    return fig, ax, plot, cbar

def select_data_MK(filelist_l2, aod_min = 0.3, npixel_min = 100*100):
    """
    select data based on aod_min and min npixel
    """
    filev2 =[]
    for i1 in range(len(filelist_l2)):
        file1 = filelist_l2[i1]
        #print(file1)
    
        npixel_valid0, npixel_valid1,filter1 = filter_data_MK(file1, wavelength_index = 1, aot_min = aod_min)
        if npixel_valid1 >=npixel_min:
            print(file1)
            filev2.append(file1)
            print('total valid pixel, valid pixel selected:', npixel_valid0, npixel_valid1)
    return filev2

def filter_data_MK(file1, wavelength_index = 1, aot_min = 0.15,  nv_ref_min=30, nv_dolp_min=20, chi2_max = 2.0):
    """
    check the file, and output total number of pixels agree with the rules based on:
    aot, nv_ref, nv_dolp, chi2_max
    """
    datatree = xr.open_datatree(file1)
    dataset = xr.merge(datatree.to_dict().values())
    
    nv_ref = dataset["nv_ref"].values
    nv_dolp = dataset["nv_dolp"].values
    chi2 = dataset["chi2"].values
    
    aot = dataset["aot"].values

    data = aot[:, :, wavelength_index]
    npixel_valid0 = np.sum(~np.isnan(data))

    filter1 = (aot[:, :, wavelength_index] >= aot_min) & (nv_ref>=nv_ref_min) & (nv_dolp>=nv_dolp_min) & (chi2 <=chi2_max)
    data = np.where(filter1,data , np.nan)
    npixel_valid1 = np.sum(~np.isnan(data))

    return npixel_valid0, npixel_valid1, filter1


def make_plot(filev2, plot_path, l1c_path="./data/", flag_cloud=True):
    """generate plots according to filev2"""
    
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(l1c_path, exist_ok=True)
    
    for file1 in filev2[:]:
        try:
            plot_l1c_l2(file1, plot_path, l1c_path=l1c_path, flag_cloud=flag_cloud)
        except:
            print('failed', file1)