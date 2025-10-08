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
from PIL import Image, ImageEnhance
from pathlib import Path
from matplotlib import rcParams

from detection_html_all_MK import *
from detection_plot_map_MK import *
from detection_download_MK import *

import os
import requests
import subprocess
from matplotlib.colors import LogNorm


    
def setup_data(tspan):
    """
    setup folders, and download l2 data
    tspan: time range
    """
    date_str = tspan[0]
    date_str = date_str.replace('-', '')
    ## cannot change, default for download tool
    data_path = './data/' + date_str
    os.makedirs(data_path, exist_ok=True)

    l2_path = './data/' + date_str + '/L2/'
    os.makedirs(l2_path, exist_ok=True)

    l3_path = './data/' + date_str + '/L3/'
    os.makedirs(l3_path, exist_ok=True)

    plot_path = './figures/' + date_str + '/png/'
    os.makedirs(plot_path, exist_ok=True)

    html_path = './figures/' + date_str + '/html/'
    os.makedirs(html_path, exist_ok=True)

    # Convert to absolute paths for printing
    print("The following directories have been created:")
    print(f"{os.path.abspath(data_path)}\n"
          f"{os.path.abspath(l2_path)}\n"
          f"{os.path.abspath(l3_path)}\n"
          f"{os.path.abspath(plot_path)}\n"
          f"{os.path.abspath(html_path)}")

    return data_path, l2_path, l3_path, plot_path, html_path

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

def plot_xrarray_map(dataset, var_str, fig, ax, cmap=cmocean.cm.haline, vmin=None, vmax=None, log_scale=True, output_path=None):
    """
    Generalized function to plot a variable from an xarray.Dataset.

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
    #fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    #ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    #ax.coastlines()

    # Dynamically calculate vmin and vmax if not provided
    if vmin is None:
        vmin = dataset[var_str].min().item()
    if vmax is None:
        vmax = dataset[var_str].max().item()

    # Determine normalization based on log_scale
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    # Plot the data
    # plot = dataset[var_str].plot(
    #     x="longitude",
    #     y="latitude",
    #     ax=ax,
    #     cmap=cmap,
    #     norm=norm,
    #     extend="neither",
    #     robust=False,
    #     add_colorbar=False,
    #     vmin=vmin,
    #     vmax=vmax 
    # )
    dataset['chlor_a'].plot(x="longitude", y="latitude", cmap=cmocean.cm.haline, norm=LogNorm(vmin=.01, vmax=5), extend="neither")

    # Add and customize the colorbar
    #cbar = plt.colorbar(plot, ax=ax, orientation='vertical', pad=0.05)
    #cbar.set_label(var_str)  # Use the variable name as the label

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)

    #return plot, cbar

def plot_xrarray_map2(dataset, var_str, ax, cmap=cmocean.cm.haline, vmin=None, vmax=None, log_scale=True, output_path=None):
    """
    Generalized function to plot a variable from an xarray.Dataset with 2D lat/lon and Cartopy axis.
    """
    data = dataset[var_str]
    lon = dataset['longitude']
    lat = dataset['latitude']

    if vmin is None:
        vmin = np.nanmin(data.values)
    if vmax is None:
        vmax = np.nanmax(data.values)

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    # Use pcolormesh for Cartopy axes and 2D coordinates
    mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, fraction=0.03)
    cbar.set_label(var_str)

    if output_path:
        plt.savefig(output_path, dpi=300)

    return mesh, cbar

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

def enhance(rgb, scale = 0.01, vmin = 0.01, vmax = 1.04, gamma=0.95, contrast=1.2, brightness=1.1, sharpness=2, saturation=1.1):
    """The SeaDAS recipe for RGB images from Ocean Color missions.

    Args:
        rgb: a data array with three dimensions, having 3 or 4 bands in the third dimension
        scale: scale value for the log transform
        vmin: minimum pixel value for the image
        vmax: maximum pixel value for the image
        gamma: exponential factor for gamma correction
        contrast: amount of pixel value differentiation 
        brightness: pixel values (intensity)
        sharpness: amount of detail
        saturation: color intensity

    Returns:
       a transformed data array better for RGB display
    """
    rgb = rgb.where(rgb > 0)
    rgb = np.log(rgb / scale) / np.log(1 / scale)
    rgb = rgb.where(rgb >= vmin, vmin)
    rgb = rgb.where(rgb <= vmax, vmax)    
    rgb_min = rgb.min(("number_of_lines", "pixels_per_line"))
    rgb_max = rgb.max(("number_of_lines", "pixels_per_line"))
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    rgb = rgb * gamma
    img = rgb * 255
    img = img.where(img.notnull(), 0).astype("uint8")
    img = Image.fromarray(img.data)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation)
    rgb[:] = np.array(img) / 255
    return rgb

def l3_anomaly_bbox(target_dataset, window_dataset, block_size_lat=100, block_size_lon=100, anomaly_threshold=1, count_min=1000):
    """
    Identify bounding boxes of significant chlorophyll-a anomalies in L3 datasets.

    This function computes the chlorophyll-a anomaly by subtracting the mean chlorophyll-a
    from a 30-day window dataset from the target day dataset. It then divides the anomaly grid
    into blocks and identifies bounding boxes where the number of pixels with an anomaly
    greater than `anomaly_threshold` exceeds `count_min`.

    Parameters
    ----------
    target_dataset : xarray.Dataset
        The target L3 dataset (e.g., for a specific day) containing 'chlor_a'.
    window_dataset : xarray.Dataset
        The window L3 dataset (e.g., previous 30 days) containing 'chlor_a'.
    block_size_lat : int, optional
        Block size in the latitude dimension (default is 100).
    block_size_lon : int, optional
        Block size in the longitude dimension (default is 100).
    anomaly_threshold : float, optional
        Minimum absolute anomaly value (mg/m^3) to consider significant (default is 1).
    count_min : int, optional
        Minimum number of significant pixels in a block to identify bounding box (default is 1000).

    Returns
    -------
    bboxes : list of tuple
        List of bounding boxes (min_lon, min_lat, max_lon, max_lat) in the original longitude convention.
    bboxes_0360 : list of tuple
        List of bounding boxes with longitudes converted to the 0-360 degree range.

    Notes
    -----
    - The function assumes the datasets have 'chlor_a', 'lat', and 'lon' variables.
    - Bounding boxes are defined as blocks with a sufficient number of significant anomaly pixels.
    - Use `bboxes_0360` for datasets or plotting that require longitudes in the the 0-360 degree range.

    Example
    -------
    >>> bboxes, bboxes_0360 = l3_anomaly_bbox(target_ds, window_ds)
    """
    chl_anomaly = target_dataset['chlor_a'] - window_dataset['chlor_a'].mean('time')
    chl_anomaly_np = np.squeeze(chl_anomaly.values)

    lat_vals = chl_anomaly['lat'].values
    lon_vals = chl_anomaly['lon'].values

    bboxes = []
    n_lat, n_lon = chl_anomaly_np.shape
    for i in range(0, n_lat, block_size_lat):
        for j in range(0, n_lon, block_size_lon):
            block = chl_anomaly_np[i:i+block_size_lat, j:j+block_size_lon]
            count_above = np.sum(np.abs(block) > anomaly_threshold)
            if count_above > count_min:
                min_lat = np.min(lat_vals[i:i+block_size_lat])
                max_lat = np.max(lat_vals[i:i+block_size_lat])
                min_lon = np.min(lon_vals[j:j+block_size_lon])
                max_lon = np.max(lon_vals[j:j+block_size_lon])
                bbox = (min_lon, min_lat, max_lon, max_lat)
                bboxes.append(bbox)

    bboxes = [tuple(float(x) for x in bbox) for bbox in bboxes]
    bboxes_0360 = bbox_convert_long_0360(bboxes)

    return bboxes, bboxes_0360

def bbox_convert_long_0360(bboxes):
    """
    Convert bounding box longitudes to the 0-360 degree range.

    This function takes a list of bounding boxes defined as (min_lon, min_lat, max_lon, max_lat)
    and converts the longitude values to span from 0 to 360 degrees using modulo arithmetic.
    This is useful for datasets or plotting routines that require longitudes in the 0–360 degree system.

    Parameters
    ----------
    bboxes : list of tuple
        List of bounding boxes, each as (min_lon, min_lat, max_lon, max_lat), where longitudes
        may be in any range (e.g., [-180, 180] or [0, 360]).

    Returns
    -------
    bbox_0360 : list of tuple
        List of bounding boxes with min_lon and max_lon converted to the 0-360 degree range.

    Example
    -------
    >>> bboxes = [(-170, -10, 170, 10), (350, -5, 10, 5)]
    >>> bbox_0360 = bbox_convert_long_0360(bboxes)
    >>> print(bbox_0360)
    [(190.0, -10.0, 170.0, 10.0), (350.0, -5.0, 10.0, 5.0)]
    """
    bbox_0360 = []
    for bbox in bboxes:
        min_lon, min_lat, max_lon, max_lat = bbox
        min_lon_0360 = min_lon % 360
        max_lon_0360 = max_lon % 360
        bbox_0360.append((min_lon_0360, min_lat, max_lon_0360, max_lat))

    return bbox_0360

def l2_granules_by_l3bbox(l3_bboxes, tspan, short_names=['PACE_OCI_L2_SFREFL_NRT','PACE_OCI_L2_SFREFL_NRT'],print_flag=False):
    """
    Search for unique L2 granules across all identified bounding boxes and product, and pair results by scene.

    This function queries Earthdata for L2 granules for each bounding box in l3_bboxes and for each
    product in short_names over the specified time span, tspan. It deduplicates granules by their
    native-id for each product, then pairs the unique results by index, so each tuple in the output
    corresponds to a set of L2 files (one per product) for the same scene.

    Parameters
    ----------
    l3_bboxes : list of tuple
        List of bounding boxes, each as (min_lon, min_lat, max_lon, max_lat).
    tspan : tuple of str
        Time span for the search, e.g., ("2025-09-15", "2025-09-15").
    short_names : list of str, optional
        List of Earthdata product short names to search for. Default is two SFREFL products.
    print_summary : bool, optional
        If True, print summary information about the search and results.

    Returns
    -------
    final_results : list of tuple
        Each tuple contains (results_shortname1, results_shortname2, ...) for a scene,
        where each element is a granule dictionary for the corresponding product.

    Notes
    -----
    - The pairing is by index, so it assumes the order of unique granules matches across products.
    - If the number of unique granules differs between products, only pairs up to the shortest list are returned.
    - Each granule is deduplicated by its 'meta'['native-id'] field.

    Example
    -------
    >>> short_names = ['PACE_OCI_L2_SFREFL_NRT', 'PACE_OCI_L2_BGC_NRT']
    >>> pairs = l2_unique_granules_paired(l3_bboxes, tspan, short_names=short_names, print_summary=True)
    """
    all_results_per_shortname = [[] for _ in short_names]

    for bbox in l3_bboxes:
        for i, short_name in enumerate(short_names):
            results = earthaccess.search_data(
                short_name=short_name,
                temporal=tspan,
                bounding_box=bbox,
                version="3.1"
            )
            all_results_per_shortname[i].extend(results)
            if print_flag:
                print(f" Number of granules for {short_name} in bbox {bbox}: {len(results)}")

    # Deduplicate by native-id for each product type
    unique_results_per_shortname = []
    for results in all_results_per_shortname:
        unique = {}
        for result in results:
            unique[result['meta']['native-id']] = result
        unique_results_per_shortname.append(list(unique.values()))

    final_results = list(zip(*unique_results_per_shortname))
    
    if print_flag:
        print("Total unique granules found for each product:")
        for short_name, unique in zip(short_names, unique_results_per_shortname):
            print(f"  {short_name}: {len(unique)}")
        print(f"Total scenes found: {sum(len(pairs) for pairs in final_results)}")
    return final_results

def bboxes_in_granule(ds, l3_bboxes_0360):
    """
    Identify bounding boxes that overlap with the spatial extent of a dataset.

    This function checks which bounding boxes from a provided list overlap with the
    longitude and latitude range of the given xarray.Dataset. It returns the indices
    of the overlapping bounding boxes.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing 'longitude' and 'latitude' coordinates.
    l3_bboxes_0360 : list of tuple
        List of bounding boxes in the form (min_lon, min_lat, max_lon, max_lat),
        with longitudes in the 0-360 degree convention.

    Returns
    -------
    overlapping_bbox_idx : list of int
        Indices of bounding boxes that overlap with the dataset's spatial extent.

    Example
    -------
    >>> idx = bboxes_in_granule(ds, l3_bboxes_0360)
    >>> print(idx)
    [0, 2, 5]
    """
    overlapping_bbox_idx = []
    for j, bbox in enumerate(l3_bboxes_0360):
        min_lon, min_lat, max_lon, max_lat = bbox
        # Check if bbox overlaps with the dataset's longitude and latitude range
        if (max_lon > ds['longitude'].min().item() and min_lon < ds['longitude'].max().item() and
            max_lat > ds['latitude'].min().item() and min_lat < ds['latitude'].max().item()):
            overlapping_bbox_idx.append(j)
    return overlapping_bbox_idx

def count_valid_pixels(ds, bbox_idx, l3_bboxes_0360, print_flag=False):
    """
    Count the number of valid pixels within specified bounding boxes in a dataset.

    For each bounding box index in bbox_idx, this function creates a mask to identify
    pixels within the bounding box in the dataset's longitude and latitude coordinates.
    It then counts the number of such pixels and returns both the total and per-box counts.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing 'longitude' and 'latitude' coordinates.
    bbox_idx : list of int
        Indices of bounding boxes (from l3_bboxes_0360) to check.
    l3_bboxes_0360 : list of tuple
        List of bounding boxes in the form (min_lon, min_lat, max_lon, max_lat),
        with longitudes in the 0-360 degree convention.
    print_flag : bool, optional
        If True, print the total number of valid pixels for the selected boxes.

    Returns
    -------
    tolpixels : int
        Total number of valid pixels across all specified bounding boxes.
    pixel_counts : dict
        Dictionary mapping bbox index to the number of valid pixels in that box.

    Example
    -------
    >>> total, counts = count_valid_pixels(ds, [0, 2], l3_bboxes_0360)
    """
    tolpixels = 0
    pixel_counts = {}
    for j in bbox_idx:
        # Create a mask for pixels within the bounding box
        mask = (
            (ds['longitude'] >= l3_bboxes_0360[j][0]) & (ds['longitude'] <= l3_bboxes_0360[j][2]) &
            (ds['latitude'] >= l3_bboxes_0360[j][1]) & (ds['latitude'] <= l3_bboxes_0360[j][3])
        )
        numpixels = mask.sum().item()
        pixel_counts[j] = numpixels
        tolpixels += numpixels
    if print_flag:
        print(f"Box(es) {bbox_idx} has(ve) {tolpixels} pixels in granule.")
    return tolpixels, pixel_counts

def filter_l2_by_valid(final_paths, l3_bboxes_0360):
    """
    Filter L2 granule paths by spatial overlap and valid pixel count with L3 bounding boxes.

    For each L2 granule, this function:
      - Opens the dataset and ensures longitude is in 0-360 convention.
      - Identifies which L3 bounding boxes overlap with the granule's spatial extent.
      - Counts the number of valid pixels within each overlapping bounding box.
      - Keeps only granules with at least one valid pixel and latitude coverage within ±89.5°.

    Parameters
    ----------
    final_paths : list
        List of tuples or lists, where each entry contains paths to L2 granule files.
        The second element (path[1]) is used to open the data.
    l3_bboxes_0360 : list of tuple
        List of bounding boxes in the form (min_lon, min_lat, max_lon, max_lat),
        with longitudes in the 0-360 degree convention.

    Returns
    -------
    final_paths_filt : list
        Filtered list of L2 granule paths that overlap with at least one L3 bounding box
        and contain valid pixels.
    granule_bbox_pixel_counts : dict
        Dictionary mapping each kept granule path to a dict of {bbox_idx: valid_pixel_count}.

    Example
    -------
    >>> filtered, pixel_counts = filter_l2_by_valid(final_paths, l3_bboxes_0360)
    """
    final_paths_filt = []
    granule_bbox_pixel_counts = {}
    for path in final_paths:
        # Open the downloaded data as an xarray DataTree and merge into a single Dataset
        dt = xr.open_datatree(path[1], decode_timedelta=True)
        ds = xr.merge(dt.to_dict().values())
        ds = ds.set_coords(("longitude", "latitude"))
        ds = ds.assign_coords(longitude=(ds.longitude % 360))  # Convert longitudes to 0-360
        max_abs_lat = np.abs(ds['latitude']).max().item()

        # Identify indices of boxes that overlap with the dataset's lat/lon range
        overlapping_bbox_idx = bboxes_in_granule(ds, l3_bboxes_0360)
 
        # Count valid pixels in overlapping boxes
        tolpixels, pixel_counts = count_valid_pixels(ds, overlapping_bbox_idx, l3_bboxes_0360, print_flag=False)

        # Only keep granules with valid pixels and reasonable latitude coverage
        if max_abs_lat <= 89.5 and tolpixels > 0:
            final_paths_filt.append(path)
            granule_bbox_pixel_counts[path] = pixel_counts

    return final_paths_filt, granule_bbox_pixel_counts

def map_props(path):
    """
    Extract map properties from a dataset for plotting.

    Opens a dataset from the given path, merges all groups, and computes:
      - Longitude and latitude ranges (in 0-360 and 0-180)
      - Whether the data crosses the dateline (for map projection selection)

    Parameters
    ----------
    path : str or Path
        Path to the dataset file (NetCDF, Zarr, etc.).

    Returns
    -------
    props : dict
        Dictionary with:
            'flag_crossdateline' : bool
                True if the longitude range crosses the dateline (difference > 180°).
            'lon_range' : float
                Range of longitudes (max - min).
            'lat_range' : float
                Range of latitudes (max - min).

    Example
    -------
    >>> props = map_props("granule.nc")
    """
    dt = xr.open_datatree(path, decode_timedelta=True)  # Open the downloaded data as an xarray DataTree
    ds = xr.merge(dt.to_dict().values())  # Merge all datasets in the DataTree into a single xarray Dataset
    ds = ds.set_coords(("longitude", "latitude"))  # Set longitude and latitude as coordinates

    # Convert longitude and latitude to 0-360 and 0-180 for range calculation
    lons = ds['longitude'].values % 360
    lats = ds['latitude'].values % 180

    lonrange = float(np.nanmax(lons) - np.nanmin(lons))
    latrange = float(np.nanmax(lats) - np.nanmin(lats))

    # Determine if longitudes cross the dateline
    flag_crossdateline = (ds['longitude'].max() - ds['longitude'].min()) > 180

    # Store outputs in a dictionary
    props = {
        "flag_crossdateline": flag_crossdateline,
        "lon_range": lonrange,
        "lat_range": latrange
    }

    return props

def plot_setup(props):
    """
    Set up a Cartopy map figure and axis with appropriate projection and aspect ratio.

    This function creates a matplotlib figure and Cartopy axis for plotting geospatial data.
    The projection is chosen based on whether the data crosses the dateline. The map is
    decorated with land, ocean, and coastline features, and gridlines are added.

    Parameters
    ----------
    props : dict
        Dictionary containing map properties, typically from `map_props()`, with keys:
            'flag_crossdateline' : bool
                Whether the longitude range crosses the dateline.
            'lon_range' : float
                Range of longitudes (degrees).
            'lat_range' : float
                Range of latitudes (degrees).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        The created Cartopy axis for plotting.

    Example
    -------
    >>> props = map_props("granule.nc")
    >>> fig, ax = plot_setup(props)
    """
    aspect = props["lon_range"] / props["lat_range"]

    base_hieght = 8
    fig_width = base_hieght * aspect
    if props["flag_crossdateline"]:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    else:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        # To use dynamic width based on aspect, uncomment the next line:
        # fig, ax = plt.subplots(figsize=(fig_width, base_hieght), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add map features for context
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    return fig, ax

def plot_rgb_from_path(path, ax, savefig=False, plot_path=None):
    """
    Plot an RGB composite from hyperspectral reflectance data on a Cartopy axis.

    This function opens a dataset, extracts the reflectance at three wavelengths
    (610, 555, 465 nm) to create an RGB image, enhances it, and plots it using pcolormesh
    on the provided Cartopy axis.

    Parameters
    ----------
    path : str or Path
        Path to the dataset file containing 'rhos' (reflectance) and coordinates.
    ax : matplotlib.axes.Axes
        Cartopy axis on which to plot the RGB image.
    savefig : bool, optional
        If True, save the figure to disk (default: False).
    plot_path : str or Path, optional
        Path to save the figure if savefig is True.

    Returns
    -------
    plot : matplotlib.collections.QuadMesh
        The pcolormesh plot object.

    Example
    -------
    >>> fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    >>> plot_rgb_from_path("granule.nc", ax)
    """
    dt = xr.open_datatree(path, decode_timedelta=True)  # Open the downloaded data as an xarray DataTree
    ds = xr.merge(dt.to_dict().values())  # Merge all datasets in the DataTree into a single xarray Dataset
    ds = ds.set_coords(("longitude", "latitude"))  # Set longitude and latitude as coordinates
    
    # Select reflectance at RGB wavelengths (610, 555, 465 nm)
    rhos_rgb = ds['rhos'].sel(wavelength_3d=[610, 555, 465], method='nearest')
    rgb = enhance(rhos_rgb)  # Enhance the RGB composite (user-defined function)

    # Plot the RGB image using pcolormesh
    plot = ax.pcolormesh(
        rgb["longitude"],
        rgb["latitude"],
        rgb,
        shading="nearest",
        rasterized=True,
        transform=ccrs.PlateCarree()
    )

    # Optionally save the figure
    if savefig and plot_path is not None:
        plt.savefig(plot_path, dpi=300)

    return plot

def plot_var_from_path(path, var_str, ax, cmap=cmocean.cm.haline, vmin=None, vmax=None, log_scale=True, output_path=None):
    """
    Plot a variable from an xarray.Dataset on a Cartopy axis with colorbar.

    Opens a dataset, extracts the specified variable, masks non-positive values (for log scale),
    and plots it using pcolormesh on the provided Cartopy axis. Adds a colorbar with units.

    Parameters
    ----------
    path : str or Path
        Path to the dataset file.
    var_str : str
        Name of the variable to plot.
    ax : matplotlib.axes.Axes
        Cartopy axis on which to plot the data.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use for the plot (default: cmocean.cm.haline).
    vmin : float, optional
        Minimum value for color normalization. If None, uses data minimum.
    vmax : float, optional
        Maximum value for color normalization. If None, uses data maximum.
    log_scale : bool, optional
        If True, use logarithmic color normalization (default: True).
    output_path : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        The pcolormesh plot object.
    cbar : matplotlib.colorbar.Colorbar
        The colorbar object.

    Example
    -------
    >>> fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    >>> plot_var_from_path("granule.nc", "chlor_a", ax)
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar placement

    dt = xr.open_datatree(path, decode_timedelta=True)  # Open the downloaded data as an xarray DataTree
    ds = xr.merge(dt.to_dict().values())  # Merge all datasets in the DataTree into a single xarray Dataset
    ds = ds.set_coords(("longitude", "latitude"))  # Set longitude and latitude as coordinates

    data = ds[var_str]
    data = data.where(data > 0)  # Mask out non-positive values for log scale
    lon = ds['longitude']
    lat = ds['latitude']

    if vmin is None:
        vmin = np.nanmin(data.values)

    if vmax is None:
        vmax = np.nanmax(data.values)

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None

    # Use pcolormesh for Cartopy axes and 2D coordinates
    mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.25, axes_class=plt.Axes)
    cbar = plt.colorbar(mesh, cax=cax, orientation='vertical')
    # Add variable name and units to colorbar label if available
    units = data.attrs['units'] if 'units' in data.attrs else ''
    cbar.set_label(f"{data.name} [{units}]" if units else data.name)

    if output_path:
        plt.savefig(output_path, dpi=300)

    return mesh, cbar

def add_bboxes_to_plot(ax, path, l3_bboxes, granule_bbox_pixel_counts):
    """
    Add bounding boxes to the plot for the given granule path.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Cartopy axis to add bounding boxes to.
    path : tuple or str
        The key for granule_bbox_pixel_counts, typically the granule path.
    l3_bboxes : list of tuple
        List of bounding boxes (min_lon, min_lat, max_lon, max_lat).
    granule_bbox_pixel_counts : dict
        Dictionary mapping path to {bbox_idx: numpixels}.
        Here, `numpixels` is the number of valid pixels within the corresponding bounding box
        for the given granule. It is computed by `count_valid_pixels()` in detection_util_MK.py,
        which counts the number of pixels in the granule that fall within each bounding box and
        meet the validity criteria (e.g., not masked or NaN).

    Returns
    -------
    None

    Notes
    -----
    Only bounding boxes with numpixels > 0 (i.e., containing valid data for the granule)
    are plotted. Optionally, the pixel count can be annotated on the plot.
    """
    import matplotlib.patches as patches       # Drawing shapes (e.g., rectangles)

    for bbox_idx, numpixels in granule_bbox_pixel_counts[path].items():
        if numpixels > 0:
            min_lon, min_lat, max_lon, max_lat = l3_bboxes[bbox_idx]
            # Handle dateline crossing
            if min_lon > max_lon:
                width1 = 180 - min_lon if max_lon < 0 else 360 - min_lon
                rect1 = patches.Rectangle(
                    (min_lon, min_lat), width1, max_lat - min_lat,
                    linewidth=2, edgecolor='red', facecolor='none',
                    transform=ccrs.PlateCarree()
                )
                ax.add_patch(rect1)
                width2 = max_lon - (-180) if max_lon < 0 else max_lon - 0
                rect2 = patches.Rectangle(
                    ((-180 if max_lon < 0 else 0), min_lat), width2, max_lat - min_lat,
                    linewidth=2, edgecolor='red', facecolor='none',
                    transform=ccrs.PlateCarree()
                )
                ax.add_patch(rect2)
            else:
                width = max_lon - min_lon
                height = max_lat - min_lat
                rect = patches.Rectangle(
                    (min_lon, min_lat), width, height,
                    linewidth=2, edgecolor='red', facecolor='none',
                    transform=ccrs.PlateCarree()
                )
                ax.add_patch(rect)
            # Optionally annotate with pixel count
            # ax.text(
            #     (min_lon + max_lon) / 2, (min_lat + max_lat) / 2,
            #     str(numpixels),
            #     color='red', fontsize=8, ha='center', va='center',
            #     transform=ccrs.PlateCarree()
            # )