"""
Define function to make plot of l1 and l2 data

Meng Gao, Sep 25, 2025

If sensors other than HARP2 is used, need check file name and RGB index.
A list of keys are used to specify the variables to be plotted. 

To do:
1. bounding box are not accurate, need get real polygon.
2. may need to get chi2, nv_ref, nv_dolp maps
3. check ocean properties too, chla, rrs etc

"""

import os
import glob
import numpy as np
import xarray as xr
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from tools.detection_download import *


def plot_l1c_l2(file1, plot_path, \
                l1c_path="./data/", iv=[10+30, 5, 10+60+10+5],\
                flag_cloud=True, \
                key1v = ['aot', 'ssa', 'fvf', 'sph'], 
                vmin1v = [0, 0.7, 0, 0],
                vmax1v = [1, 1, 1, 1],
                cmap1v = ['YlOrRd', 'jet', 'jet', 'jet'],
                wavelength_index = 1,
                ):
    """
    plot_path: where is the images
    iv: harp2 rgb angles
    lc_folder: where to save

    key1v: define a list of data to be plotted
    
    """
    ########## get l2 data ########################
    print(file1)
    timestamp3 = file1.split("PACE_HARP2.")[1].split(".L2")[0]
    print(timestamp3)
    
    datatree = xr.open_datatree(file1)
    dataset2 = xr.merge(datatree.to_dict().values())

    ########### get l1 data #######################
    if(flag_cloud):
        filelist_l1c = download_l1c_cloud(file1, l1c_path)
    else:
        filelist_l1c = download_l1c_web(file1, l1c_path)
    
    file4 = filelist_l1c[0]
    print(file4)
    datatree = xr.open_datatree(file4)
    dataset1 = xr.merge(datatree.to_dict().values())
    #############################

    #get lat, lon, and radiance
    lon2=dataset1['longitude'].values
    lat2=dataset1['latitude'].values
    
    tmp2 = dataset1.i[:, :, iv, 0].values

    #set output path
    
    plot_path2 = plot_path+'/'+timestamp3+'/'
    os.makedirs(plot_path2, exist_ok=True)
    print(plot_path2)

    #plot bounding box
    fileout= plot_path2+'pace_harp2'+'_'+timestamp3+'_globe.png'
    print(fileout)
    plot_bounding_box_glob(lat2, lon2, timestamp3, fileout=fileout)

    #plot l1 rgb
    title = 'PACE HARP2 FastMAPOL L2 @'+ timestamp3
    fileout= plot_path2+'pace_harp2'+'_'+timestamp3+'_rgb.png'
    print(fileout)
    plot_rgb(lon2, lat2, tmp2, None, figsize = (10, 5),\
            title=title, fileout=fileout,)


    #plot l2 data

    for i1, key1 in enumerate(key1v):
        try:
            tmp3 = dataset2[key1].values[:,:, wavelength_index]
        except:
            tmp3 = dataset2[key1].values[:,:]
    
        #title = 'PACE HARP2 FastMAPOL L2 @'+ timestamp3
        #cbar_label = key1
        title = key1 + ' (mean:{:0.2f}, std:{:0.2f})'.format(np.nanmean(tmp3),np.nanstd(tmp3)) + '@'+ timestamp3
        cbar_label = None
        
        fileout= plot_path2+'pace_harp2'+'_'+timestamp3+'_'+key1+'.png'
        print(fileout)
        plot_rgb(lon2, lat2, tmp2, tmp3, figsize = (10, 5), \
                 vmin1=vmin1v[i1], vmax1=vmax1v[i1], cmap=cmap1v[i1], \
                 title=title, fileout=fileout, cbar_label=cbar_label)
        
def plot_l2_product(lat, lon, data, plot_range, label, title, vmin, vmax, figsize=(12, 4), cmap="viridis"):
    """Make map and histogram (default)."""

    # Create a figure with two subplots: 1 for map, 1 for histogram
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # Map subplot
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_map.set_extent(plot_range, crs=ccrs.PlateCarree())
    ax_map.coastlines(resolution="110m", color="black", linewidth=0.8)
    ax_map.gridlines(draw_labels=True)

    # Assume lon and lat are defined globally or passed in
    pm = ax_map.pcolormesh(
        lon, lat, data, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap
    )
    plt.colorbar(pm, ax=ax_map, orientation="vertical", pad=0.1, label=label)
    ax_map.set_title(title, fontsize=12)

    # Histogram subplot
    ax_hist = fig.add_subplot(gs[1])
    flattened_data = data[~np.isnan(data)]  # Remove NaNs for histogram
    valid_count = np.sum(~np.isnan(flattened_data))
    ax_hist.hist(
        flattened_data, bins=40, color="gray", range=[vmin, vmax], edgecolor="black"
    )
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Histogram: N=" + str(valid_count))

    # plt.tight_layout()
    plt.show()


def filter_data(file1, wavelength_index = 1, aot_min = 0.15,  nv_ref_min=30, nv_dolp_min=20, chi2_max = 2.0):
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

    
def reset_data_for_rgb(tmp2, scale1=1/250, scale2=0.5, bias=-0.1):
    tmp2 = (tmp2*scale1)**scale2
    tmp2=tmp2+bias
    tmp2[tmp2<0]=0.0
    tmp2[tmp2>0.99] = 0.99
    return tmp2

def reset_lon(i, tmp2, lon2):
    """resolve the issue when lon cross dateline
    i=0: lon<0
    i=1: lon>0
    """
    if(i==0):
        tmp2t = tmp2.copy()
        #print(tmp2t.shape, lon2.shape)
        #lon2t = lon2.copy()
        tmp2t[lon2<-179]=np.nan
        tmp2t[lon2>0]=np.nan
        tmp2t=np.ma.masked_where(np.isnan(tmp2t),tmp2t)
    else:
        tmp2t = tmp2.copy()
        #lon2t = lon2.copy()
        #lon2t[lon2<0]=np.nan
        tmp2t[lon2<0]=np.nan
        tmp2t[lon2>179]=np.nan
        tmp2t=np.ma.masked_where(np.isnan(tmp2t),tmp2t)
    return tmp2t



def plot_rgb(lon2, lat2, tmp2, tmp3, figsize = (10, 5), \
            vmin1=0, vmax1=1.0, cmap='YlOrRd', title=None, fileout=None, \
             cbar_label=None, cbar_label_fontsize=14):
    """
    extent: for the map
    vmin1, vmax1: color bar range
    cmap: color style
    """
    extent=[np.min(np.concatenate(lon2)),np.max(np.concatenate(lon2)), \
            np.min(np.concatenate(lat2)),np.max(np.concatenate(lat2))]
    
    fig = plt.figure(figsize=figsize)
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    ################################
    ### plot rgb ###################
    tmp2 = reset_data_for_rgb(tmp2)
    
    plt.pcolormesh(lon2, lat2, tmp2,transform=ccrs.PlateCarree())

    ################################
    #### plot variable #############
    try:
        levels = np.linspace(vmin1, vmax1,21)
        ticks = np.linspace(vmin1, vmax1, 6)
        tick_labels = ["{:0.2f}".format(t1) for t1 in ticks]
        
        plt.pcolormesh(lon2, lat2, tmp3,transform=ccrs.PlateCarree(),cmap=cmap,vmin=levels[0], vmax=levels[-1])

        cbar=plt.colorbar(shrink=0.8, pad=0.02) #shrink=0.9, pad=0.1 
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.xaxis.set_label_position('top')  # Position the label on the top
        cbar.ax.set_xlabel(cbar_label, labelpad=10, fontsize=cbar_label_fontsize)  # Customize label

    except:
        pass
    
    ########################
    xbin=5
    ybin=5
    alpha=0.3
    
    gl=ax.gridlines(linewidth=0.5, color='gray', alpha=0.3, linestyle='-')
    cl=ax.coastlines(resolution='50m', color='k', linewidth=0.1) #10m, 110m
    ax.add_feature(cartopy.feature.OCEAN, edgecolor='w',linewidth=0.01)
    ax.add_feature(cartopy.feature.LAND, edgecolor='w',linewidth=0.01)
    gl.top_labels = False #True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180,180,xbin))
    gl.ylocator = mticker.FixedLocator(np.arange(-90,90,ybin))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_xlabel(r"Longitude($^\circ$)")
    ax.set_ylabel(r"Latitude($^\circ$)")
    plt.tight_layout()
    plt.title(title)

    if(fileout):
        plt.savefig(fileout, dpi=400, bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def plot_bounding_box_glob(lat, lon, timestamp, xbin=15, ybin=15, title=None, fileout=None):
    """
    Plot a bounding box on a global map with a text annotation displaying the timestamp.
    
    Parameters:
    - lat, lon: 2D numpy arrays of latitude and longitude values
    - timestamp: A string to display the timestamp near the bounding box
    - xbin, ybin: Gridline spacing in degrees
    - title: Title of the plot (optional)
    - fileout: If provided, saves the plot to the specified file
    """
    # Step 1: Extract bounding box coordinates
    min_lat, max_lat = np.min(lat), np.max(lat)
    min_lon, max_lon = np.min(lon), np.max(lon)
    
    # Step 2: Calculate the central latitude and longitude for the bounding box
    central_lat = (min_lat + max_lat) / 2
    central_lon = (min_lon + max_lon) / 2
    print(f"Central Latitude: {central_lat}, Central Longitude: {central_lon}")
    
    # Step 3: Set up the Orthographic projection centered on the bounding box
    proj = ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat)
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': proj})
    ax.set_global()
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # Step 4: Plot the bounding box as a polygon
    lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
    lats = [min_lat, min_lat, max_lat, max_lat, min_lat]
    ax.plot(lons, lats, transform=ccrs.Geodetic(), color='red', linewidth=2, label='Bounding Box')
    
    # Step 5: Add the timestamp as a text annotation
    # Place the text slightly outside the bounding box in the top-right corner
    ax.text(min_lon, max_lat+5, f'Timestamp: {timestamp}',
            transform=ccrs.Geodetic(), fontsize=10, color='blue',
            ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    # Add gridlines for reference
    gridlines = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gridlines.top_labels = False  # Hide top grid labels
    gridlines.right_labels = False  # Hide right grid labels
    gridlines.left_labels = False  # Hide left grid labels
    gridlines.bottom_labels = False  # Hide bottom grid labels
    gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 180, xbin))
    gridlines.ylocator = mticker.FixedLocator(np.arange(-90, 90, ybin))
    
    # Add title
    if title:
        plt.title(title, fontsize=14)

    # Save to file if fileout is provided
    if fileout:
        plt.savefig(fileout, dpi=300, bbox_inches='tight', pad_inches=0.1)
        
    # Show the plot
    plt.show()