#  Generally useful functions for working with PACE data  #
# Contribute as you see fit! And don't forget to comment! #

import xarray as xr 
import numpy as np 
import cf_xarray

def open_nc(filename, filetype="L2"):
    """ 
    Opens a L1B- or L2-like PACE file and assign lat/lon as coordinates. 
    Returns the entire observation_ or geophysical_data groups in the 
        output dataset
    Args:
        filename - Path to input PACE file
        filetype - str to tell which file type to open. Default = L2
                   Options: L1B or L2
    Returns:
        ds - xarray dataset with lat/lon as coords 
    """
    # Open file as datatree
    dt = xr.open_datatree(filename)
    if filetype =="L1B":
        ds = dt["observation_data"].to_dataset()
        ds.coords["longitude"] = dt["geolocation_data"]["longitude"]
        ds.coords["latitude"] = dt["geolocation_data"]["latitude"]
        return ds
    elif filetype =="L2":
        # First try to open the dataset with wavelength_3d as a coordinate
        try:
            ds = xr.merge(
                (
                dt.ds,
                dt["geophysical_data"].to_dataset(),
                dt["sensor_band_parameters"].coords,
                dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
                )
            )
        except:
            # If that doesn't work, don't use sensor_band_params
            ds = xr.merge(
                (
                dt.ds,
                dt["geophysical_data"].to_dataset(),
                dt["navigation_data"].ds.set_coords(("longitude", "latitude")).coords,
                )
            )
        return ds

def subset(ds, extent):
    """
    Subset a L2 PACE file
    Skye's note: I don't think the Hackweek method using slicing works
                 for L2's bc the lat/lon arrays are 2D, but please 
                 correct me if I'm wrong!
    Args:
        ds - xarray dataset with lat/lon set as coordinates
        extent - list of boundaries in [w, s, e, n] configuration
    Returns:
        sub - subset of ds clipped to boundaries
    """
    sub = ds.where(
        (
          (ds["latitude"]  > extent[1])
        & (ds["latitude"]  < extent[3])
        & (ds["longitude"] > extent[0])
        & (ds["longitude"] < extent[2])
        ),
            drop=True,
        )
    return sub

def mask_ds(ds, flag="CLDICE", mask_reverse=False):
    """
    Mask for a PACE dataset for an L2 flag using cf_xarray. Default is clouds
    Args:
        ds - xarray dataset containing "l2_flags" variable
        flag - l2 flag to mask for (see https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/)
        mask_reverse - keep only pixels with the desired flag. Default is False. E.g., use the
                       "LAND" flag to mask water pixels. 
    Returns:
        Masked dataset
    """
    # Check if cf_xarray recognizes l2_flags
    if ds["l2_flags"].cf.is_flag_variable:
        # If so, mask
        if mask_reverse==False:
            return ds.where(~(ds["l2_flags"].cf == flag))
        else:
            return ds.where((ds["l2_flags"].cf == flag))
    else:
        print("l2flags not recognized as flag variable")