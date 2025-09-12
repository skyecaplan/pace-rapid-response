#  Generally useful functions for working with PACE data  #
# Contribute as you see fit! And don't forget to comment! #

import xarray as xr 
import numpy as np 
import cf_xarray
import rasterio 
import rioxarray as rio
import cartopy.crs as ccrs
from rasterio.enums import Resampling
from PIL import Image, ImageEnhance

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
    dt = xr.open_datatree(filename, decode_timedelta=False)
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
    To do: add utility to mask multiple flags at once
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

def reproject_3d(src, crs="epsg:4326"):
    """
    Project a L2 3d variable with a given CRS. Will take either a single xr object or 
        a filepath from earthaccess.
    Note: Only tested for SFREFL, but should work for Rrs as long as src = only the Rrs
        dataset
    Args:
        src - either an xr object or a list of earthaccess paths
        crs - coordinate reference system for projection. Currently will project into
              the same as written in
    Returns:
        dst - projected xr dataset
    """
    # Open file if given a path
    if type(src)!= xr.Dataset and type(src)!= xr.DataArray:
        src = open_nc(src)
    # Make sure bands are first, as rio expects
    if list(src.dims)[0] != "wavelength_3d":
        src = src.transpose("wavelength_3d", ...)
    src = src.rio.set_spatial_dims("pixels_per_line", "number_of_lines")
    src = src.rio.write_crs(crs)

    dst = src.rio.reproject(
        dst_crs = src.rio.crs, 
        src_geoloc_array=(
            src.coords["longitude"],
            src.coords["latitude"],
        ),
        nodata=np.nan,
        resample=Resampling.nearest,
    ).rename({"x":"longitude", "y":"latitude"})
    return dst

def grid_match_3d(src, crs, dst_shape=None, transform=None):
    """
    Reproject an L2 to match an input shape and grid 
    Args:
        src - an xarray dataset or a path to a file
        crs - coordinate reference system for projection. Currently will project into
              the same as written in
        dst_shape - shape of the dataset to match
        transform - Affine transform to project into
    Returns:
        dst - projected xr dataset
    """
    # Open file if given a path
    if type(src)!= xr.Dataset and type(src)!= xr.DataArray:
        src = open_nc(src)
    # Make sure bands are first, as rio expects
    if list(src.dims)[0] != "wavelength_3d":
        src = src.transpose("wavelength_3d", ...)
    src = src.rio.set_spatial_dims("pixels_per_line", "number_of_lines")
    src = src.rio.write_crs(crs)

    dst = src.rio.reproject(
        dst_crs=src.rio.crs,
        shape=dst_shape,
        transform=transform,
        src_geoloc_array=(
            src.coords["longitude"],
            src.coords["latitude"],
        ),
        nodata=np.nan,
        resample=Resampling.nearest,
    ).rename({"x":"longitude", "y":"latitude"})
    return dst

def make_rgb(ds, scale=0.01, vmin=0, vmax=1.1, gamma=1, contrast=1.1, brightness=1, sharpness=1.1, saturation=1):
    """
    To Do: expand docstring
    Carina's code for making very nice RGB images out of reflectance data
    Args: 
        ds - l2 sfrefl data
        all others - img enhancer params
    Returns:
        rgb_ds - ds of enhanced rgb for plotting
    """
    rgb = ds["rhos"].sel({"wavelength_3d": [645, 555, 368]}, method="nearest")
    # Apply your processing steps
    rgb = rgb.where(rgb > 0)
    rgb = np.log(rgb / scale) / np.log(1 / scale)
    rgb = rgb.where(rgb >= vmin, vmin)
    rgb = rgb.where(rgb <= vmax, vmax)    
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = rgb * gamma
    
    # Convert to 8-bit image format
    img = rgb * 255
    img = img.where(img.notnull(), 0).astype("uint8")
    
    # COMPUTE the dask array to numpy before using PIL
    img_numpy = img.compute().values  # This converts dask array to numpy array
    
    # Check the shape and adjust if necessary for PIL
    #print(f"Image shape: {img_numpy.shape}")
    
    # PIL expects the wavelength dimension to be last, so we might need to transpose
    # If shape is (wavelength, y, x), transpose to (y, x, wavelength)
    if img_numpy.shape[0] == 3:  # wavelength dimension is first
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
    
    # Now PIL can work with the numpy array
    img_pil = Image.fromarray(img_numpy)
    
    # Apply enhancements
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer.enhance(sharpness)
    
    enhancer = ImageEnhance.Color(img_pil)
    img_pil = enhancer.enhance(saturation)
    
    # Convert back to numpy array and normalize to 0-1 range
    rgb_enhanced = np.array(img_pil) / 255
    rgb_ds = xr.Dataset(data_vars={"rgb":(["number_of_lines","pixels_per_line", "wavelength_3d"], rgb_enhanced)},
                    coords={"latitude":ds.latitude, 
                            "longitude":ds.longitude, 
                            "wavelength_3d":[645, 555, 368]})
    return rgb_ds
    