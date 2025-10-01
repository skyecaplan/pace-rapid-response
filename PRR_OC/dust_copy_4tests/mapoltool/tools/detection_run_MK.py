"""
Run aerosol event detection code
Meng Gao, Sep 25, 2025

flag_cloud: True, use earthaccess tool, need cloud access
flag_cloug: False: use web search tool, replace your <appkey>

"""


import earthaccess
import requests

import os
import sys
import glob
import shutil
import numpy as np
import xarray as xr

import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from matplotlib import rcParams

#add the path of the tools
mapol_path=os.path.expanduser('~/github/mapoltool')
sys.path.append(mapol_path)

from tools.detection_html_all import *
from tools.detection_plot_map import *
from tools.detection_util import *
from tools.detection_download import *

from matplotlib import rcParams

import earthaccess

####DO NOT SHARE###########
appkey=<appkey>

flag_cloud=True
flag_cloud=False
if(flag_cloud):
    auth = earthaccess.login(persist=True)

    
# Change default font to something available
rcParams['font.family'] = 'serif' 
rcParams['font.size'] = '12' 

# Add argument parsing with optional values for tspan_start and tspan_end
parser = argparse.ArgumentParser(description="Run the HARP2 FastMAPOL daily processing script.")
parser.add_argument("--tspan_start", type=str, help="Start date of the time span (YYYY-MM-DD).")
parser.add_argument("--tspan_end", type=str, help="End date of the time span (YYYY-MM-DD).")

args = parser.parse_args()

# If no --tspan_start or --tspan_end provided, use the default values as two days ago
default_start_time = "2025-09-17"
default_end_time = "2025-09-17"

tspan_start = args.tspan_start if args.tspan_start else default_start_time
tspan_end = args.tspan_end if args.tspan_end else default_end_time

# Compose tspan based on final values
tspan = (tspan_start, tspan_end)
#tspan = ("2025-09-19", "2025-09-19")

day1 = tspan[0]+'_'+tspan[1]

data_path, l1c_path, plot_path, html_path = setup_data(tspan)

try:
    short_name="PACE_HARP2_L2_MAPOL_OCEAN"
    sensor_id=48
    dtid=1547
    if(flag_cloud):
        filelist_l2 = download_l2_cloud(tspan, short_name=short_name)
    else:
        filelist_l2 = download_l2_web(tspan, appkey, output_folder=data_path,  \
                                      sensor_id=sensor_id, dtid=dtid)
except:
    short_name="PACE_HARP2_L2_MAPOL_OCEAN_NRT"
    sensor_id=48
    dtid=1546
    if(flag_cloud):
        filelist_l2 = download_l2_cloud(tspan, short_name=short_name)
    else:
        filelist_l2 = download_l2_web(tspan, appkey, output_folder=data_path,\
                                     sensor_id=sensor_id, dtid=dtid)
    
print("found:", short_name)

nfile = len(filelist_l2)
print("total file before selection", nfile)

aod_min = 0.3
npixel_min = 100*100
filev2 = select_data(filelist_l2, aod_min=aod_min, npixel_min=npixel_min)
nfile = len(filev2)
print("total file after selection", nfile)


##make plots
make_plot(filev2, plot_path, l1c_path, flag_cloud=flag_cloud)

output_file = html_path+"harp2_fastmapol_"+day1+'_n'+str(nfile)+".html"

sequence = [['globe', 'rgb', 'aot', ], ['ssa', 'fvf', 'sph']]
titlev_custom = [["", "", "AOD (550nm)"], ["Single Scattering Albedo (550nm)", 
                                           "Fine Mode Volume Fraction", "Spherical Fraction"]]

#title = 'PACE Rapid Response on HARP2 FastMAPOL L2:\n
#    {} granule found for valid #pixel > {} (when aod(550nm) > {})'.format(nfile, npixel_min, aod_min)

title = "PACE HARP2 FastMAPOL L2 Rapid Response:{}-{}".format(tspan[0], tspan[1])
title2 = "Total {} granule found for valid #pixel > {} (when aod(550nm) > {})".format(nfile, npixel_min, aod_min)
             
print(title)
print(title2)

image_groups = get_images_from_subfolders(plot_path)
create_html_from_subfolders(image_groups, output_file, sequence, title=title, title2=title2,
                             titlev=titlev_custom, resolution_factor=2, quality=75)


#### copy and clean files
source_file = output_file
destination_folder = "/mnt/mfs/FILESHARE/meng_gao/rapid_pace"
os.makedirs(destination_folder, exist_ok=True)

# Copy the file to the destination folder
destination_path = os.path.join(destination_folder, os.path.basename(source_file))

try:
    shutil.copy(source_file, destination_path)
except:
    print("failed copy the html file")

l2_path, l2_file = os.path.split(filelist_l2[0])

pathv = [l1c_path, l2_path]
for path1 in pathv:
    try:
        shutil.rmtree(path1)  # Recursively remove the folder and its contents
        print(f"✅ Folder removed: {path1}")
    except:
        print("do not exist", path1)