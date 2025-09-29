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
from pathlib import Path
from matplotlib import rcParams

from tools.detection_html_all import *
from tools.detection_plot_map import *
from tools.detection_download import *

import os
import requests
import subprocess

    
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

def select_data(filelist_l2, aod_min = 0.3, npixel_min = 100*100):
    """
    select data based on aod_min and min npixel
    """
    filev2 =[]
    for i1 in range(len(filelist_l2)):
        file1 = filelist_l2[i1]
        #print(file1)
    
        npixel_valid0, npixel_valid1,filter1 = filter_data(file1, wavelength_index = 1, aot_min = aod_min)
        if npixel_valid1 >=npixel_min:
            print(file1)
            filev2.append(file1)
            print('total valid pixel, valid pixel selected:', npixel_valid0, npixel_valid1)
    return filev2

def make_plot(filev2, plot_path, l1c_path="./data/", flag_cloud=True, aod_min=None):
    """generate plots according to filev2"""
    
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(l1c_path, exist_ok=True)

    boxv = []
    for file1 in filev2[:]:
        try:
            timestamp3, boundingbox, center = plot_l1c_l2(file1, plot_path, l1c_path=l1c_path, flag_cloud=flag_cloud, aod_min=aod_min)
            boxv.append([timestamp3, boundingbox, center])
        except:
            print('failed', file1)
    return boxv