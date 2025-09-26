"""
Download L1c and L2 data using both cloud or web tools

Meng Gao, Sep 25, 2025

Need cloud access if earthaccess is used. 
Need appkey 
"""

import os
import requests
import subprocess
import earthaccess

from pathlib import Path


def download_l2_cloud(tspan, short_name="PACE_HARP2_L2_MAPOL_OCEAN_NRT"):
    """download ata using earthaccess"""
    
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=tspan,
        )

    ###save into a temporary path as listed in filelist_l2
    filelist_l2 = earthaccess.download(results)
    
    return filelist_l2
    
def download_l1c_cloud(file1, l1c_path):
    timestamp3 = file1.split("PACE_HARP2.")[1].split(".L2")[0]
    FILENAME = "PACE_HARP2."+timestamp3+".L1C.V3.5km.nc"
    print(FILENAME)
    
    fs = earthaccess.get_fsspec_https_session()
    OB_DAAC_PROVISIONAL = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/"
    fs.get(f"{OB_DAAC_PROVISIONAL}/{FILENAME}", l1c_path)
    filelist_l1c = list(Path(l1c_path).glob("*"+timestamp3+"*.nc"))
    return filelist_l1c


def download_l2_web(tspan_date, appkey, sensor_id=48, dtid=1546, \
                    output_folder="./downloads", filelist_name="./filelist_harp2.txt"):
    """
    Function to search, validate, and download files for a given time range.

    Parameters:
    ----------
    tspan : tuple
        A tuple containing the start and end times (e.g., ("YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS")).
    appkey : str, optional
        API key for authentication with the data service.
    output_folder : str, optional
        Directory to save the downloaded files.
    filelist_name : str, optional
        Path to save the file list from the API.
    
    Returns:
    -------
    downloaded_files : list
        A list of file paths that were downloaded.

    Notes;
    harp2 mapol refined: sensor_id=48&dtid=1547
    harp2 mapol nrt: sensor_id=48&dtid=1546
    """

    # Adding time components to the start and end of the range
    tspan_start = f"{tspan_date[0]} 00:00:00"  # Add start time '00:00:00'
    tspan_end = f"{tspan_date[1]} 23:59:59"    # Add end time '23:59:59'

    # Combine start and end times into tspan
    tspan = (tspan_start, tspan_end)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Query the API to generate the list of files
    api_url = "https://oceandata.sci.gsfc.nasa.gov/api/file_search"

    payload = {
        "results_as_file": 1,
        "sensor_id": 48,                  # Sensor ID for HARP2
        "dtid": 1546,                    # Data format ID
        "sdate": tspan[0],               # Start date
        "edate": tspan[1],               # End date
        "appkey": appkey,                # API key
    }

    # POST request to get the file list
    try:
        response = requests.post(api_url, data=payload)
        response.raise_for_status()  # Raise an error if the request failed
        with open(filelist_name, "w") as file_list:
            file_list.write(response.text)
        print(f"✅ File list saved to {filelist_name}")
    except requests.RequestException as e:
        print(f"❌ Error in API request: {e}")
        return []

    # Read file names from the file list
    with open(filelist_name, "r") as file_list:
        file_names = [line.strip() for line in file_list.readlines()]

    # Step 2: Process and download files
    downloaded_files = []
    for file_name in file_names:
        output_file_path = os.path.join(output_folder, file_name)

        # Check if the file already exists and is valid
        if os.path.exists(output_file_path):
            try:
                subprocess.run(
                    ["ncdump", "-h", output_file_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"✅ File '{file_name}' already exists and is valid.")
                continue
            except subprocess.CalledProcessError:
                print(f"⚠️ File '{file_name}' is invalid. Deleting and re-downloading.")
                os.remove(output_file_path)

        # Download the file
        try:
            download_url = f"https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/{file_name}"
            print(f"⬇️  Downloading: {file_name}")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raise an error if the request failed
            with open(output_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"✅ File downloaded: {file_name}")
            downloaded_files.append(output_file_path)
        except requests.RequestException as e:
            print(f"❌ Failed to download {file_name}: {e}")

    print(f"✅ Total downloaded files: {len(downloaded_files)}")
    return downloaded_files
    

def download_l1c_web(file1, l1c_path):
    """
    Retrieve Level 1C data file based on the timestamp extracted from a Level 2 file.

    Parameters:
    ----------
    file1 : str
        The name of the Level 2 file.
        Example: "PACE_HARP2.2025-09-20T12-00-00.L2.V3.nc"
    l1c_path : str
        The directory where the downloaded Level 1C file will be stored.

    Returns:
    -------
    filelist_l1c : list
        A list of file paths matching the Level 1C netCDF files downloaded to the `l1c_path` directory.
    """
    # Ensure the output directory exists
    os.makedirs(l1c_path, exist_ok=True)

    # Extract the timestamp from the Level 2 file
    try:
        timestamp3 = file1.split("PACE_HARP2.")[1].split(".L2")[0]
    except IndexError:
        raise ValueError(f"Invalid file format: {file1}. Unable to extract timestamp.")

    # Construct the Level 1C file name
    file_name = f"PACE_HARP2.{timestamp3}.L1C.V3.5km.nc"
    print(f"Constructed filename for Level 1C data: {file_name}")

    # Download the file
    output_file_path = os.path.join(l1c_path, file_name)
    #downloaded_files = []
    try:
        download_url = f"https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/{file_name}"
        print(f"⬇️  Downloading: {file_name}")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an error if the request failed
        with open(output_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"✅ File downloaded: {file_name}")
        #downloaded_files.append(output_file_path)
    except requests.RequestException as e:
        print(f"❌ Failed to download {file_name}: {e}")

    # Find all .nc files in the l1c_path directory that match the timestamp
    filelist_l1c = list(Path(l1c_path).glob(f"*{timestamp3}*.nc"))
    return filelist_l1c