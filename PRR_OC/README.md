# PACE Rapid Response - Ocean Color (PRR_OC)
---

Main directroy for ocean color related PACE Rapid Response Code. There are directories with fully mature code for the automatic detection of potential bloom forming regions (/Bloom_Detection) and for the analysis of hurricanes and tropical strom (/Hurricane_Analysis).

This README document provides infromation about the directories and files within the PRR_OC repository.

---

# Bloom_Detection
Subdirectory within PRR_OC repository containting the files for the construction of the automatic potential [Bloom Detection Dashboard] (https://oceancolor.gsfc.nasa.gov/fileshare/graham_trolley/chla_anomaly_rapid_response)
The most important files for end-users within the subdirectory are listed below.

## Bloom_Detection.py
Main python script used to automatically chlorophyll-a anamolies to identify potential bloom conditions using PACE L3 data

## gt_html_utils.py
Utility functions used to assemble figures produced from Bloom_Detection.py script into interpretable .html files, which are hosted on fileshare

---

# Hurricane_Analysis
Subdirectory within PRR_OC repository containing the files for to analyze the physical and biogeochemical from hurricane events in the North Atlantic. The most important files for end-users within the subdirectory are listed below.

## PRR_Hurricane_ScienceNugget_Template.ipynb
Geneneral Jupyter notebook for the analysis of tropical storm and hurricane events that occur in the North Atlantic. End user's should be able to follow along with the description and instuctions listed in the notebook to analyze new events.

---

Contributors: Matthew Kehrli, and Graham Trolley\
Contacts: Matthew Kehrli (matthew.d.kehrli@nasa.gov) and Graham Trolley (graham.r.trolley@nasa.gov)\
Ocean Ecology Laboratory, Goddard Space Flight Center, NASA

---
