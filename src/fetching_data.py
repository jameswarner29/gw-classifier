#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:58:21 2026

@author: james
"""

from gwosc.datasets import event_gps, find_datasets
from gwpy.timeseries import TimeSeries
import random 
import numpy as np

# Fetch strain data around a known event
#gps = event_gps('GW150914')                                     #GW150914 - first ever event
#strain = TimeSeries.fetch_open_data('H1', gps-10, gps+10)
#strain.plot()

#naming available catalogs:
#available_catalogs = find_datasets(type = 'catalog')
#print(available_catalogs)

#------------------------------------------------finding gps times for all O3 confirmed events--------------------------------------------------------
events = find_datasets(type='event', catalog='GWTC-3-confident')
#print(events)

for i, event in enumerate(events):
    o3b_gps = event_gps(event)
    print(i + 1 ,":",event, o3b_gps)
    

#---------------------------------------------collecting 32 second window around each event from H1----------------------------------------------------
signal_data = {}
for event in events:
    gps = event_gps(event)
    try:
        data = TimeSeries.fetch_open_data('H1', gps - 16, gps + 16, cache = True)
        signal_data[event] = data
        print(event, data.shape)
    except Exception as e:
        print(f"Skipping {event} : {e} ")
    
#---------------------------------------------picking negative class from O3b (1256655618 to 1269363618)-----------------------------------------------

rng = np.random.default_rng(42)
o3b_start = 1256655618 
o3b_end = 1269363618

event_times = []
for event in events:
    gps = event_gps(event)
    if 1256655618 <= gps <= 1269363618:
        event_times.append(gps)
        
samples = []
while len(samples) < 175:
    sample = rng.uniform(o3b_start, o3b_end)
    if all(abs(sample - o3b_event) >= 100 for o3b_event in event_times):
        samples.append(sample)
  
print(samples)

noise_data = {}

for i, sample in enumerate(samples):
    quiet_data = TimeSeries.fetch_open_data('H1', gps - 16, gps + 16, cache = True)
    noise_data[i] = quiet_data
    print(i + 1, quiet_data.shape)