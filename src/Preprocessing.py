#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:57:16 2026

@author: james
"""

from gwosc.datasets import event_gps, find_datasets
from gwpy.timeseries import TimeSeries
import numpy as np

#------------------------------------------------finding gps times for all O3 confirmed events--------------------------------------------------------
events = find_datasets(type='event', catalog='GWTC-3-confident')
#print(events)

for i, event in enumerate(events):
    o3b_gps = event_gps(event)
    print(i + 1 ,":",event, o3b_gps)
    
#------------------------------------------------------picking confirmed events from O3b--------------------------------------------------------------


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
    
#--------------------------------------------------------bandpassing filter on all data--------------------------------------------------------------
signal_data['GW191103_012549-v1'].plot()

filtered = signal_data['GW191103_012549-v1'].bandpass(20, 500)
filtered.plot()

bp_events = {}
for event in signal_data:
    filtered = signal_data[event].bandpass(20, 500)
    bp_events[event] = filtered
    
bp_noise = {}
for sample in noise_data:
    filtered = noise_data[sample].bandpass(20, 500)
    bp_noise[sample] = filtered

#--------------------------------------------------------------whitening data-------------------------------------------------------------------------
wbp_events = {}
for event in bp_events:
    whitened = bp_events[event].whiten()
    wbp_events[event] = whitened
    
wbp_noise = {}
for sample in bp_noise:
    whitened = bp_noise[sample].whiten()
    wbp_noise[sample] = whitened

wbp_events['GW191103_012549-v1'].plot()
    
#----------------------------------------------------------cropping two seconds off each end----------------------------------------------------------
    
cwbp_events = {}
for event in wbp_events:
    gps = event_gps(event)
    cropped = wbp_events[event].crop(gps-14, gps+14)
    cwbp_events[event] = cropped
    
cwbp_noise = {}
for i, sample in enumerate(samples):
    cropped = wbp_noise[i].crop(sample - 14, sample + 14)
    cwbp_noise[i] = cropped

cwbp_events['GW191103_012549-v1'].plot()

#-----------------------------------------------------------saving processed data to numpy arrays----------------------------------------------------

event_array = np.array([cwbp_events[event].value for event in cwbp_events])
noise_array = np.array([cwbp_noise[sample].value for sample in cwbp_noise])
    

labels = np.array([1]*len(event_array) + [0]*len(noise_array))

np.save('/Users/james/Documents/PHYSICS/gw-classifier/data/events.npy', event_array)
np.save('/Users/james/Documents/PHYSICS/gw-classifier/data/noise.npy', noise_array)
np.save('/Users/james/Documents/PHYSICS/gw-classifier/data/labels.npy', labels)

