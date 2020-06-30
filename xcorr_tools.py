#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:26:25 2020

xcorr module

@author: amt
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.quakeml.core import Unpickler
import pandas as pd
from libcomcat.dataframes import get_phase_dataframe, get_magnitude_data_frame
from libcomcat.search import get_event_by_id
import pandas as pd
from datetime import timedelta
from obspy import Stream, read, UTCDateTime, Trace 
from obspy.clients.fdsn import Client
from distutils.version import LooseVersion
from obspy.signal.cross_correlation import correlate_template
from sklearn.cluster import DBSCAN

def make_template(df,sr):
    client = Client("IRIS")
    # make templates
    regional=df['Regional']
    eventid = regional+str(df['ID'])
    detail = get_event_by_id(eventid, includesuperseded=True)
    phases = get_phase_dataframe(detail, catalog=regional)
    phases = phases[phases['Status'] == 'manual']
    print(phases)
    phases=phases[~phases.duplicated(keep='first',subset=['Channel','Phase'])]
    print(phases)
    st=Stream()
    tr=Stream()
    print(phases)
    for ii in range(len(phases)):
        net=phases.iloc[ii]['Channel'].split('.')[0]
        sta=phases.iloc[ii]['Channel'].split('.')[1]
        comp=phases.iloc[ii]['Channel'].split('.')[2]
        #phase=phases.iloc[ii]['Phase']
        arr=UTCDateTime(phases.iloc[ii]['Arrival Time'])
        #print(int(np.round(arr.microsecond/(1/sr*10**6))*1/sr*10**6)==1000000)
        if int(np.round(arr.microsecond/(1/sr*10**6))*1/sr*10**6)==1000000:
            arr.microsecond=0
            arr.second=arr.second+1
        else:
            arr.microsecond=int(np.round(arr.microsecond/(1/sr*10**6))*1/sr*10**6)
        t1=arr-1
        t2=arr+9
        try:
            tr = client.get_waveforms(net, sta, "*", comp, t1-2, t2+2)
        except:
            print("No data for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
        else:
            print("Data available for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
            tr.detrend()
            tr.trim(starttime=t1-2, endtime=t2+2, nearest_sample=1, pad=1, fill_value=0)
            tr.filter("bandpass",freqmin=2,freqmax=7)
            tr.interpolate(sampling_rate=sr, starttime=t1)
            tr.trim(starttime=t1, endtime=t2, nearest_sample=1, pad=1, fill_value=0)
            st+=tr 
    return st

def get_daily_data(st,year,mo,day,sr):
    dayst=Stream()
    tr=Stream()
    # set client
    client = Client("IRIS")
    t1=UTCDateTime(year,mo,day)
    t2=t1+timedelta(days=1)
    for ii in range(len(st)):
        net=st[ii].stats.network
        sta=st[ii].stats.station
        comp=st[ii].stats.channel
        try:
            tr = client.get_waveforms(net, sta, "*", comp, t1-2, t2+2)
        except:
            print("No data for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
        else:
            print("Data available for "+net+" "+sta+" "+comp+" "+str(t1)+" "+str(t2))
            tr.detrend()
            tr.merge()
            #print(tr)
            if isinstance(tr[0].data, np.ma.masked_array):
                tr[0].data = tr[0].data.filled()
            tr.filter("bandpass",freqmin=2,freqmax=7)      
            tr.trim(starttime=t1-2, endtime=t2+2, nearest_sample=1, pad=1, fill_value=0)
            tr.interpolate(sampling_rate=sr, starttime=t1)
            tr.trim(starttime=t1, endtime=t2, nearest_sample=1, pad=1, fill_value=0)
            dayst+=tr 
    return dayst

def plot_templates_shifted(st):
    plt.figure(figsize=(6,10))
    for ii in range(len(st)):
        plt.plot(st[ii].times('relative'),st[ii].data/np.max(np.abs(st[ii].data))+ii)
    plt.xlim((0,10))
    
def plot_templates(st):
    plt.figure(figsize=(6,10))
    for ii in range(len(st)):
        plt.plot(st[ii].times('timestamp'),st[ii].data/np.max(np.abs(st[ii].data))+ii)
        
def plot_detections(st,dayst,shifts,curdect,sr):
    plt.figure(figsize=(10,10))
    winlen=30*sr    
    templen=len(st[0].data)
    if np.max(shifts)+templen > winlen:
        winlen=int(np.ceil((templen+np.max(shifts))/sr))*sr
    t=1/sr*np.arange(winlen)
    # plot template and detection relative to origin time
    stas=[]
    for ii in range(len(st)):
        clip=dayst[ii].data[curdect:curdect+winlen]
        stclip=st[ii].data
        plt.plot(t,clip/np.max(1.5*np.abs(clip))+ii,color=(0.5,0.5,0.5))
        plt.plot(t[shifts[ii]:shifts[ii]+templen],stclip/np.max(1.5*np.abs(stclip))+ii,color=(0.5,0.0,0.0),linestyle='--')
        stas.append(st[ii].stats.station+"-"+st[ii].stats.channel)
    plt.xlim((0,t[-1]))
    plt.yticks(range(len(st)), stas)
    plt.xlabel('Time (s)')
    
def sort_shift(df,st,dayst,sr):
    st.sort()
    dayst.sort()
    # check to make sure same length
    if len(st)>len(dayst):
        # remove extra traces from st
        for tr in st:
            if len(dayst.select(station=tr.stats.station, channel=tr.stats.channel))==0:
                st.remove(tr)
    # gets number of samples between template time and event origin time
    origintime=UTCDateTime(df['Date']+'T'+df['Time'])
    regional=df['Regional']
    eventid = regional+str(df['ID'])
    detail = get_event_by_id(eventid, includesuperseded=True)
    phases = get_phase_dataframe(detail, catalog=regional)
    phases = phases[phases['Status'] == 'manual']
    shifts=np.zeros(len(st),dtype=int)   
    for ii in range(len(phases)):
        net=phases.iloc[ii]['Channel'].split('.')[0]
        sta=phases.iloc[ii]['Channel'].split('.')[1]
        comp=phases.iloc[ii]['Channel'].split('.')[2]
        arr=UTCDateTime(phases.iloc[ii]['Arrival Time'])
        shift=int(np.round((arr-origintime)*sr))
        for jj in range(len(st)):
            if sta==st[jj].stats.station and comp==st[jj].stats.channel:
                print(sta+" "+comp+" "+str(shift))
                shifts[jj]=shift
    return shifts, st, dayst, phases

def mad(x):
    return np.median(np.abs(x-np.median(x)))

def clusterdects(dects,windowlen):
    dbscan_dataset1 = DBSCAN(eps=windowlen, min_samples=1, metric='euclidean').fit_predict(dects.reshape(-1, 1))
    dbscan_labels1 = dbscan_dataset1
    return dbscan_labels1

def culldects(dects,clusters,xcorr):
    newdect=np.empty(clusters[-1]+1,dtype=np.int)
    for ii in range(clusters[-1]+1):
#    print('ii='+str(ii))
        tinds=np.where(clusters==ii)[0]
#    print(tinds)
        dectinds=dects[tinds]
#    print(dectinds)
        values=xcorr[dectinds]
#    print(values)
#    print(np.argmax(values))
#    print(dects[np.argmax(values)])
        newdect[ii]=int(dectinds[np.argmax(values)])   
    return newdect

def plot_data_streams(st):
    plt.figure(figsize=(10,10))
    # plot template and detection relative to origin time
    stas=[]
    for ii in range(len(st)):
        print(ii)
        clip=st[ii].data
        t=st[ii].times('relative')
        plt.plot(t,clip/np.max(1.5*np.abs(clip))+ii,color=(0.5,0.5,0.5))
        if st[ii].stats.location=='P':
            print('its a P')
            plt.plot([10,10],[ii-0.5, ii+0.5],color=(0.5,0.0,0.0),linestyle='--')
        if st[ii].stats.location=='S':
            print('its a S')
            plt.plot([10,10],[ii-0.5, ii+0.5],color=(0.0,0.0,0.5),linestyle='--')
        stas.append(st[ii].stats.station+"-"+st[ii].stats.channel)
    plt.xlim((0,t[-1]))
    plt.yticks(range(len(st)), stas)
    plt.xlabel('Time (s)')
    return None

def check_phase_info(df):
    exists=1
    regional=df['Regional']
    eventid=regional+str(df['ID'])
    try:
        detail=get_event_by_id(eventid, includesuperseded=True)
    except:
        exists=0
    try: 
        phases = get_phase_dataframe(detail, catalog=regional)
    except:
        exists=0    
    return exists