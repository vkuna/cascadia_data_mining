#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:11:19 2020

This makes training data for a CNN from PNSN

@author: amt
"""

import numpy as np
import pandas as pd
import cnn_tools
from obspy import Stream, read
import matplotlib.pyplot as plt
import random
import pickle

# # load catalogs and make dataframes
# pd.set_option('display.max_columns', None)

# # PNSN Catalog format: 0Evid,1Magnitude,2Magnitude Type,3Epoch(UTC),4Time UTC,5Time Local,6Distance From,7Lat,8Lon,9Depth Km,10Depth Mi
# pnsn_cat=np.genfromtxt('pnsn_2005_2020.cat', delimiter=',', skip_header=1, 
#                         usecols=(0,1,4,8,9,10), dtype=("i8", float,"|U19", float, float, float))
# df= pd.DataFrame(pnsn_cat, columns=['ID','Date','Time','Magnitude','Lat','Lon','Depth','Regional'])
# for ii in range(len(pnsn_cat)):
#     print(str(ii)+' of '+str(len(pnsn_cat)))
#     df= df.append({'ID': pnsn_cat[ii][0], 'Magnitude': pnsn_cat[ii][1], 'Date': pnsn_cat[ii][2][:10], 'Time': pnsn_cat[ii][2][11:],
#                               'Lat': pnsn_cat[ii][3], 'Lon': pnsn_cat[ii][4], 'Depth': pnsn_cat[ii][5], 'Regional': 'uw'}, ignore_index=True)

# # NCEDC Catalog format: 0Date, 1Time, 2Lat, 3Lon, 4Depth, 5Mag, 6Magt, 7Nst, 8Gap, 9Clo, 10RMS, 11SRC, 12Event ID
# ncedc_cat=np.genfromtxt('ncedc_2005_2020.cat', delimiter=' ', skip_header=13, 
#                         usecols=(0,1,2,3,4,5,12), dtype=("|U10", "|U11", float, float, float, float, "i8"))
# #ncedc_df= pd.DataFrame(ncedc_cat, columns=['ID','Date','Time','Magnitude','Lat','Lon','Depth','Regional'])
# for ii in range(len(ncedc_cat)): #len(ncedc_cat)):
#     print(str(ii)+' of '+str(len(ncedc_cat)))
#     df= df.append({'ID': ncedc_cat[ii][6], 'Magnitude': ncedc_cat[ii][5], 'Date': ncedc_cat[ii][0], 'Time': ncedc_cat[ii][1],
#                               'Lat': ncedc_cat[ii][2], 'Lon': ncedc_cat[ii][3], 'Depth': ncedc_cat[ii][4], 'Regional': 'nc'}, ignore_index=True)

# # Save/read pickle
# df.to_pickle("pnsn_ncedc_2005_2020.pkl")
df = pd.read_pickle("pnsn_ncedc_2005_2020.pkl")

# set upts
sr=40 # sample rate for datad
winsize=30 # seconds

# make templates
for myphases in ['P']: #,'S','N']:
    stall=np.array([]).reshape(0,winsize*sr+1)
    porsall=np.array([])
    inds = list(range(0,4)) #list(range(0, len(df)))
    random.shuffle(inds) # which index from dataframe do you want to look at?
    ii = 0
    while len(stall) < 250000:
        if ii == len(inds):
            break
        else:
            print("Index: "+str(inds[ii]), flush=True)
            evdf=df.iloc[inds[ii],:] # get template stats 
            checkphaseinfo=cnn_tools.check_phase_info(evdf)
            if checkphaseinfo:  
                stout, porsout=cnn_tools.make_training_data(evdf,sr,winsize,myphases)      
                stall=np.vstack([stall, stout])
                porsall=np.hstack([porsall, porsout])
                print(str(stall.shape[0])+" TOTAL TRACES", flush=True)
            ii+=1
    
    # plot the data
    cnn_tools.plot_training_data(stall,sr,porsall,winsize)
    
    # save the data
    with open('pnsn_ncedc_'+myphases+'_training_data.pkl', 'wb') as f:
        pickle.dump([stall, porsall], f)

