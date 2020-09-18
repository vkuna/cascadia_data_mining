#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:36:50 2020

Makes figure 2

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
from scipy import signal
import unet_tools
import matplotlib.pylab as pl

# OPTIONS
subset=1 #True # train on a subset or the full monty?
epos=50 # how many epocs?
epsilon=1e-6
firstflag=1
sr=100
ponly=1
drop=0
std =0.2

fig2 = plt.figure(constrained_layout=True,figsize=(10,8))
gs = fig2.add_gridspec(2, 1)
ax1 = fig2.add_subplot(gs[0:1, 0])
ax2 = fig2.add_subplot(gs[1, 0])
colors = [[0.5,0,0],
          [0,0.5,0],
          [0,0,0.5],
          [0.5,0.5,0]]
for ii,large in enumerate([0.5,1,2,4]):       
    if ponly==1:
        model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
    else:
        model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"     
    
    if large:
        fac=large
        model_save_file="large_"+str(fac)+"_"+model_save_file
    
    if drop:
        model_save_file="drop_"+model_save_file
    
    # training stats
    training_stats = np.genfromtxt("./result_files/"+model_save_file+'.csv', delimiter=',',skip_header=1)
    ax1.plot(training_stats[:,0],training_stats[:,2],label='Size '+str(large)+ ' network loss',color=colors[ii],linestyle='dashed')
    ax1.plot(training_stats[:,0],training_stats[:,4],label='Size '+str(large)+ ' network validation loss',color=colors[ii],linestyle='solid')
    ax1.set_xlim((0,49))
    ax1.legend(loc='upper right',ncol=2,prop={'size': 12})
    ax1.set_ylabel('Loss',fontsize=14)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.text(1,.0215,'A',fontsize=28,fontweight='bold')
    ax2.plot(training_stats[:,0],training_stats[:,4]/training_stats[:,2],color=colors[ii],label='Size '+str(large))
    print(training_stats[-1,4]/training_stats[-1,2])
    ax2.set_xlabel('Epoch',fontsize=14)
    ax2.set_ylabel('Loss Ratio',fontsize=14)
    ax2.legend(loc='upper left',ncol=1,prop={'size': 12})
    ax2.set_xlim((0,49))
    ax2.text(1,.86,'B',fontsize=28,fontweight='bold')
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
fig2.savefig("figure2.png")