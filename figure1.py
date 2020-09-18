#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:59:46 2020

Make figure 1 data

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
from scipy import signal
import sys
import unet_tools

# OPTIONS
subset=1 #True # train on a subset or the full monty?
ponly=1 #True # 1 - P+Noise, 2 - S+noise
train=1 #True # do you want to train?
drop=0 #True # drop?
plots=1 #False # do you want to make some plots?
resume=0 #False # resume training
large=1 # large unet
epos=50 # how many epocs?
std=0.1 # how long do you want the gaussian STD to be?
sr=40

epsilon=1e-6

print("subset "+str(subset))
print("ponly "+str(ponly))
print("train "+str(train))
print("drop "+str(drop))
print("plots "+str(plots))
print("resume "+str(resume))
print("large "+str(large))
print("epos "+str(epos))
print("std "+str(std))
print("sr "+str(sr))

# LOAD THE DATA
print("LOADING DATA")
if sr==40:
    if ponly:
        if subset:
            n_data, _ = pickle.load( open( 'pnsn_N_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_P_training_data.pkl', 'rb' ) )  
            model_save_file="unet_logfeat_10000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
        else:
            n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_ncedc_P_training_data.pkl', 'rb' ) ) 
            model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
    else:
        if subset:
            n_data, _ = pickle.load( open( 'pnsn_N_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_S_training_data.pkl', 'rb' ) )  
            model_save_file="unet_logfeat_10000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
        else:
            n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_ncedc_S_training_data.pkl', 'rb' ) ) 
            model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
elif sr==100:
    if ponly:
        if subset:
            n_data, _ = pickle.load( open( 'pnsn_N_100_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_P_100_training_data.pkl', 'rb' ) )  
            model_save_file="unet_logfeat_10000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
        else:
            n_data, _ = pickle.load( open( 'pnsn_ncedc_N_100_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_ncedc_P_100_training_data.pkl', 'rb' ) ) 
            model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
    else:
        if subset:
            n_data, _ = pickle.load( open( 'pnsn_N_100_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_S_100_training_data.pkl', 'rb' ) )  
            model_save_file="unet_logfeat_10000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
        else:
            n_data, _ = pickle.load( open( 'pnsn_ncedc_N_100_training_data.pkl', 'rb' ) )
            x_data, _ = pickle.load( open( 'pnsn_ncedc_S_100_training_data.pkl', 'rb' ) ) 
            model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 

# plot the data
np.random.seed(11)
import matplotlib.gridspec as gridspec
if plots:
    fig1 = plt.figure(constrained_layout=True,figsize=(10,10))
    gs = fig1.add_gridspec(3, 1)
    f1_ax1 = fig1.add_subplot(gs[0:2, 0])
    
    # plot template and detection relative to origin time
    inds=np.random.choice(len(x_data),20)
    t=1/sr*np.arange(x_data.shape[1])
    for ii,ind in enumerate(inds):
        print(ii)
        clip=x_data[ind,:]
        f1_ax1.plot(t,clip/np.max(1*np.abs(clip))+ii,color=(0.5,0.5,0.5))
    
    f1_ax1.set_xlim((0,30))
    f1_ax1.set_ylim((-1.5,20.5))
    f1_ax1.set_yticks([], [])
    f1_ax1.text(0.4,19.,'A',fontsize=28,fontweight='bold')
    f1_ax1.tick_params(axis="x", labelsize=12)
    
    f1_ax2 = fig1.add_subplot(gs[2, 0])
    f1_ax2.set_xlabel('Time (s)',fontsize=14)
    f1_ax2.set_xlim((0,30))

    
    clip=x_data[inds[-12],:]
    f1_ax2.plot(t,signal.gaussian(len(clip),std=int(sr*0.05)),label='$\sigma$=0.05',color=(0.5,0.,0.),linestyle='dashdot')
    f1_ax2.plot(t,signal.gaussian(len(clip),std=int(sr*0.1)),label='$\sigma$=0.1',color=(0.,0.5,0.),linestyle='dashed')
    f1_ax2.plot(t,signal.gaussian(len(clip),std=int(sr*0.2)),label='$\sigma$=0.2',color=(0.,0.,0.5),linestyle='dotted')
    f1_ax2.set_ylabel("Target Amplitude",fontsize=14)
    f1_ax2b = f1_ax2.twinx() 
    f1_ax2b.set_yticks([], [])

    f1_ax2b.plot(t,clip/np.max(1*np.abs(clip)),color=(0.5,0.5,0.5))
    f1_ax2.set_xlim((11,19))
    f1_ax2.tick_params(axis="x", labelsize=12)
    f1_ax2.tick_params(axis="y", labelsize=12)
    f1_ax2.legend(loc='lower left',prop={'size': 12})
    f1_ax2.text(11.1,0.9,'B',fontsize=28,fontweight='bold')
    
fig1.savefig("figure1.png")

# # MAKE FEATURES AND TARGET VECTOR N=0, P/S=2
# print("MAKE FEATURES AND TARGET VECTOR")
# features=np.concatenate((n_data,x_data))
# target=np.concatenate((np.zeros(n_data.shape[0]),np.ones(x_data.shape[0])))
# del n_data
# del x_data

# # MAKE TRAINING AND TESTING DATA
# print("MAKE TRAINING AND TESTING DATA")
# np.random.seed(0)
# inds=np.arange(target.shape[0])
# np.random.shuffle(inds)
# train_inds=inds[:int(0.75*len(inds))]
# test_inds=inds[int(0.75*len(inds)):]
# x_train=features[train_inds,:]
# y_train=target[train_inds]
# x_test=features[test_inds,:]
# y_test=target[test_inds]

# # do the shifts and make batches
# print("SETTING UP GENERATOR")
# def my_data_generator(batch_size,dataset,targets,sr,std,valid=False):
#     while True:
#         start_of_batch=np.random.choice(dataset.shape[0]-batch_size)
#         if valid:
#             start_of_batch=0
#         #print('start of batch: '+str(start_of_batch))
#         # grab batch
#         batch=dataset[start_of_batch:start_of_batch+batch_size,:]
#         # make target data for batch
#         batch_target=np.zeros_like(batch)
#         # some params
#         winsize=15 # winsize in seconds
#         # this just makes a nonzero value where the pick is
#         # batch_target[:, batch_target.shape[1]//2]=targets[start_of_batch:start_of_batch+batch_size]
#         for ii, targ in enumerate(targets[start_of_batch:start_of_batch+batch_size]):
#             #print(ii,targ)
#             if targ==0:
#                 batch_target[ii,:]=1/dataset.shape[1]*np.ones((1,dataset.shape[1]))
#             elif targ==1:
#                 batch_target[ii,:]=signal.gaussian(dataset.shape[1],std=int(std*sr))
#         # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
#         time_offset=np.random.uniform(0,winsize,size=batch_size)
#         new_batch=np.zeros((batch_size,int(winsize*sr)))
#         new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
#         for ii,offset in enumerate(time_offset):
#             bin_offset=int(offset*sr) #HZ sampling Frequency
#             start_bin=bin_offset 
#             end_bin=start_bin+int(winsize*sr) 
#             new_batch[ii,:]=batch[ii,start_bin:end_bin]
#             new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
#         # does feature log
#         new_batch_sign=np.sign(new_batch)
#         new_batch_val=np.log(np.abs(new_batch)+epsilon)
#         batch_out=[]
#         for ii in range(new_batch_target.shape[0]):
#             batch_out.append(np.hstack( [new_batch_val[ii,:].reshape(-1,1), new_batch_sign[ii,:].reshape(-1,1) ] ) )
#         batch_out=np.array(batch_out)
#         yield(batch_out,new_batch_target)

# # generate batch data
# print("FIRST PASS WITH DATA GENERATOR")
# my_data=my_data_generator(32,x_train,y_train,sr,std)
# x,y=next(my_data)

# # PLOT GENERATOR RESULTS
# if plots:
#     for ind in range(5):
#         fig, ax1 = plt.subplots()
#         t=1/40*np.arange(x.shape[1])
#         ax1.set_xlabel('Time (s)')
#         ax1.set_ylabel('Amplitude', color='tab:red')
#         ax1.plot(t, x[ind,:,0], color='tab:red') #, label='data')
#         ax1.plot(t, x[ind,:,1], color='tab:red') #, label='data')
#         ax1.tick_params(axis='y')
#         ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#         ax2.set_ylabel('Prediction', color='black')  # we already handled the x-label with ax1
#         ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
#         ax2.plot(t, x[ind,:,0], color='tab:green') #, label='data')
#         ax2.plot(t, x[ind,:,1], color='tab:blue') #, label='data')
#         ax2.tick_params(axis='y')
#         #ax2.set_ylim((-0.1,2.1))
#         fig.tight_layout()  # otherwise the right y-label is slightly clipped
#         plt.legend(('target1','target2'))
#         plt.show()