#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:27:07 2020

Train a CNN to pick P and S wave arrivals with log features

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
talapas=1
if talapas:
    subset=int(sys.argv[1]) #True # train on a subset or the full monty?
    ponly=int(sys.argv[2]) #True # 1 - P+Noise, 2 - S+noise
    train=int(sys.argv[3]) #True # do you want to train?
    drop=int(sys.argv[4]) #True # drop?
    plots=int(sys.argv[5]) #False # do you want to make some plots?
    resume=int(sys.argv[6]) #False # resume training
    large=int(sys.argv[7]) # large unet
    epos=int(sys.argv[8]) # how many epocs?
    std=float(sys.argv[9]) # how long do you want the gaussian STD to be?
    sr=int(sys.argv[10])
else:
    subset=0 #True # train on a subset or the full monty?
    ponly=1 #True # 1 - P+Noise, 2 - S+noise
    train=1 #True # do you want to train?
    drop=0 #True # drop?
    plots=0 #False # do you want to make some plots?
    resume=0 #False # resume training
    large=1 # large unet
    epos=50 # how many epocs?
    std=0.1 # how long do you want the gaussian STD to be?
    sr=100

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
        
if large:
    fac=large
    model_save_file="large_"+str(fac)+"_"+model_save_file

if drop:
    model_save_file="drop_"+model_save_file

# plot the data
if plots:
    # plot ps to check
    plt.figure()
    for ii in range(100):
        plt.plot(x_data[ii,:]/np.max(np.abs(x_data[ii,:])))
        
    # plot noise to check
    plt.figure()
    for ii in range(100):
        plt.plot(n_data[ii,:]/np.max(np.abs(n_data[ii,:])))

# MAKE FEATURES AND TARGET VECTOR N=0, P/S=2
print("MAKE FEATURES AND TARGET VECTOR")
features=np.concatenate((n_data,x_data))
target=np.concatenate((np.zeros(n_data.shape[0]),np.ones(x_data.shape[0])))
del n_data
del x_data

# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
np.random.seed(0)
inds=np.arange(target.shape[0])
np.random.shuffle(inds)
train_inds=inds[:int(0.75*len(inds))]
test_inds=inds[int(0.75*len(inds)):]
x_train=features[train_inds,:]
y_train=target[train_inds]
x_test=features[test_inds,:]
y_test=target[test_inds]

# do the shifts and make batches
print("SETTING UP GENERATOR")
def my_data_generator(batch_size,dataset,targets,sr,std,valid=False):
    while True:
        start_of_batch=np.random.choice(dataset.shape[0]-batch_size)
        if valid:
            start_of_batch=0
        #print('start of batch: '+str(start_of_batch))
        # grab batch
        batch=dataset[start_of_batch:start_of_batch+batch_size,:]
        # make target data for batch
        batch_target=np.zeros_like(batch)
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        # batch_target[:, batch_target.shape[1]//2]=targets[start_of_batch:start_of_batch+batch_size]
        for ii, targ in enumerate(targets[start_of_batch:start_of_batch+batch_size]):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/dataset.shape[1]*np.ones((1,dataset.shape[1]))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(dataset.shape[1],std=int(std*sr))
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        new_batch=np.zeros((batch_size,int(winsize*sr)))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:]=batch[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:].reshape(-1,1), new_batch_sign[ii,:].reshape(-1,1) ] ) )
        batch_out=np.array(batch_out)
        yield(batch_out,new_batch_target)

# generate batch data
print("FIRST PASS WITH DATA GENERATOR")
my_data=my_data_generator(32,x_train,y_train,sr,std)
x,y=next(my_data)

# PLOT GENERATOR RESULTS
if plots:
    for ind in range(5):
        fig, ax1 = plt.subplots()
        t=1/40*np.arange(x.shape[1])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude', color='tab:red')
        ax1.plot(t, x[ind,:,0], color='tab:red') #, label='data')
        ax1.plot(t, x[ind,:,1], color='tab:red') #, label='data')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Prediction', color='black')  # we already handled the x-label with ax1
        ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
        ax2.plot(t, x[ind,:,0], color='tab:green') #, label='data')
        ax2.plot(t, x[ind,:,1], color='tab:blue') #, label='data')
        ax2.tick_params(axis='y')
        #ax2.set_ylim((-0.1,2.1))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(('target1','target2'))
        plt.show()

# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=unet_tools.make_large_unet_drop(fac,sr)    
else:
    model=unet_tools.make_large_unet(fac,sr)
        
# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    monitor='val_acc', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    batch_size=32
    if resume:
        print('Resuming training results from '+model_save_file)
        model.load_weights(checkpoint_filepath)
    else:
        print('Training model and saving results to '+model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)
    history=model.fit_generator(my_data_generator(batch_size,x_train,y_train,sr,std),
                        steps_per_epoch=len(x_train)//batch_size,
                        validation_data=my_data_generator(batch_size,x_test,y_test,sr,std),
                        validation_steps=len(x_test)//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./"+model_save_file)

# # See how things went
# my_test_data=my_data_generator(20,x_test,y_test,valid=True)
# x,y=next(my_test_data)

# test_predictions=model.predict(x)

# # PLOT A FEW EXAMPLES
# for ind in range(20):
#     fig, ax1 = plt.subplots()
#     t=1/40*np.arange(x.shape[1])
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Amplitude')
#     trace=np.multiply(np.power(x[ind,:,0],10),x[ind,:,1])
#     ax1.plot(t, trace, color='tab:red') #, label='data')
#     ax1.tick_params(axis='y')
#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     ax2.set_ylabel('Prediction')  # we already handled the x-label with ax1
#     ax2.plot(t, test_predictions[ind,:], color='tab:blue') #, label='prediction')
#     ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
#     ax2.tick_params(axis='y')
#     ax2.set_ylim((-0.1,2.1))
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.legend(('prediction','target'))
#     plt.show()
