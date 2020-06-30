#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:14:23 2020

Train a CNN to pick P and S wave arrivals

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
from scipy import signal


# some params
plots=0
train=1
ponly=0
epos=3
if ponly:
    model_save_file="./unet_250000_pn_eps_"+str(epos)+"_std_0.2.tf"
    n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
    p_data, _ = pickle.load( open( 'pnsn_ncedc_P_training_data.pkl', 'rb' ) )
else:
    model_save_file="./unet_250000_sn_eps_"+str(epos)+"_std_0.2.tf"
    n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
    s_data, _ = pickle.load( open( 'pnsn_ncedc_S_training_data.pkl', 'rb' ) )

# plot the data
if plots:
    # plot ps to check
    plt.figure()
    for ii in range(100):
        plt.plot(p_data[ii,:]/np.max(np.abs(p_data[ii,:])))
        
    # # plot ss to check
    # plt.figure()
    # for ii in range(10):
    #     plt.plot(s_data[ii,:])
        
    # plot noise to check
    plt.figure()
    for ii in range(100):
        plt.plot(n_data[ii,:]/np.max(np.abs(n_data[ii,:])))

# make inputs
if ponly:
    features=np.concatenate((n_data,p_data))
else:
    features=np.concatenate((n_data,s_data))
    
# make target vector N=0, P=1, S=2
if ponly:
    target=np.concatenate((np.zeros(n_data.shape[0]),np.ones(p_data.shape[0])))
else:
    target=np.concatenate((np.zeros(n_data.shape[0]),np.ones(s_data.shape[0])))

# make training and testing data
inds=np.arange(target.shape[0])
np.random.shuffle(inds)
train_inds=inds[:int(0.75*len(inds))]
test_inds=inds[int(0.75*len(inds)):]
x_train=features[train_inds,:]
y_train=target[train_inds]
x_test=features[test_inds,:]
y_test=target[test_inds]

# do the shifts and make batches
def my_data_generator(batch_size,dataset,targets):
    while True:
        start_of_batch=np.random.choice(dataset.shape[0]-batch_size)
        #print('start of batch: '+str(start_of_batch))
        # grab batch
        batch=dataset[start_of_batch:start_of_batch+batch_size,:]
        # make target data for batch
        batch_target=np.zeros_like(batch)
        # some params
        winsize=15 # winsize in seconds
        sr=40 # sample rate
        # this just makes a nonzero value where the pick is
        # batch_target[:, batch_target.shape[1]//2]=targets[start_of_batch:start_of_batch+batch_size]
        for ii, targ in enumerate(targets[start_of_batch:start_of_batch+batch_size]):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/dataset.shape[1]*np.ones((1,dataset.shape[1]))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(dataset.shape[1],std=int(0.2*sr))
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        new_batch=np.zeros((batch_size,int(winsize*sr)))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) # keep 4s worth of samples
            new_batch[ii,:]=batch[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        yield(new_batch,new_batch_target)

# generate batch data
my_data=my_data_generator(32,x_train,y_train)
x,y=next(my_data)

# PLOT GENERATOR RESULTS
if plots:
    for ind in range(15):
        fig, ax1 = plt.subplots()
        t=1/40*np.arange(x.shape[1])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude', color='tab:red')
        ax1.plot(t, x[ind,:], color='tab:red') #, label='data')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Prediction', color='tab:blue')  # we already handled the x-label with ax1
        ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
        ax2.tick_params(axis='y')
        ax2.set_ylim((-0.1,2.1))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(('prediction','target'))
        plt.show()

# BUILD THE MODEL
#These models start with an input
input_layer=tf.keras.layers.Input(shape=(600,)) # 1 Channel seismic data

#These Convolutional blocks expect 2D data (time-steps x channels)
#This is just one channel, but if you wanted to add more stations as extra channels you can
network=tf.keras.layers.Reshape((600,1))(input_layer)

# Batch normalization
network=tf.keras.layers.BatchNormalization()(network)

# build the network, here is your first convolution layer
level1=tf.keras.layers.Conv1D(32,21,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding

# This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
network=tf.keras.layers.BatchNormalization()(level1)

#Max Pooling Layer
network=tf.keras.layers.MaxPooling1D()(network) #300

#Next Block
level2=tf.keras.layers.Conv1D(64,15,activation='relu',padding='same')(network)
network=tf.keras.layers.BatchNormalization()(level2)
network=tf.keras.layers.MaxPooling1D()(network) #150

#Next Block
level3=tf.keras.layers.Conv1D(128,11,activation='relu',padding='same')(network)
network=tf.keras.layers.BatchNormalization()(level3)
network=tf.keras.layers.MaxPooling1D()(network) #75

#Base of Network
network=tf.keras.layers.Flatten()(network)
base_level=tf.keras.layers.Dense(75,activation='relu')(network)

network=tf.keras.layers.BatchNormalization()(base_level)
network=tf.keras.layers.Reshape((75,1))(network)

#Upsample and add skip connections
network=tf.keras.layers.Conv1D(128,11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
network=tf.keras.layers.UpSampling1D()(network)
level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding

#Upsample and add skip connections
network=tf.keras.layers.Conv1D(64,15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
network=tf.keras.layers.UpSampling1D()(network)
level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding

#Upsample and add skip connections
network=tf.keras.layers.Conv1D(32,21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
network=tf.keras.layers.UpSampling1D()(network)
level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
#End of network
network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding

model=tf.keras.models.Model(input_layer,output)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

# TRAIN THE MODEL
if train:
    batch_size=32
    history=model.fit_generator(my_data_generator(batch_size,x_train,y_train),
                        steps_per_epoch=len(x_train)//batch_size,
                        validation_data=my_data_generator(batch_size,x_test,y_test),
                        validation_steps=len(x_test)//batch_size,
                        epochs=epos)
    model.save_weights(model_save_file)
else:
    model.load_weights(model_save_file)

# # See how things went
# my_test_data=my_data_generator(10,x_test,y_test)
# x,y=next(my_test_data)

# test_predictions=model.predict(x)

# # PLOT A FEW EXAMPLES
# for ind in range(10):
#     fig, ax1 = plt.subplots()
#     t=1/40*np.arange(x.shape[1])
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Amplitude')
#     ax1.plot(t, x[ind,:], color='tab:red') #, label='data')
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
    
# # PLOT PREDICTIONS
# plt.figure()
# for ind in range(10):
#     plt.plot(t,test_predictions[ind,:])

