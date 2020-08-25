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
import unet_tools
from sklearn.metrics import accuracy_score, precision_score, recall_score

# OPTIONS
subset=0 #True # train on a subset or the full monty?
epos=50 # how many epocs?
epsilon=1e-6
firstflag=1

for sr in [100]: #, 100]:
    # LOAD THE DATA
    for ponly in [0,1]: # 1 - P+Noise, 2 - S+noise    
        print("LOADING DATA")
        if sr==40:
            if ponly:
                n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
                x_data, _ = pickle.load( open( 'pnsn_ncedc_P_training_data.pkl', 'rb' ) ) 
            else:
                n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
                x_data, _ = pickle.load( open( 'pnsn_ncedc_S_training_data.pkl', 'rb' ) ) 
        elif sr==100:
            if ponly:
                n_data, _ = pickle.load( open( 'pnsn_ncedc_N_100_training_data.pkl', 'rb' ) )
                x_data, _ = pickle.load( open( 'pnsn_ncedc_P_100_training_data.pkl', 'rb' ) ) 
            else:
                n_data, _ = pickle.load( open( 'pnsn_ncedc_N_100_training_data.pkl', 'rb' ) )
                x_data, _ = pickle.load( open( 'pnsn_ncedc_S_100_training_data.pkl', 'rb' ) ) 
        
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
                    end_bin=start_bin+int(winsize*sr) # keep 4s worth of samples
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
        
        
        for drop in [0,1]: # True # drop?
            for large in [1]: # large unet
                for std in [0.05]: #[0.1, 0.2]: # how long do you want the gaussian STD to be?
                
                    print("ponly "+str(ponly))
                    print("drop "+str(drop))
                    print("large "+str(large))
                    print("epos "+str(epos))
                    print("std "+str(std))
                    print("sr "+str(sr))
                    
                    # generate batch data
                    print("FIRST PASS WITH DATA GENERATOR")
                    my_data=my_data_generator(32,x_train,y_train,sr,std)
                    x,y=next(my_data)
                    
                    if ponly==1:
                        model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
                    else:
                        model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"     
            
                    if large:
                        fac=large
                        model_save_file="large_"+str(fac)+"_"+model_save_file
                    
                    if drop:
                        model_save_file="drop_"+model_save_file
                    
                    # BUILD THE MODEL
                    print("BUILD THE MODEL")
                    if drop:
                        model=unet_tools.make_large_unet_drop(fac,sr)    
                    else:
                        model=unet_tools.make_large_unet(fac,sr)
                    
                    # LOAD THE MODEL
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
                    
                    # GET PERFORMANCE STATS
                    # this plots accuracy, precision, and recall for each model
                    if firstflag==1:
                        my_test_data=my_data_generator(10000,x_test,y_test,sr,std)
                        x,y=next(my_test_data)
                        firstflag==0
                    test_predictions=model.predict(x)
                    threshs=np.arange(0.01,1,0.01)
                    metrics=np.zeros((len(threshs),3))
                    for ii,thresh in enumerate(threshs):
                        y_true=np.where(np.max(y,axis=1) > thresh, 1, 0)
                        y_pred=np.where(np.max(test_predictions,axis=1) > thresh, 1, 0)
                        metrics[ii,0]=accuracy_score(y_true,y_pred)
                        metrics[ii,1]=precision_score(y_true,y_pred)
                        metrics[ii,2]=recall_score(y_true,y_pred)
                    
                    plt.figure()
                    plt.plot(threshs,metrics[:,1])
                    plt.plot(threshs,metrics[:,2])
                    plt.plot(threshs,metrics[:,0])
                    plt.legend(('precision','recall','accuracy'))
                    plt.title(model_save_file)
                    plt.xlabel('Threshold value')
                    plt.ylabel('Percent')
                    plt.savefig('pra_'+model_save_file+'.png')
                    
                    # this looks at pick accuracy
                    y_true=np.where(np.max(y,axis=1) > 0.75, 1, 0)
                    y_pred=np.where(np.max(test_predictions,axis=1) > 0.75, 1, 0)
                    inds=np.where(y_true+y_pred==2)[0]
                    pickdiff=np.zeros(len(inds))
                    for ii,ind in enumerate(inds):
                        pickdiff[ii]=np.where(test_predictions[ind]==np.max(test_predictions[ind]))[0][0]-np.where(y[ind]==np.max(y[ind]))[0][0]
                    plt.figure()
                    plt.hist(pickdiff,bins=np.arange(-10.5,10.5,1), alpha=0.5, rwidth=0.8, density=True)
                    plt.xticks((np.arange(-10, 12, step=2)))
                    plt.xlabel("Predicted pick - Actual pick (samples)")
                    plt.ylabel("Probability")
                    plt.title(model_save_file)
                    plt.text(-10,0.1,'%< 4 samples='+str(np.round(100*len(np.where(np.abs(pickdiff)<=4)[0])/len(pickdiff))/100))
                    if sr==40:
                        plt.text(-10,0.05,'%< 0.1 seconds='+str(np.round(100*len(np.where(np.abs(pickdiff)<=4)[0])/len(pickdiff))/100))  
                    if sr==100:
                        plt.text(-10,0.05,'%< 0.1 seconds='+str(np.round(100*len(np.where(np.abs(pickdiff)<=10)[0])/len(pickdiff))/100)) 
                    plt.savefig('pick_accuracy_'+model_save_file+'.png')
                    
                    # training stats
                    training_stats = np.genfromtxt(model_save_file+'.csv', delimiter=',',skip_header=1)
                    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    ax1.plot(training_stats[:,0],training_stats[:,1])
                    ax1.plot(training_stats[:,0],training_stats[:,3])
                    ax1.legend(('acc','val_acc'))
                    ax2.plot(training_stats[:,0],training_stats[:,2])
                    ax2.plot(training_stats[:,0],training_stats[:,4])
                    ax2.legend(('loss','val_loss'))
                    ax2.set_xlabel('Epoch')
                    ax1.set_title(model_save_file)
                    f.savefig('training_stats_'+model_save_file+'.png')
                    # len(np.where(np.abs(pickdiff)<=4)[0])/len(pickdiff)