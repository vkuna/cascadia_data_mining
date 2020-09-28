#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:26:08 2020

@author: amt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

epos=50

colors = pl.cm.jet(np.linspace(0,1,12))

for sr in [100]:
    for large in [0.5]:
        c=0
        f, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,10))
        ax1.set_title('Large='+str(large)+"  Sample Rate="+str(sr))
        for ponly in [0, 1]:
            for drop in [0, 1]:
                for std in [0.05, 0.1, 0.2]:
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
                    box = ax1.get_position()
                    ax1.set_position([box.x0, box.y0, box.width, box.height])
                    ax1.plot(training_stats[:,0],training_stats[:,2], color=colors[c], linestyle='--', label='Loss')
                    ax1.plot(training_stats[:,0],training_stats[:,4], color=colors[c], linestyle='-', label='Val_Loss')
                    ax1.set_xlim((0,49))
                    if c==0:
                        ax1.legend()
    
                    ax2.plot(training_stats[:,0],training_stats[:,4]/training_stats[:,2], color=colors[c], label='Loss '+str(ponly)+'_'+str(drop)+'_'+str(std))
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylim((0.5,1.8))
                    ax2.legend(loc='lower right')
                    ax2.set_xlim((0,49))
                    c+=1


# # STD
# large=1     
# stdres=np.zeros((8,2))
# c=-1               
# for sr in [40, 100]:
#     for ponly in [0, 1]:
#         for drop in [0, 1]:
#             c+=1
#             for std in [0.1, 0.2]:
#                 if std==0.1:
#                     ind=0
#                 else:
#                     ind=1
#                 print(str(ponly)+"_sr_"+str(sr)+"_std_"+str(std))
#                 if ponly==1:
#                     model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
#                 else:
#                     model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
#                 if large:
#                     fac=large
#                     model_save_file="large_"+str(fac)+"_"+model_save_file
#                 if drop:
#                     model_save_file="drop_"+model_save_file
                    
#                 training_stats = np.genfromtxt(model_save_file+'.csv', delimiter=',',skip_header=1)
#                 stdres[c,ind]=training_stats[-1,4]
# print(stdres)
                
# # DROP
# large=1     
# dropres=np.zeros((8,2))
# c=-1               
# for sr in [40, 100]:
#     for ponly in [1, 0]:
#         for std in [0.1, 0.2]:
#             c+=1
#             for drop in [0, 1]:
#                 print("ponly_"+str(ponly)+"_sr_"+str(sr)+"_std_"+str(std))
#                 if ponly==1:
#                     model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
#                 else:
#                     model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
#                 if large:
#                     fac=large
#                     model_save_file="large_"+str(fac)+"_"+model_save_file
#                 if drop:
#                     model_save_file="drop_"+model_save_file
                    
#                 training_stats = np.genfromtxt(model_save_file+'.csv', delimiter=',',skip_header=1)
#                 dropres[c,drop]=training_stats[-1,4]
# print(dropres)

# # DROP
# large=1     
# pres=np.zeros((8,2))
# c=-1               
# for sr in [40, 100]:
#     for std in [0.1, 0.2]:
#         for drop in [0, 1]:
#             c+=1
#             for ponly in [1, 0]:
#                 print("ponly_"+str(ponly)+"_sr_"+str(sr)+"_std_"+str(std))
#                 if ponly==1:
#                     model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
#                 else:
#                     model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
#                 if large:
#                     fac=large
#                     model_save_file="large_"+str(fac)+"_"+model_save_file
#                 if drop:
#                     model_save_file="drop_"+model_save_file
                    
#                 training_stats = np.genfromtxt(model_save_file+'.csv', delimiter=',',skip_header=1)
#                 pres[c,ponly]=training_stats[-1,4]
# print(dropres)

# DROP
large=1     
pres=np.zeros((12,2))
c=-1               
for sr in [100]:
    for drop in [1,0]:
        for std in [0.05, 0.1, 0.2]:
            c+=1
            for ponly in [0]:
                if drop==1:
                    print("drop_ponly_"+str(ponly)+"_sr_"+str(sr)+"_std_"+str(std))
                else:
                    print("ponly_"+str(ponly)+"_sr_"+str(sr)+"_std_"+str(std))
                if ponly==1:
                    model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
                else:
                    model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"  
                if large:
                    fac=large
                    model_save_file="large_"+str(fac)+"_"+model_save_file
                if drop:
                    model_save_file="drop_"+model_save_file
                    
                training_stats = np.genfromtxt("./result_files/"+model_save_file+'.csv', delimiter=',',skip_header=1)
                pres[c,ponly]=np.round(100000*training_stats[-1,4])/100000
print(pres)