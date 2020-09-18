#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:12:43 2020

@author: amt
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

N=1500
sr=100

for std in [0.05, 0.1, 0.2]:
    label=signal.gaussian(N,std=int(std*sr))+1e-320
    guess=np.roll(label,2)+1e-320
    plt.plot(label)
    plt.plot(guess)
    bce=-1*np.mean(label*np.log(guess)+(1-label)*np.log(1-guess))
    print(bce)