# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 00:04:18 2018

@author: gwenda
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from neuronpy.graphics import spikeplot


"""
two varibles
V_ADP
V_inh
"""
V_rest = -60
V_thresh = -50

tau_inh = 5
tau_ADP = 200
A_inh = -4
A_ADP = 10
V_ADP = 0 
V_inh = 0
k_inh = math.exp(-1.0/tau_inh)
k_adp = math.exp(-1.0/tau_ADP)

spiketrain = []

for time in range(2000):
    
    V_ADP = V_ADP*k_ADP
    V_inh = V_inh*k_inh
    
    V = V_rest + V_ADP + V_inh
    
    if V > V_thresh:
        V = V_rest
        V_inh = V_inh + A_inh
        V_ADP = A_ADP
        spiketrain.append(time)
    
sp = spikeplot.SpikePlot()
sp.set_markerscale(0.5)

sp.plot_spikes(spiketrain)

