# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:46:55 2024

@author: TS2000-user
"""
import os
import numpy as np
import glob
import sys
import pandas as pd
import scipy
from datetime import datetime 
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.stats import linregress


from tqdm import tqdm
import collections

#%% Filter data 
def filter_data(I, V, t, Nmean=5):
    
    # Running median along the rows
    I_filt0 = scipy.signal.medfilt(I, kernel_size=Nmean)
    V_filt0 = scipy.signal.medfilt(V, kernel_size=Nmean)
    
    # Cumulative sum along the rows
    I_filt = (np.cumsum(I_filt0[Nmean:]) - np.cumsum(I_filt0[:-Nmean])) / Nmean
    V_filt = (np.cumsum(V_filt0[Nmean:]) - np.cumsum(V_filt0[:-Nmean])) / Nmean
    
    
    # Adjust t accordingly
    t_filt = t[int(Nmean/2)+1:-int(Nmean/2)]
    
    return I_filt, V_filt, t_filt

#%% Find read section
def find_read_sections(read_indices, n = 20, min_length = 50):
    read_indices = np.array(read_indices)
    diff_indices = read_indices[1:]-read_indices[:-1]
    gaps = np.where(diff_indices>n)[0]
    indices_read = np.array_split(read_indices,gaps+1)
    indices_read = [read for read in indices_read if len(read) > min_length]
    
    return indices_read

#%% Calc R
def calc_R1_list(I, V,V_max, V_min, V_max_small=0.9, V_min_small=0.2, I_res=8e-6):
    """
    \For 1 d vectors
    
    Parameters
    ----------
    I_filt : TYPE
        DESCRIPTION.
    V_filt : TYPE
        DESCRIPTION.
    V_max : TYPE
        DESCRIPTION.
    V_min : TYPE
        DESCRIPTION.
    
    Returns
    -------
    TYPE
        DESCRIPTION.
    
    """
    R_value = []
    
    for i_read, indices in enumerate(find_read_sections(np.where(np.logical_and(abs(V)<V_max_small,
                                                              abs(V)>V_min_small))[0],n=10,min_length=50)):
    
        R=0
        # indices = indices[I_filt[i][indices]<50e-6]
        x_all, y_all = abs(V[indices]), I[indices]
        if min(x_all)<0.25:

            if sum(y_all > I_res)<len(y_all)*0.5: 
                
                for V_max in np.linspace(0.9,0.5, 7):
                    try:
                        mask = find_read_sections(np.where(np.logical_and(x_all<V_max,x_all>V_min))[0],n=10,min_length=50)[0]
                    except:
                        break
                    x, y = x_all[mask],y_all[mask]
                    if linregress(x,y).pvalue>1e-7:
                        R = abs(1/np.polyfit(x,y,1)[0])
                        break
            else:
                pass

            if R ==0:
                
                V_max, V_min = 0.9, 0.15
                mask = find_read_sections(np.where(np.logical_and(x_all<V_max,x_all>V_min))[0],n=10,min_length=50)[0]
                x, y =x_all[mask], y_all[mask]
                if min(y)>50e-6:
                    R = abs(1/np.polyfit(x,y,1)[0])
                else:
                    V_max, V_min = 0.4, 0.1
                    mask = find_read_sections(np.where(np.logical_and(x_all<V_max,x_all>V_min))[0],n=10,min_length=50)[0]
                    x, y =x_all[mask], y_all[mask]
                
                    R = abs(1/np.polyfit(x,y,1)[0])
            if R==0:
                raise Exception('Something wrong')
            R_value.append(R)         
            
    return np.array(R_value[::2]), np.array(R_value[1::2])
#%% Search Threshold voltage

def V_threshold_T(V,I):
    V_threshold=[]
    bool_threshold=[]
    circle = []
    for i, Icyc in enumerate(I):
        sweep_segments = find_read_sections(np.where(V[i]>0.15)[0])
        for segments in sweep_segments:
            Isweep =  Icyc[segments]
            index_jump = np.where(np.logical_and(np.diff(Isweep,n=3)>3e-6, Isweep[3:]>50e-6))[0]
            if len(index_jump)==0 or np.min(Isweep)>5e-6:
                bool_threshold.append(False)
                V_threshold.append(-1)
            else: 
                bool_threshold.append(True)
                V_threshold.append(V_filt[i][segments][int(np.argmax(np.diff(Isweep)))])
            circle.append(i)
            
    return V_threshold, bool_threshold, circle

def V_threshold_2(V,I, I_thresh=20e-6):
    V_threshold=[]
    V_down = []
    bool_threshold=[]
    circle=[]
    for i, Icyc in enumerate(I):
        sweep_segments = find_read_sections(np.where(V[i]>0.15)[0])
        for segments in sweep_segments:
            Isweep =  Icyc[segments]
            index_jump = np.where(Isweep>I_thresh)
            try:
                V_start, V_end = V[i][segments][index_jump[0][[1,-1]]]    
                bool_threshold.append(True)
                V_threshold.append(V_start)
                V_down.append(V_end)
            
            except:
                bool_threshold.append(False)
                V_threshold.append(-1)
                V_down.append(-1)
                
            circle.append(i)      
    return V_threshold, V_down, bool_threshold, circle

def V_threshold_C(V,I, I_thresh=50e-6):
    V_threshold=[]
    V_down = []
    bool_threshold=[]
    circle=[]
    for i, Icyc in enumerate(I):
        sweep_segments = find_read_sections(np.where(V[i]>0.15)[0])
        for segments in sweep_segments:
            Isweep =  Icyc[segments]
            index_jump = np.where(Isweep>I_thresh)
            try:
                V_start, V_end = V[i][segments][index_jump[0][[1,-1]]]
                if V_start > (2*V_end) :
                    
                    bool_threshold.append(True)
                    V_threshold.append(V_start)
                    V_down.append(V_end)
                else:
                    bool_threshold.append(False)
                    V_threshold.append(-1)
                    V_down.append(-1)
                    
            except:
                bool_threshold.append(False)
                V_threshold.append(-1)
                V_down.append(-1)
                
            circle.append(i)      
    return V_threshold, V_down, bool_threshold, circle
def SymmetryCheck(I_filt, V_filt, t_filt):
    
    t_int=t_filt[np.where(I_filt == max(I_filt))]-t_filt[np.where(I_filt==min(I_filt))]
    comp_segments = find_read_sections(np.where(abs(I_filt)>0.5e-4)[0])
    comp_r1 = (V_filt[comp_segments[0][10]]-V_filt[comp_segments[0][0]])/(I_filt[comp_segments[0][10]]-I_filt[comp_segments[0][0]])
    comp_r2 = (V_filt[comp_segments[1][-1]]-V_filt[comp_segments[1][-11]])/(I_filt[comp_segments[1][-1]]-I_filt[comp_segments[1][-11]])
    comp1 = abs(len(comp_segments[0])-len(comp_segments[1]))/len(comp_segments[0])
    comp2 = abs(abs(max(I_filt))-abs(min(I_filt)))
    comp3 = abs(comp_r1 - comp_r2) / comp_r1
    
    if comp1<0.1 and comp2<5e-6 and comp3<0.1:
        print('Symmetry of Current Curve is good')
    else:
        print('Not good symmetry yet')
    result=comp1<10 and comp2<5e-6 and comp3<0.1
    return result
