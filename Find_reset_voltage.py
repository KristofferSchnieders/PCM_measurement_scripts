# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:59:29 2024

@author: TS2000-user
"""

#%% Import packages

from measurement_control.instrument_setups import TS2000Setup
from measurement_control.instrument_setups.instrument_setup import ConnectionType
from measurement_control.instrument_setups.k4200_setup import MeasurementType, PMUFourChannelMeasurement
from measurement_control.files import TextFile
from measurement_control.instrument_setups.k4200_setup import IntegrationTime

import matplotlib.pyplot as plt

import requests
import time
import os 
import numpy as np
import pandas as bearcats           # 熊猫
import sys
print(__file__)
sys.path.append(os.path.join(os.path.split(__file__)[0], 'Functions'))
from eval_functions import *
from waveform_functions import *
from wafermap import * 

#%%Symmetry Check
data= TextFile(r'\\iff200\transfer\Zhao\Data\PCM\Test_sweep.txt')
data_content = data.read()
V = data_content['v1'].to_numpy()
I = data_content['i2'].to_numpy()
t = abs(data_content['time'].to_numpy())

I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=11)
plt.plot(t_filt, I_filt)



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
#%% Base Settings
### Sample information 

sample_layout = "Jalil\LineArray\SingleCell" # Mohit
sample_material = 'GST_023'
sample_name = 'NT_LPCM_line_array_AIXCCT_2'


### id devices measured


start_site = 0
stop_site  =10
dist_meas = 5
DevicesTest = np.arange(start_site, stop_site, dist_meas)


### Starting voltages


#Initial Sweep
Vpulse1_start=[1,0]
Vpulse2_start=[-1,0]

#Precycling
Vset_start=[2,0]
Vreset_start=[-1,0]
Vread_start=[0.4,0]


### Pulse timing


#Precycling
sweep_rate_1=1e6
sweep_rate_2=1e6

#Switching
sweep_rate_set_start=1e6
timewidthPulse2_start=30e-9
timeLEPulse2_start=10e-9
timeTEPulse2_start=5e-9
timewidthRead_start=1e-5
timeEdgesRead_start=1e-5
timeWait_start=1e-5


### Repetition numbers


list_repetition = [1, 10, 100, 100]


### Settings


#Measurement resolution
meas_range1=1e-3
meas_range2=1e-5


# Definition resistive states
R_min_HRS, R_max_LRS = 1e5, 5e3

# Measurement setup
bool_TS2000 = True # if True TS2000 if False MSAuto 

# Load wafer info
positions, label, geometry = mask_singleDev_LineArray()
#%% Connect to Setup
with TS2000Setup(keithley_connection=ConnectionType.TCPIP) as setup:
    keithley = setup.k4200
    if bool_TS2000:
        ts2000 = setup.ts2000
    else: 
        raise Exception('define Setup')
        ts2000 = setup.?
    PathSample = os.path.join(r'F:\transfer\Zhao\Data\PCM', sample_layout, sample_material, sample_name)


    for device in DevicesTest:
        PathDevice = os.path.join(PathSample, .....)
        ##########################################
        ## Initial sweep 
        ##########################################
        # 1. Start: Vsweep = 1 V, n_rep = 1 
        # 2. While R > R_max_LRS and sweep not symmetric 
        # 3. Measure one sweep and if 2. False Vsweep += 0.2 V
        n_cyc_presweep, delta_V_presweep = 1, 0.2
        Vpulse1, Vpulse2 = Vpulse1_start, Vpulse2_start 
        presweep_wf = init_waveform_fast_sweep(keithley, 
                                          Vpulse1=Vpulse1,
                                          Vpulse2=Vpulse2,
                                          sweep_rate_1=sweep_rate_1,
                                          sweep_rate_2=sweep_rate_2,
                                          nr_cycle=1,
                                          record_all=True,
                                          meas_range1=1e-3,
                                          meas_range2=1e-3)
        
        cycle_time = time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime())
        presweep = presweep_wf.run(file=TextFile(os.path.join(PathDevice,'Test_sweep.txt')))
        
    
        V = presweep['v1'].to_numpy()
        I = presweep['i2'].to_numpy()
        t = abs(presweep['time'].to_numpy())
        I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=11)
        plt.plot(V_filt, abs(I_filt))
        plt.yscale('log')
        plt.xlabel('Voltage / V')
        plt.ylabel('I  / $\mu$A')
        plt.yscale('log')
        plt.title('First sweep')
        plt.savefig(os.path.join(PathDevice,"Set_"+device+cycle_time+".png"))
        plt.show()
        R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
        Symm =SymmetryCheck(I_filt, V_filt, t_filt)
        while Symm==False or any(R_LRS>R_max_LRS):
          # 5. If current not Symmetric and  R > R_max_LRS 
          # 6. GOTO 2 with Vsweep += 0.2V

            presweep_wf = init_waveform_fast_sweep(keithley, 
                                              Vpulse1=Vpulse1,
                                              Vpulse2=Vpulse2,
                                              sweep_rate_1=sweep_rate_1,
                                              sweep_rate_2=sweep_rate_2,
                                              nr_cycle=n_cyc_presweep,
                                              record_all=True,
                                              meas_range1=1e-3,
                                              meas_range2=1e-3)
            
            cycle_time = time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime())
            presweep = presweep_wf.run(file=TextFile(os.path.join(PathDevice,'Test_sweep.txt')))
            
        
            V = presweep['v1'].to_numpy()
            I = presweep['i2'].to_numpy()
            t = abs(presweep['time'].to_numpy())
            I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=11)
            plt.plot(V_filt, abs(I_filt))
            plt.yscale('log')
            plt.xlabel('Voltage / V')
            plt.ylabel('I  / $\mu$A')
            plt.yscale('log')
            plt.title('First sweep')
            plt.savefig(os.path.join(PathDevice,"Set_"+device+cycle_time+".png"))
            plt.show()
            R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
            Symm=SymmetryCheck(I_filt, V_filt, t_filt)
            if R_LRS < R_max_LRS: 
                delta_V_presweep=0.1
                n_cyc_presweep=1000
            Vpulse1[np.argmax(np.abs(Vpulse1))] += np.sign(Vpulse1[np.argmax(np.abs(Vpulse1))])*delta_V_presweep
            Vpulse2[np.argmax(np.abs(Vpulse2))] += np.sign(Vpulse2[np.argmax(np.abs(Vpulse2))])*delta_V_presweep
        # 7. End Initial Sweep
        
        
     

    ##########################################
    ## Precycling reset
    ##########################################
    # 7. If R_HRS >R_min_HRS and R_LRS < R_max_LRS and n_switch < 1000 goto 6. n_switch += 500
    
    
    # 1. Start: Vreset = 1 V and Vset = 2 V, n_rep = 1, V_step = 0.5 V (Waveform sweep and pulse)
    Vset = Vset_start
    Vreset =  Vreset_start
    n_cyc_prepar_switch = 1
    delta_set_pre, delta_reset_pre = 0.1, -0.3
    # 2. While R_HRS < R_min_HRS and R_LRS > 10e3
    while any(R_HRS<R_min_HRS) or any(R_LRS> R_max_LRS) or n_cyc_prepar_switch<500:
        # 3. Measure
        switch_wf = init_waveform_switching_SetReset(keithley, 
                                          Vset=Vset,
                                          Vreset=Vreset,
                                          sweep_rate_1=1e6,
                                          timewidthPulse2=200e-9,
                                          timeLEPulse2=20e-9,
                                          timeTEPulse2=20e-9,
                                          timewidthRead=1e-2,
                                          timeEdgesRead=1e-3,
                                          timeWait=1e-5,
                                          nr_cycle=n_cyc_prepar_switch,
                                          meas_range1=1e-2,
                                          meas_range2=1e-3,
                                          bool_retention=False,
                                          time_gap_lists=[1e-6, 1e-5,1e-4, 1e-3, 1e-2])
        switch = switch_wf.run(file=TextFile(os.path.join(PathDevice,'Test_switch.txt')))
        V = switch['v1'].to_numpy()
        I = switch['i2'].to_numpy()
        t = abs(switch['time'].to_numpy())
        I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=5)
        plt.plot(V_filt, abs(I_filt))
        plt.yscale('log')
        plt.xlabel('Voltage / V')
        plt.ylabel('I  / $\mu$A')
        plt.yscale('log')
        plt.title('First sweep')
        plt.savefig(os.path.join(PathDevice,"Set_"+device+cycle_time+".png"))
        plt.show()
        R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)

        if any(R_HRS<R_min_HRS): 
            Vreset +=  delta_reset_pre
        elif all(R_HRS>5*R_min_HRS) and not any(R_LRS>R_max_LRS):
            Vreset += 0.1 
            delta_reset_pre = -0.1
        elif any(R_LRS>R_max_LRS):
            V_set += 0.2
        if all(R_HRS>R_min_HRS) and all(R_LRS<R_max_LRS) : 
            n_cyc_prepar_switch=500

    ##########################################
    ## Varry reset
    ##########################################
    # 1. Start at Vreset = 1 V and Vset = 2 V n_switch = 100 one sweep one reset in every waveform
    """
    #Vset=Vset_start
    Vreset=Vreset_start
    switch_wf = init_waveform_switching_SetReset(keithley, 
                                      Vset=Vset,
                                      Vreset=Vreset,
                                      sweep_rate_1=1e6,
                                      timewidthPulse2=200e-9,
                                      timeLEPulse2=20e-9,
                                      timeTEPulse2=20e-9,
                                      timewidthRead=1e-2,
                                      timeEdgesRead=1e-3,
                                      timeWait=1e-5,
                                      nr_cycle=1000,
                                      meas_range1=1e-2,
                                      meas_range2=1e-3,
                                      bool_retention=False,
                                      time_gap_lists=[1e-6, 1e-5,1e-4, 1e-3, 1e-2])
    switch = switch_wf.run(file=TextFile(os.path.join(PathDevice,'Test_switch.txt')))
    V = switch['v1'].to_numpy()
    I = switch['i2'].to_numpy()
    t = abs(switch['time'].to_numpy())
    I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=5)
    plt.plot(V_filt, abs(I_filt))
    plt.yscale('log')
    plt.xlabel('Voltage / V')
    plt.ylabel('I  / $\mu$A')
    plt.yscale('log')
    plt.title('First sweep')
    plt.savefig(os.path.join(PathDevice,"Set_"+device+cycle_time+".png"))
    plt.show()
    R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
    """
    # 2. For Vreset in {1 + 0.3 * i| i in N, i<11}
    for  Vreset in {np.round(1 + 0.3 * i,2) for i in range(11)}:
        # 5. Measure
        n_meas, n_rep_var = 0, 50
        while n_meas < 999:
            switch_wf = init_waveform_switching_SetReset(keithley, 
                                              Vset=Vset,
                                              Vreset=Vreset,
                                              sweep_rate_1=1e6,
                                              timewidthPulse2=200e-9,
                                              timeLEPulse2=20e-9,
                                              timeTEPulse2=20e-9,
                                              timewidthRead=1e-2,
                                              timeEdgesRead=1e-3,
                                              timeWait=1e-5,
                                              nr_cycle=n_rep_var,
                                              meas_range1=1e-2,
                                              meas_range2=1e-3,
                                              bool_retention=False,
                                              time_gap_lists=[1e-6, 1e-5,1e-4, 1e-3, 1e-2])
            switch = switch_wf.run(file=TextFile(os.path.join(PathDevice,'Test_switch.txt')))
            V = switch['v1'].to_numpy()
            I = switch['i2'].to_numpy()
            t = abs(switch['time'].to_numpy())
            I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=5)
            # 4. if device broken(I_max too low)
            if max(I_filt)< 25e-5:
                print("Devices damaged at Vreset", Vrs, "V")
                # 5. Break
                break
            plt.plot(V_filt, abs(I_filt))
            plt.yscale('log')
            plt.xlabel('Voltage / V')
            plt.ylabel('I  / $\mu$A')
            plt.yscale('log')
            plt.title('First sweep')
            plt.savefig(os.path.join(PathDevice,"Set_"+device+cycle_time+".png"))
            plt.show()
            R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
            n_meas += n_rep_var
    # 6. End