# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:59:29 2024

@author: TS2000-user
"""

#%% Import packages
from measurement_control.instruments import KVS30, M30XY
from measurement_control.instrument_setups import TS2000Setup
from measurement_control.instrument_setups.instrument_setup import ConnectionType
from measurement_control.instruments.keithley_4200a import Keithley4200A, MeasurementType, PMUFourChannelMeasurement
from measurement_control.communicators import TCPIPCommunicator

from measurement_control.instrument_setups.k4200_setup import MeasurementType, PMUFourChannelMeasurement
from measurement_control.instruments.thorlabs_kinesis_stage import KinesisStage
from measurement_control.files import TextFile
from measurement_control.instrument_setups.k4200_setup import IntegrationTime
from measurement_control import config

import matplotlib.pyplot as plt

import requests
import time
import os 
import numpy as np
import pandas as bearcats           # 熊猫
import sys
import copy
print(__file__)
sys.path.append(os.path.join(os.path.split(__file__)[0], 'Functions'))
from eval_functions import find_read_sections, filter_data, calc_R1_list, V_threshold_2
from waveform_functions import init_waveform_fast_sweep, init_waveform_switching_SetReset
from wafermap import mask_singleDev_LineArray

#%%Symmetry Check
"""
data= TextFile(r'\\iff200\transfer\Zhao\Data\PCM\Test_sweep.txt')
data_content = data.read()
V = data_content['v1'].to_numpy()
I = data_content['i2'].to_numpy()
t = abs(data_content['time'].to_numpy())

I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=11)
plt.plot(t_filt, I_filt)
"""


def SymmetryCheck(I_filt, V_filt, t_filt):
    
    t_int=t_filt[np.where(I_filt == max(I_filt))]-t_filt[np.where(I_filt==min(I_filt))]
    try:
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
    except:
        result = False
    return result
def SymmetryCheck(I_filt, V_filt, t_filt, I_thresh=1e-4):
    
    try:
        max_I, min_I = max(I_filt), min(I_filt)
        neg_beg, neg_end = V_filt[np.where(I_filt>I_thresh)[0][[-2,-1]]]
        pos_beg, pos_end = V_filt[np.where(I_filt<-I_thresh)[0][[-2,-1]]]
        
        comp_1 = abs((max_I-abs(min_I))/max_I)
        comp_2 = np.max([abs(neg_beg-neg_end), abs(pos_beg-pos_end)])
        comp_3 = abs((len(np.where(I_filt>1e-4)[0])-len(np.where(I_filt<-1e-4)[0]))/len(np.where(I_filt>1e-4)[0]))
        result = (comp_1<0.1) and (comp_2<0.2) and (comp_3<0.1)
    except:
        result = False
    return result


def rotate_sample(position1, position2, homeposition):
    """
    Please use two dots with a defined distance in the x dir and 0 diff in y dir

    Parameters
    ----------
    position : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    alpha_rot = np.arctan((position1-position2)[1]/(position1-position2)[0])
    print(alpha_rot*180 / np.pi)
    def calc_with_dist(position):
        if np.max(np.abs(position))>50: 
            position = position*1e-3
        x = homeposition[0]+position[0]*np.cos(alpha_rot)-position[1]*np.sin(alpha_rot)
        y = homeposition[1]+position[0]*np.sin(alpha_rot)+position[1]*np.cos(alpha_rot)
        return np.array([x, y])
    return calc_with_dist
#%% Base Settings
### Sample information 

sample_layout = "Jalil\LineArray\SingleCell" 
sample_material = 'AIST'
sample_name = 'NT_LPCM_line_array_AIXCCT_2'

config.operator = "Zhao"
config.sample = "PCM" + sample_material + '_' + sample_name

### id devices measured
# Load wafer info
positions, label, geometry = mask_singleDev_LineArray()


start_site =1472
stop_site  =2400
dist_meas = 10
blocks_meas = ['C1','D1','A2','C2','D2','A6','C6','D6']

DevicesTest = np.arange(start_site, stop_site, dist_meas)
DevicesTest = DevicesTest[[l[0][0] in blocks_meas for l in label[DevicesTest]]]
DevicesTest = DevicesTest[np.argmin(abs(DevicesTest-1482))-1:]

### Starting voltages


#Initial Sweep
Vpulse1_start=[1,0]
Vpulse2_start=[-1,0]

#Precycling
Vset_start=[1,0]
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


list_repetition = [1, 10, 100]


### Settings


#Measurement resolution
meas_range1=1e-3
meas_range2=1e-5


# Definition resistive states
R_min_HRS, R_max_LRS = 30e3, 5e3

# Measurement setup
bool_TS2000 = True # if True TS2000 if False MSAuto 



#%% Connect to Setup

##########################################
## Calibration
##########################################
position1, position2 = np.array([-13.694, 13.733]), np.array([7.625, 13.099])


home_pos = np.array([-12.3663, 13.2528])

calc_with_dist = rotate_sample(position1, position2, home_pos)
with KinesisStage(
    M30XY('101383394', load_position_x=0, load_position_y=15),
    KVS30('24402914', max_height=10, contact_height=10, separation_height=9.5, load_position=0)
) as stage:
    
    stage.set_velocity(4, 1)
    if False:
        stage.home()
    #stage.load()
    #stage.move_by(0,-0.61, 0)
    # stage.move_by(-0.03, 0, 0) #negativ=nach links
    # stage.move_by(0, -0.15, 0)
    # stage.move_by(0.07, -0.04, 0)
    # stage.move_by(0,1.83, 0)
    # stage.move_by(-0.03, 0, 0)
    # position=stage.get_position()
    # stage.move_to( 1.2497, 11.902, 9.5)

#%%    
data_provide = {}
with Keithley4200A(communicator=TCPIPCommunicator('169.254.26.11',1225,read_termination='\0',\
                                   write_termination='\0',),num_pmus=2,num_smus=4) as keithley:
    with KinesisStage(
        M30XY('101383394', load_position_x=0, load_position_y=15),
        KVS30('24402914', max_height=10, contact_height=10, separation_height=9.5, load_position=0)
    ) as stage:
    
        PathSample = os.path.join(r'C:\Transfer\Zhao\Data\PCM', sample_layout, sample_material, sample_name)
    
    
        for device_id in DevicesTest:
            print()
            
            device_broken=False
            pos_wfm =positions[device_id]
            pos_step = calc_with_dist(pos_wfm)
            stage.separation()
            stage.move_to(pos_step[0], pos_step[1], 9.5)
            stage.contact()

            
            device =  label[device_id][0][0]+'_'+label[device_id][1][0]
            
            # Save info device
            data_provide['id_device'] = str(device_id)
            data_provide['label_device'] = device
            data_provide['position_device_y'] = str(positions[device_id][1])
            data_provide['position_device_x'] = str(positions[device_id][0])

            print('Measuring device ' + device)
            PathDevice = os.path.join(PathSample, os.path.join(label[device_id][0][0], label[device_id][1][0]))
            os.makedirs(PathDevice, exist_ok=True)
            ##########################################
            ## Initial sweep 
            ##########################################
            # 1. Start: Vsweep = 1 V, n_rep = 1 
            # 2. While R > R_max_LRS and sweep not symmetric 
            # 3. Measure one sweep and if 2. False Vsweep += 0.2 V
            
            data_provide['action'] = 'Init sweep'

            n_cyc_presweep, delta_V_presweep = 1, 0.2
            Vpulse1, Vpulse2 = copy.deepcopy(Vpulse1_start), copy.deepcopy(Vpulse2_start)
            data_provide['cycle'] = '1'
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
            presweep = presweep_wf.run(file=TextFile(os.path.join(PathDevice,'IniSweep_'+str(np.argmax(np.abs(Vpulse1)))+'V_'+cycle_time+'.txt')),
                                   custom_metadata=data_provide)
            
        
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
            #plt.savefig(os.path.join(PathDevice,"Set_"+device+cycle_time+".png"))
            plt.show()
            
            R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
            Symm =SymmetryCheck(I_filt, V_filt, t_filt)
            
            while Symm==False or any(R_LRS>R_max_LRS) or n_cyc_presweep<100:
              # 5. If current not Symmetric and  R > R_max_LRS 
              # 6. GOTO 2 with Vsweep += 0.2V
                data_provide['cycle'] = str(n_cyc_presweep)
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
                presweep = presweep_wf.run(file=TextFile(os.path.join(PathDevice,'IniSweep_'+str(np.argmax(np.abs(Vpulse1)))+'V_'+cycle_time+'.txt')),
                                       custom_metadata=data_provide)

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
                plt.savefig(os.path.join(PathDevice,"IniSweep_"+device+cycle_time+".png"))
                plt.show()
                R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
                Symm=SymmetryCheck(I_filt, V_filt, t_filt)
                if all(R_LRS < R_max_LRS): 
                    n_cyc_presweep=1000
                else:
                    Vpulse1[np.argmax(np.abs(Vpulse1))] += np.sign(Vpulse1[np.argmax(np.abs(Vpulse1))])*delta_V_presweep
                    Vpulse2[np.argmax(np.abs(Vpulse2))] += np.sign(Vpulse2[np.argmax(np.abs(Vpulse2))])*delta_V_presweep
                if np.abs(max(Vpulse1))>5 or np.abs(max(Vpulse2))>5:
                    break
                if max(abs(I))<50e-6 and np.max(np.abs(Vpulse1))>3:
                    device_broken=True
                    break
            
            if np.abs(max(Vpulse1))>5 or np.abs(max(Vpulse2))>5 or device_broken:
                continue
            # 7. End Initial Sweep
                
                
             
        
            ##########################################
            ## Precycling reset
            ##########################################
            # 7. If R_HRS >R_min_HRS and R_LRS < R_max_LRS and n_switch < 1000 goto 6. n_switch += 500
            
            data_provide['action'] = 'Precycle'
            
            # 1. Start: Vreset = 1 V and Vset = 2 V, n_rep = 1, V_step = 0.5 V (Waveform sweep and pulse)
            Vset = copy.deepcopy(Vset_start)
            Vreset =  copy.deepcopy(Vreset_start)
            n_cyc_prepar_switch = 1
            delta_set_pre, delta_reset_pre = 0.1, 0.3
            # 2. While R_HRS < R_min_HRS and R_LRS > 10e3
            n_meas=0
            while (any(R_HRS<R_min_HRS) or any(R_LRS> R_max_LRS)) or n_meas<1000:
                # 3. Measure
                data_provide['cycle'] = str(n_cyc_prepar_switch)
                switch_wf = init_waveform_switching_SetReset(keithley, 
                                                  Vset=Vset,
                                                  Vreset=Vreset,
                                                  sweep_rate_1=1e6,
                                                  timewidthPulse2=100e-9,
                                                  timeLEPulse2=50e-9,
                                                  timeTEPulse2=20e-9,
                                                  timewidthRead=1e-2,
                                                  timeEdgesRead=1e-3,
                                                  timeWait=1e-5,
                                                  nr_cycle=n_cyc_prepar_switch,
                                                  meas_range1=1e-3,
                                                  meas_range2=1e-3,
                                                  bool_retention=False,
                                                  time_gap_lists=[1e-6, 1e-5,1e-4, 1e-3, 1e-2])
                cycle_time = time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime())
                switch = switch_wf.run(file=TextFile(os.path.join(PathDevice,'Precycle_'+str(np.argmax(np.abs(Vset)))+'V_Reset'+str(np.argmax(np.abs(Vreset)))+'V_'+cycle_time+'.txt')),
                                       custom_metadata=data_provide)
                V = switch['v1'].to_numpy()
                I = switch['i2'].to_numpy()
                t = abs(switch['time'].to_numpy())
                I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=3)
                plt.plot(V, abs(I))
                plt.yscale('log')
                plt.xlabel('Voltage / V')
                plt.ylabel('I  / $\mu$A')
                plt.yscale('log')
                plt.title(r'Prepare device n$_{cycle}$'+str(n_meas))
                plt.savefig(os.path.join(PathDevice,"Precycle_"+device+cycle_time+".png"))
                plt.show()
                R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
        
                if any(R_HRS<R_min_HRS): 
                    if len(R_HRS>=10):
                        if np.median(R_HRS[:5])>R_min_HRS and np.median(R_HRS[-5:])<R_min_HRS:
                            Vreset[np.argmax(np.abs(Vreset))] -= np.sign(Vreset[np.argmax(np.abs(Vreset))])*3*delta_V_presweep
                        else:
                            Vreset[np.argmax(np.abs(Vreset))] += np.sign(Vreset[np.argmax(np.abs(Vreset))])*delta_V_presweep

                    else:
                        Vreset[np.argmax(np.abs(Vreset))] += np.sign(Vreset[np.argmax(np.abs(Vreset))])*delta_V_presweep
                    n_cyc_prepar_switch = 1
                elif all(R_HRS>5*R_min_HRS) and not any(R_LRS>R_max_LRS):
                    Vreset[np.argmax(np.abs(Vreset))] -= np.sign(Vreset[np.argmax(np.abs(Vreset))])*delta_reset_pre 
                    delta_reset_pre = 0.1
                elif any(R_LRS>R_max_LRS):
                    Vset[np.argmax(np.abs(Vset))] += np.sign(Vset[np.argmax(np.abs(Vset))])*delta_set_pre
                    n_cyc_prepar_switch=1
                    
                if all(R_HRS>R_min_HRS) and all(R_LRS<R_max_LRS) : 
                    n_meas+=n_cyc_prepar_switch
                    i_nr = np.min([np.where(np.array(list_repetition)==n_cyc_prepar_switch)[0][0]+1,len(list_repetition)-1])
                    n_cyc_prepar_switch=list_repetition[i_nr]
                V_threshold, V_down, bool_threshold = V_threshold_2(V,I,I_thresh=-20e-6)
                if max(np.abs(V_threshold))>np.max(np.abs(Vset))-0.3:
                    n_cyc_prepar_switch = 1
                    Vreset[np.argmax(np.abs(Vreset))] -= np.sign(Vreset[np.argmax(np.abs(Vreset))])*delta_reset_pre *3
                    Vset[np.argmax(np.abs(Vset))] += np.sign(Vset[np.argmax(np.abs(Vset))])*delta_set_pre
                if (max(abs(I))<50e-6 and np.max(np.abs(Vset))>3) or np.max(np.abs(Vreset))<0.5:
                    device_broken=True
                    break
            if device_broken:
                continue
               
            ##########################################
            ## Varry reset
            ##########################################
            # 1. Start at Vreset = 1 V and Vset = 2 V n_switch = 100 one sweep one reset in every waveform
           
            data_provide['action'] = 'Var. reset'
            runs_bad = 0
            # 2. For Vreset in {1 + 0.3 * i| i in N, i<11}
            for  Vreset_volt in np.sort([a for a in {np.round(1 + 0.1 * i,2) for i in range(46)}]):
                Vreset[np.argmax(np.abs(Vreset))] = -Vreset_volt
                # 5. Measure
                n_meas, n_rep_var = 1, 20
                no_reset = 0
                seems_reset = 0
                device_broken =False
                runs_bad = 0
                while n_meas < 999:
                    data_provide['cycle'] = str(n_rep_var)
                    switch_wf = init_waveform_switching_SetReset(keithley, 
                                                      Vset=Vset,
                                                      Vreset=Vreset,
                                                      sweep_rate_1=1e6,
                                                      timewidthPulse2=100e-9,
                                                      timeLEPulse2=50e-9,
                                                      timeTEPulse2=20e-9,
                                                      timewidthRead=1e-2,
                                                      timeEdgesRead=1e-3,
                                                      timeWait=1e-5,
                                                      nr_cycle=n_rep_var,
                                                      meas_range1=1e-2,
                                                      meas_range2=1e-3,
                                                      bool_record_all = True, 
                                                      bool_retention=False,
                                                      time_gap_lists=[1e-6, 1e-5,1e-4, 1e-3, 1e-2])
                    cycle_time = time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime())
                    switch = switch_wf.run(file=TextFile(os.path.join(PathDevice,'Varry_Vreset_'+str(np.round(np.max(np.abs(Vset)),1))+'V_Reset'+str(np.round(np.max(np.abs(Vset)),1))+'V_'+cycle_time+'.txt')),
                                           custom_metadata=data_provide)
                    V = switch['v1'].to_numpy()
                    I = switch['i2'].to_numpy()
                    t = abs(switch['time'].to_numpy())
                    I_filt, V_filt, t_filt = filter_data(I, V, t, Nmean=5)
                    # 4. if device broken(I_max too low)
                    plt.plot(V_filt, abs(I_filt))
                    plt.yscale('log')
                    plt.xlabel('Voltage / V')
                    plt.ylabel('I  / $\mu$A')
                    plt.yscale('log')
                    plt.title(r'Var reset V$_{reset}$ = ' + str(np.round(Vreset[0],1)))
                    plt.savefig(os.path.join(PathDevice,"Varry_Vreset_"+device+cycle_time+".png"))
                    plt.show()
                    R_HRS, R_LRS = calc_R1_list(I, V,0.9, 0.1, V_max_small=0.9, V_min_small=0.2, I_res=8e-6)
                    n_meas += n_rep_var
                    
                    if max(abs(R_HRS))< 3*max(abs(R_LRS)):
                        no_reset += 1
                        if no_reset > 5:
                            #device_broken = True
                            break
                    elif max(abs(R_HRS)) < R_min_HRS:
                        seems_reset += 1
                        if seems_reset > 10:
                            #device_broken = True
                            break

                    if max(abs(I))<50e-6 and device_broken:
                        device_broken=True
                        break
                    if any(R_LRS>R_max_LRS):
                        Vset[np.argmax(np.abs(Vset))] += np.sign(Vset[np.argmax(np.abs(Vset))])*0.2
                        runs_bad += 1
                        if runs_bad>3:
                            device_broken=True
                            
                        print("Devices damaged at Vreset", Vreset_volt, "V")
                    else: 
                        runs_bad = 0
                        device_broken=False

                if device_broken:
                
                    print("Devices not useable for Vreset = ", Vreset_volt, " V")
                    break
            if device_broken:
                continue
#%%