# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:01:24 2024

@author: TS2000-user
"""
## Import common packages

##############################################################################
from measurement_control.instrument_setups import TS2000Setup
from measurement_control.instrument_setups.instrument_setup import ConnectionType
from measurement_control.instrument_setups.k4200_setup import MeasurementType, PMUTwoChannelMeasurement
from measurement_control.files import TextFile
from measurement_control.instrument_setups.k4200_setup import IntegrationTime

import matplotlib.pyplot as plt

import requests
import time
import os 
import numpy as np
import pandas as bearcats           # 熊猫
#%% PMU sweeps

def init_waveform_fast_sweep(keithley, 
                                  Vpulse1=[2,0],
                                  Vpulse2=[-2,0],
                                  sweep_rate_1=1e6,
                                  sweep_rate_2=1e6,
                                  timeWait=1e-6,
                                  nr_cycle=1,
                                  record_all=True,
                                  meas_range1=1e-3,
                                  meas_range2=1e-5):    
    '''
    

    Parameters
    ----------
    Vset : list, optional
        Gate and source voltage applied during forming. The default is [3,0].
    Vreset : list, optional
        Gate and source voltage applied during reset after forming.
        The default is [-2,0].
    Vread : list, optional
        Gate and source voltage applied during read pulse.
        The default is [0.2,0].
    sr : int, optional
        Sampling rate. The default is int(1e6).
    timePulseSet : float, optional
        Time max voltage of forming pulse applied. The default is 1e-5.
    timePulseReset : float, optional
        Time max voltage of reset pulse applied. The default is 1e-5.
    timeRead : float, optional
        Duration read pulse. The default is 1e-5.
    timeSlope : float, optional
        Flanc duration. (Becautions depending on other parameters) 
        The default is 1e-6.
    Vgate : list, optional
        Gate voltages applied during 
        [0] Forming pulse,
        [1] first read pulse
        [2] reset pulse
        [3] second read pulse (should be the same as [1]). 
        Needs to be checked, if we measure 1T1R devices instead of 1R devices.
    nr_cycle : int, optional
        Number of times, you want to repeat pulse sequence in waveform.
    nr_cycle : int, optional
        Number of times, you want to repeat the waveform.
        The default is [0,0,0,0].
    Sample : str, optional
        Name of current sample. The default is 'Sample'.
    Block : str, optional
        Name of current block. The default is 'Block'.
    Device : str, optional
        Name of current device. The default is 'Device'.
    Dir : str, optional
        Directory in which we want to store the data in. 
        Always change if you use this function
        The default is 'C:\transfer\Schnieders'.
    plot_meas : bool, optional
        Decide, if measurement is ploted. 
        The default is False.
   

    Returns
    -------
    None.

    '''
    
    # Define waveform object 
    Switching_Pulse = PMUTwoChannelMeasurement(keithley=keithley,
                                                i_range_ch1=meas_range1, 
                                                i_range_ch2=meas_range2)
    # Todo find out what type of object
    if record_all:
        measurement_type_donotrecord = MeasurementType.WAVEFORM_DISCRETE
        
    else:
        measurement_type_donotrecord = MeasurementType.DISABLED
    
        # Read
    Switching_Pulse.add_segment(timeWait, 
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=measurement_type_donotrecord)
    ##########################################################################
    ######################### Begin definition waveform ######################
    ##########################################################################
    timeflank1, timeflank2 = max(np.abs(Vpulse1))/sweep_rate_1, max(np.abs(Vpulse2))/sweep_rate_2
    for cycle in range(nr_cycle-1):
        Switching_Pulse.add_segment(timeflank1, 
                                     voltage_ch1=Vpulse1[0],
                                     voltage_ch2=Vpulse1[1],
                            meas_type=measurement_type_donotrecord)
        Switching_Pulse.add_segment(timeflank1, 
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=measurement_type_donotrecord)
        
        Switching_Pulse.add_segment(timeflank2,  
                                    voltage_ch1=Vpulse2[0],
                                    voltage_ch2=Vpulse2[1],
                            meas_type=measurement_type_donotrecord)
        Switching_Pulse.add_segment(timeflank2,  
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=measurement_type_donotrecord)

    Switching_Pulse.add_segment(timeflank1, 
                                 voltage_ch1=Vpulse1[0],
                                 voltage_ch2=Vpulse1[1],
                        meas_type=MeasurementType.WAVEFORM_DISCRETE)
    Switching_Pulse.add_segment(timeflank1, 
                                 voltage_ch1=0,
                                 voltage_ch2=0,
                        meas_type=MeasurementType.WAVEFORM_DISCRETE)
    
    Switching_Pulse.add_segment(timeflank2,  
                                voltage_ch1=Vpulse2[0],
                                voltage_ch2=Vpulse2[1],
                        meas_type=MeasurementType.WAVEFORM_DISCRETE)
    Switching_Pulse.add_segment(timeflank2,  
                                 voltage_ch1=0,
                                 voltage_ch2=0,
                        meas_type=MeasurementType.WAVEFORM_DISCRETE)
    Switching_Pulse.add_segment(timeWait, 
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=measurement_type_donotrecord)    
    ##########################################################################
    ######################### End definition waveform ########################
    ##########################################################################
    Switching_Pulse.samplerate = np.min([float(2000/sum(Switching_Pulse.ch1.durations)),
                                        float(2e8)])
    
    # Return the waveform like it was defined..
    return Switching_Pulse

def init_waveform_switching_SetReset(keithley, 
                                  Vset=[2,0],
                                  Vreset=[-3,0],
                                  Vread =[-0.4,0],
                                  sweep_rate_1=1e6,
                                  timewidthPulse2=30e-9,
                                  timeLEPulse2=10e-9,
                                  timeTEPulse2=5e-9,
                                  timewidthRead=1e-5,
                                  timeEdgesRead=1e-5,
                                  timeWait=1e-5,
                                  nr_cycle=1,
                                  meas_range1=1e-3,
                                  meas_range2=1e-5,
                                  bool_retention=False,
                                  time_gap_lists=[1e-6, 1e-5,1e-4, 1e-3, 1e-2]):    
    '''
    

    Parameters
    ----------
    Vset : list, optional
        Gate and source voltage applied during forming. The default is [3,0].
    Vreset : list, optional
        Gate and source voltage applied during reset after forming.
        The default is [-2,0].
    Vread : list, optional
        Gate and source voltage applied during read pulse.
        The default is [0.2,0].
    sr : int, optional
        Sampling rate. The default is int(1e6).
    timePulseSet : float, optional
        Time max voltage of forming pulse applied. The default is 1e-5.
    timePulseReset : float, optional
        Time max voltage of reset pulse applied. The default is 1e-5.
    timeRead : float, optional
        Duration read pulse. The default is 1e-5.
    timeSlope : float, optional
        Flanc duration. (Becautions depending on other parameters) 
        The default is 1e-6.
    Vgate : list, optional
        Gate voltages applied during 
        [0] Forming pulse,
        [1] first read pulse
        [2] reset pulse
        [3] second read pulse (should be the same as [1]). 
        Needs to be checked, if we measure 1T1R devices instead of 1R devices.
    nr_cycle : int, optional
        Number of times, you want to repeat pulse sequence in waveform.
    nr_cycle : int, optional
        Number of times, you want to repeat the waveform.
        The default is [0,0,0,0].
    Sample : str, optional
        Name of current sample. The default is 'Sample'.
    Block : str, optional
        Name of current block. The default is 'Block'.
    Device : str, optional
        Name of current device. The default is 'Device'.
    Dir : str, optional
        Directory in which we want to store the data in. 
        Always change if you use this function
        The default is 'C:\transfer\Schnieders'.
    plot_meas : bool, optional
        Decide, if measurement is ploted. 
        The default is False.
   

    Returns
    -------
    None.

    '''
    
    # Define waveform object 
    Switching_Pulse = PMUTwoChannelMeasurement(keithley=keithley,
                                                i_range_ch1=meas_range1, 
                                                i_range_ch2=meas_range2)
    
    record_type, no_record_type = MeasurementType.WAVEFORM_DISCRETE, MeasurementType.DISABLED
    
    timeflank1 = max(np.abs(Vset))/sweep_rate_1

    ##########################################################################
    ######################### Begin definition waveform ######################
    ##########################################################################
    for cycle in range(nr_cycle):
        # Set pulse
        
        # Todo find out what type of object

        # Set
        Switching_Pulse.add_segment(timeflank1, 
                                     voltage_ch1=Vset[0],
                                     voltage_ch2=Vset[1],
                            meas_type=record_type)
        Switching_Pulse.add_segment(timeflank1, 
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=record_type)
        

        # Reset
        Switching_Pulse.add_segment(timeWait,
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=no_record_type)
        Switching_Pulse.add_segment(timeLEPulse2, 
                                     voltage_ch1=Vreset[0],
                                     voltage_ch2=Vreset[1],
                            meas_type=record_type)
        Switching_Pulse.add_segment(timewidthPulse2,
                                     voltage_ch1=Vreset[0],
                                     voltage_ch2=Vreset[1],
                            meas_type=record_type)
        Switching_Pulse.add_segment(timeTEPulse2, 
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=no_record_type)
        Switching_Pulse.add_segment(timewidthPulse2,
                                     voltage_ch1=0,
                                     voltage_ch2=0,
                            meas_type=record_type)
        
        if bool_retention:
            current_time = 0 
            for time_gap in time_gap_lists:
                # retention_reads
                if time_gap-current_time > 5e-9:
                    Switching_Pulse.add_segment(time_gap-current_time, 
                                                 voltage_ch1=0,
                                                 voltage_ch2=0,
                                        meas_type=no_record_type)
            
                Switching_Pulse.add_segment(timeflank1, 
                                             voltage_ch1=Vread[0],
                                             voltage_ch2=Vread[1],
                                    meas_type=record_type)
                Switching_Pulse.add_segment(timeflank1, 
                                             voltage_ch1=0,
                                             voltage_ch2=0,
                                    meas_type=record_type)
                current_time =  time_gap + 2 * timeflank1
    ##########################################################################
    ######################### End definition waveform ########################
    ##########################################################################
    
    Switching_Pulse.samplerate = np.min([float(2000/sum(Switching_Pulse.ch1.durations)),
                                        float(2e8)])
    
    # Return the waveform like it was defined..
    return Switching_Pulse
