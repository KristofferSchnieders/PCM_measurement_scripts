# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:30:11 2024

@author: zhao
"""


import numpy as np

def mask_singleDev_LineArray():
    # Distances
    dist_blocks = np.array([2920, -2080])
    dist_devices = np.array([320, -260])
    
    # Number structures
    nr_blocks = (5, 10)
    nr_devices = (7, 8)
    
    # Labels
    Row_label = [
                 ["A", "B", "C", "D", "E", "F", "H"],      # Labels Blocks 
                 ["a", "b", "c", "d", "e", "f", "g", "h","i","j"]  # Labels devices
                 ]
    
    # Dimensions of device 
    # Format [100, 200] means Length 200 nm x Width 100 nm. 
    geometries_blocks = np.array([[[200, 200], [200, 100],
                                   [200, 70], [200, 50], 
                                   [200, 30]], # first row
                                  [[180, 200], [180, 100],
                                   [180, 70], [180, 50], 
                                   [180, 30]], # second row
                                  [[150, 200], [150, 100],
                                   [150, 70], [150, 50], 
                                   [150, 30]], # third row
                                  [[120, 200], [120, 100],
                                   [120, 70], [120, 50], 
                                   [120, 30]], # forth row
                                  [[100, 200], [100, 100],
                                   [100, 70], [100, 50], 
                                   [100, 30]], # fifth row
                                  [[90, 200], [90, 100],
                                   [90, 70], [90, 50], 
                                   [90, 30]], # sixth row
                                  [[80, 200], [80, 100],
                                   [80, 70], [80, 50], 
                                   [80, 30]], # seventh row
                                  [[70, 200], [70, 100],
                                   [70, 70], [70, 50], 
                                   [70, 30]], # etighth row
                                  [[50, 200], [50, 100],
                                   [50, 70], [50, 50], 
                                   [50, 30]], # nineth row
                                  [[30, 200], [30, 100],
                                   [30, 70], [30, 50], 
                                   [30, 30]], # tenth row
                        ])
    
    positions = list()
    geometry  = list()
    label     = list()
    
    # Loop making list of devices
    for block in np.ndindex(nr_blocks):
        
        block_position = block*dist_blocks
        block = block[::-1]
        for device in np.ndindex(nr_devices):
            
            device = device[::-1]
    
            # These devices are missing to have space for the lithography markers
            if device == (0,0) or device == (1,0) or device == (0,1):
                continue
            if device == (7,6):
                continue
            if device == (0,6):
                continue
            if device == (7,0) or device == (6,0) or device == (7,1):
                continue
            
            # calculate position
            positions.append(block_position + dist_devices*(device-np.array([2,0])))
            # store geometry
            geometry.append(geometries_blocks[block])
            # store_label
            device = device[::-1]
            label.append([
                          [Row_label[0][block[1]] + str(block[0]+1)],
                          [Row_label[1][device[1]] + str(device[0]+1)]
                           ])
    
           
    positions = np.array(positions)
    geometry = np.array(geometry)
    label = np.array(label)
    return positions, label, geometry       