import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
import argparse
import sys
import ast
from pathlib import Path
import torch

import skimage as ski


def get_surrounding(coord_array):
    coords = np.zeros((coord_array.shape[0]*8,2))
    for ind,coord in enumerate(coord_array):
        coords[ind*8:ind*8+8] = get_surrounding_for1(coord)
    uniques = np.unique(coords,axis=0).tolist()
    
    #Lastly remove any cases that are out of bounds
    final_uniques = np.array([i for i in uniques if -1<i[0]<704 and -1<i[1]<704 and i not in coord_array.tolist()])
    return final_uniques

def get_surrounding_for1(coord):
    x,y = coord
    surrounding = np.array([[x-1,y],[x-1,y+1],[x-1,y-1],[x+1,y],[x+1,y+1],[x+1,y-1],[x,y-1],[x,y+1]])
    return surrounding

def pixel_post_one(filename, classesPixels, comb_thr, numClasses):
    if isinstance(filename, torch.Tensor):
        orig_seg_pixel_array = filename
        if orig_seg_pixel_array.dim() == 2:  # If it has shape (H, W), add a new first dimension
            orig_seg_pixel_array = orig_seg_pixel_array.unsqueeze(0)
    elif isinstance(filename, np.ndarray):
        orig_seg_pixel_array = filename
        if orig_seg_pixel_array.ndim == 2:  # If it has shape (H, W), add a new first dimension
            orig_seg_pixel_array = np.expand_dims(orig_seg_pixel_array, axis=0)
    else:
        # Load the image if filename is not an array
        orig_seg = sitk.ReadImage(filename)
        orig_seg_pixel_array = sitk.GetArrayFromImage(orig_seg)
    if comb_thr == 1:
        orig_seg_pixel_array[orig_seg_pixel_array==9] = 10

    orig_seg_copy = np.copy(orig_seg_pixel_array)
    

    for frame in range(orig_seg_pixel_array.shape[0]): #So it will work for either full pullback or single frame
        for item in classesPixels.items(): 
            #Need to adapt to work with background
            class_ind = item[0]
            pixel_minimum = item[1][0]
            temp_array = np.copy(orig_seg_copy[frame]) # was initially temp_array = np.copy(orig_seg_pixel_array[frame]) 
            if class_ind == 0:
                temp_array[temp_array==0] = 20
                temp_array[temp_array<20] = 0
                temp_array[temp_array==20] = 1
            else:
                temp_array[temp_array!=class_ind] = 0
                temp_array[temp_array==class_ind] = 1
        
            labeled_image, count = ski.measure.label(temp_array, connectivity=item[1][1], return_num=True)
        
            unique, counts = np.unique(labeled_image, return_counts=True)
        
            for group_ind in range(1,count+1):#Loop over the groups
                if group_ind >= len(counts):
                    print(f"⚠️ Problem at Frame {frame}, class {class_ind}: group {group_ind} not in counts (len={len(counts)})")    
                    continue
                if counts[group_ind]>pixel_minimum:
                    continue
                else:
                    main_indices = np.where(labeled_image == group_ind)
                    usable_main_indices = np.array(list(zip(main_indices[0],main_indices[1])))
                    neighbour_indices = get_surrounding(usable_main_indices)
                
                    neighbour_indices = neighbour_indices.astype(int)
                
                    current_counts = np.zeros(numClasses)
                    #Loop over the neighbour indices and get counts of classes
                    for neighbour in neighbour_indices: 
                        current_counts[int(orig_seg_copy[frame,neighbour[0],neighbour[1]])] += 1
                
                    if (current_counts[0]/np.sum(current_counts)) > 0.75:
                        new_class = 0
                    else:
                        new_class = np.argmax(current_counts[1:])+1
                
                    #Will fail if only 1 index is found (is that really the case?)
                    for main in usable_main_indices:
                        orig_seg_copy[frame,main[0],main[1]] = new_class
                        
    return orig_seg_copy