# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:17:59 2024

@author: paulo
"""

import czifile
import numpy as np
import pandas as pd
import skimage as sk
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Load the .czi file and apply max projection to each channel

def ReadCZI(file):
    
    with czifile.CziFile(file) as czi:
        
        # Read the image data
        multi_channel_array = czi.asarray()
        multi_channel_array = np.max(multi_channel_array, axis = -1)
        
        # Perform a Z-projection for each channel (max intensity here)
        
        c1 = np.max(multi_channel_array, axis = 1)[0] # RED
        c2 = np.max(multi_channel_array, axis = 1)[1] # GREEN
        c3 = np.max(multi_channel_array, axis = 1)[2] # BLUE
    
        c1 = np.clip(c1 - np.mean(c3), 0, None)
        c2 = np.clip(c2 - np.mean(c3), 0, None)
        
        return c1, c2, c3
    
def DrawMarkerContours(point_coords, ax, color = 'red', lw = 1):
    '''Draw contour for each marker using point coordinates as input'''
    for marker in point_coords:
        hull = ConvexHull(marker)
        for contour in hull.simplices:
                ax.plot(marker[contour, 1], marker[contour, 0], '-', lw = lw, color = color, alpha = 0.5)
    
def AddLabelsToFigure(ax, regionprops_table, X = "weighted_centroid-1", Y = "weighted_centroid-0", s = 2, color = 'red'):
    '''Adds labels to an existing Matplotlib figure based on axis selection.
    s: size of text; color: text color'''
    
    table = regionprops_table

    for i in range(table.shape[0]):
        label = str(table.iloc[i]["label"].astype(int))
        x = table.iloc[i][X].astype(float)
        y = table.iloc[i][Y].astype(float)
        
        # Add text label to the plot
        ax.text(x, y, label, fontsize=s, color=color, weight='semibold', ha='center', va='center')
    
    return ax

def GetRegionProps(markers, im):

    markers = markers.astype('int8')
    
    if len(markers.shape) == 3: 
        markers = np.max(markers, axis =2)    
    
    props = sk.measure.regionprops_table(markers, im,
            properties = ['label', 'area', 'weighted_centroid', 'major_axis_length', 'minor_axis_length', 'mean_intensity', 'coords'])
    
    props = pd.DataFrame(props)

    # remove small markers
    props = props[props['area'] != 1].reset_index(drop=True)

    # redefine labels; start with ONE
    props.label = props.index + 1
    
    return props

def saveFig(name, folder = 'figures/'):
    plt.savefig(folder + name, dpi = 300, bbox_inches = 'tight', transparent=False)