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
    
def CreateMarkerContours(point_coords):
    '''Generates contours for each marker, using point coordinates as input'''
    
    contours = []
    
    for marker in point_coords:
        if len(marker) >= 3:  # ConvexHull requires at least 3 points
            hull = ConvexHull(marker)
            hull_vertices = marker[hull.vertices]
            contours.append(hull_vertices)
        
    return contours

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
	
def CountCellFoci(foci_skimage_props, dapi_skimage_props, col_name):
    '''counts how many foci are inside each cell'''
    
    # get center of mass of each foci
    foci_centers = [(x,y) for x,y in zip(foci_skimage_props['weighted_centroid-0'], 
                                         foci_skimage_props['weighted_centroid-1'])]
    # get conntour of each cell nuclei
    cell_contours = CreateMarkerContours(dapi_skimage_props['coords'])

    # list to save counts for each cell

    cells_with_foci_inside = []

    for ind, cell in enumerate(cell_contours, start = 1):
        is_foci_inside = sk.measure.points_in_poly(foci_centers, cell)
        foci_count = np.count_nonzero(is_foci_inside)
        cells_with_foci_inside.append([ind, foci_count])
    
    cells_with_foci_inside = pd.DataFrame(cells_with_foci_inside, columns=['label', col_name])

    return cells_with_foci_inside
	
def GroupCellCounts (foci_count, header):
    ''' group cell counts according to the following categories:
	. Empty: nuclei without red or green foci
	. XIST only: nuclei with one or more red foci & no green foci
	. Normal Cells: nuclei with one or more red foci & one green foci
	. Erosion Onset: nuclei with one or more red foci & two or more green foci
	. XACT only: nuclei with one green & no red
	. Eroded: nuclei with two or more greens & no red 
	'''
    counts = {
    'Empty nuclei': foci_count[(foci_count['red(#)'] == 0) & (foci_count['green(#)'] == 0)].shape[0],
    'Normal Cells': foci_count[(foci_count['red(#)'] >= 1) & (foci_count['green(#)'] == 1)].shape[0],
    'XIST only': foci_count[(foci_count['red(#)'] >= 1) & (foci_count['green(#)'] == 0)].shape[0],
    'XACT only': foci_count[(foci_count['red(#)'] == 0) & (foci_count['green(#)'] == 1)].shape[0],
    'Erosion Onset': foci_count[(foci_count['red(#)'] >= 1) & (foci_count['green(#)'] >= 2)].shape[0],
    'Eroded': foci_count[(foci_count['red(#)'] == 0) & (foci_count['green(#)'] >= 2)].shape[0]
    }
	
	# convert into a dataframe
    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=[header])
    
    return counts_df

def saveFig(name, folder = 'figures/'):
    '''quick function to save images in folder'''
    plt.savefig(folder + name, dpi = 300, bbox_inches = 'tight', transparent=False)