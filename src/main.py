# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:49:42 2024

@author: paulo
"""

# basics
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import skimage as sk
import cv2 as cv

# my packages
sys.path.append(os.path.abspath('src'))

import utils, segmentation


def NuclearSegmentation(czi_file, sigma = 51, foci_offset = -2, cell_dist = 40, 
                        clear_border = True, output_folder = 'output'):
    
    im_r, im_g, im_dapi = utils.ReadCZI(czi_file)
    
    # apply segmentation column and play with the filters
    im_g_seg = segmentation.FociSegmentation(im_g,  sigma = sigma, offset = foci_offset, min_size = 100, max_size=500)
    im_r_seg = segmentation.FociSegmentation(im_r,  sigma = sigma, offset = foci_offset, min_size = 100, max_size=500)
    
    # apply threshold and watershed segmentation to nuclei
    im_markers = segmentation.NucleiSegmentation(im_dapi, sigma = sigma, cell_dist = cell_dist, 
                                                 clear_border = clear_border, small_objects = 10000)
    
    
    # GET MARKER PROPERTIES
    
    # get properties of each foci
    n_green_markers, green_markers = cv.connectedComponents(np.uint8(im_g_seg))
    n_red_markers, red_markers = cv.connectedComponents(np.uint8(im_r_seg))
    
    green_props = utils.GetRegionProps(green_markers, im_g_seg)
    red_props = utils.GetRegionProps(red_markers, im_r_seg)
    
    # get properties for dapi
    dapi_props = utils.GetRegionProps(im_markers, im_dapi)
    
    # visualize segmentation result
    # figure props
    fig, axis = plt.subplots(2, 3, figsize = (6,4), dpi = 180, constrained_layout=True)
    font = 6; 
    
    # turn all axis off; define axis name
    for ax in axis.flatten():ax.axis('off')
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axis
    
    # first row - plot raw images after max intensity projection
    ax1.imshow(im_dapi, plt.cm.gray); ax1.set_title('DAPI Raw (Max Proj)', fontsize = font)
    ax2.imshow(im_g, plt.cm.gray); ax2.set_title('FISH-Green Raw (Max Proj)', fontsize = font)
    ax3.imshow(im_r, plt.cm.gray); ax3.set_title('FISH-Red Raw (Max Proj)', fontsize = font)
    
    # show dapi segmentation result - watershed labels with numbers
    ax4.imshow(sk.color.label2rgb(im_markers)); ax4.set_title('DAPI-Watershed-Markers', fontsize = font)
    utils.AddLabelsToFigure(ax4, dapi_props, s = 4, color = 'k');
    
    # show green and red foci segmentation result (on top of dapi for better visualization)
    ax5.imshow(im_dapi, plt.cm.gray, alpha = 0.8); ax5.set_title('Green/Red Segmented', fontsize = font)
    ax5.plot(green_props['weighted_centroid-1'], green_props['weighted_centroid-0'], 'o', color = 'g', markersize = 0.3)
    ax5.plot(red_props['weighted_centroid-1'], red_props['weighted_centroid-0'], 'o', color = 'r', markersize = 0.3);
    
    # show all segmentation results overlapped
    ax6.imshow(im_dapi, plt.cm.gray, alpha = 0.8); ax6.set_title('All Segmented Combined', fontsize = font)
    ax6.plot(green_props['weighted_centroid-1'], green_props['weighted_centroid-0'], 'o', color = 'g', markersize = 0.3)
    ax6.plot(red_props['weighted_centroid-1'], red_props['weighted_centroid-0'], 'o', color = 'r', markersize = 0.3);
    contours = utils.DrawMarkerContours(dapi_props.coords, ax6, color = 'darkblue', lw = 0.5)
        
    # save figure
    utils.saveFig(os.path.basename(czi_file).strip('.czi') + '_OutImg.png', folder = output_folder+'/')
    
#if __name__ == "__main__":
#    # Add your main execution code here
#    NuclearSegmentation(czi_file)
    