# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:23:57 2024

@author: paulo
"""

import numpy as np
import cv2 as cv
import skimage as sk
import scipy.ndimage as ndi

def FociSegmentation(im_gray, sigma=51, offset=-1, min_size=100, max_size=np.inf):
    '''Returns a binary image of small spots in the image.
    
    Parameters:
    - im_gray: Input grayscale image.
    - sigma: Block size for the local thresholding.
    - offset: Offset for local thresholding.
    - min_size: Minimum size of objects to retain.
    - max_size: Maximum size of objects to retain.
    
    Returns:
    - Binary image with small spots, excluding objects that are too small or too large.
    '''

    # Apply local threshold filter
    th = sk.filters.threshold_local(im_gray, block_size=sigma, offset=offset, method = 'gaussian')
    im_seg = im_gray > th
    
    # Remove small objects
    im_seg = sk.morphology.remove_small_objects(im_seg, min_size=min_size)
    
    # Remove large objects by size filtering
    if max_size < np.inf:
        labeled_image, num_labels = sk.morphology.label(im_seg, connectivity=2, return_num=True)
        sizes = np.bincount(labeled_image.ravel())
        mask_size = (sizes >= min_size) & (sizes <= max_size)
        mask_size[0] = 0  # Background label should not be filtered
        im_seg = np.isin(labeled_image, np.nonzero(mask_size)[0])

    # make shapes a bit more rounded by applying morphological operations
        
    # close small holes in the objects
    dim = sk.morphology.disk(3)
    
    im_seg = sk.morphology.closing(im_seg, footprint=dim)
    
    # remove small noise and smooth the edges
    im_seg = sk.morphology.opening(im_seg, footprint=dim)
    
    # Optional: Apply dilation to enhance the roundness of particles
    im_seg = sk.morphology.dilation(im_seg, footprint=dim)
    
    return im_seg.astype("float")

def NucleiSegmentation(im_dapi, sigma = 31, cell_dist = 31, 
                       clear_border = True, small_objects = 10000):
    
    # apply threshold and watershed segmentation to nuclei
    im_thres = MakeBinary(im_dapi, sigma = sigma, iter = 1)
    im_markers = WatershedSplit(im_thres, min_dist = cell_dist, 
                                clear_border = clear_border, small_objects = small_objects)
    
    return im_markers

def MakeBinary(im_dapi, sigma = 11, iter = 1):
    
    # smooth image
    im_blur = ndi.gaussian_filter(im_dapi, sigma = sigma)

    # apply threshold
    thres_val = sk.filters.threshold_otsu(im_blur)
    im_thres = im_dapi > thres_val

    # apply morphological processes; filter small particles
    im_thres = ndi.binary_closing(im_thres, iterations=iter).astype('int8')
    
    return im_thres
    
def WatershedSplit(im_binary, min_dist = 30, clear_border = True, small_objects = 10000):
    # compute distance transform
    distance = ndi.distance_transform_edt(im_binary)
    
    # find the local max of each the distance map
    local_max_coords = sk.feature.peak_local_max(distance, min_distance = min_dist)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    # label connectected regions
    markers = sk.measure.label(local_max_mask, connectivity=1)
    watershed_markers = sk.segmentation.watershed(-distance, markers, mask=im_binary)
    
    # remove markers touching the border
    if clear_border == True:
        watershed_markers = sk.segmentation.clear_border(watershed_markers, buffer_size=1)

    # remove small objects
    watershed_markers = sk.morphology.remove_small_objects(watershed_markers, min_size =  small_objects)
    
    # convert to rbg labels for better visuai«lization
    #watershed_markers = sk.color.label2rgb(watershed_markers)

    return watershed_markers