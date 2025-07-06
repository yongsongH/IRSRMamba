# Copyright (c) 2023, Yongsong Huang, Email: hyongsong@ieee.org
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is designed for generating high-resolution (HR) and low-resolution (LR) image pairs 
from a source directory of images. It is commonly used for preparing datasets for 
Super-Resolution (SR) tasks.

The script performs the following steps:
1. Reads images from a specified source directory.
2. Applies a 'modcrop' to the images to ensure their dimensions are divisible by the largest scaling factor.
3. Saves the cropped high-resolution (HR) images.
4. Generates low-resolution (LR) images by downscaling the HR images using a bicubic interpolation method.
5. Saves the LR images in separate subdirectories based on the scaling factor (e.g., X2, X4).

Usage:
    - Modify the `dataDir` and `saveDir` variables in the `if __name__ == "__main__":` block.
    - `dataDir`: Path to the directory containing the original high-resolution images.
    - `saveDir`: Path to the directory where the generated HR and LR image pairs will be saved.
    - Run the script from the command line:
        python bic_dataset.py
"""

import os
import sys
import cv2
import torch
import math
import numpy as np
from tqdm import tqdm

def cubic(x):
    """
    Cubic interpolation kernel.
    This function defines the cubic interpolation kernel used for resizing.
    """
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))

def CalculateWeightsIndices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """
    Calculates the weights and indices for the interpolation.
    This is a key function for the bicubic resizing process, preparing the necessary
    convolution weights and corresponding pixel indices.
    """
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias - larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # Left-most pixel involved in the computation
    left = torch.floor(u - kernel_width / 2)

    # Maximum number of pixels involved
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output pixel
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    
    # Apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
        
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
        
    weights = weights.contiguous()
    indices = indices.contiguous()
    
    # Handle boundary conditions
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    
    return weights, indices, int(sym_len_s), int(sym_len_e)

def ImResizeNp(img, scale, antialiasing=True):
    """
    Performs bicubic resizing on a NumPy image.
    Args:
        img (np.array): Input image in HWC BGR format [0,1].
        scale (float): The scaling factor.
        antialiasing (bool): Whether to apply antialiasing.
    Returns:
        np.array: Resized image in HWC BGR format [0,1].
    """
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    out_H, out_W = math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Get weights and indices for horizontal and vertical dimensions
    weights_H, indices_H, sym_len_Hs, sym_len_He = CalculateWeightsIndices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = CalculateWeightsIndices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
        
    # Process H dimension with symmetric padding
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    # Apply convolution for H dimension
    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for c in range(in_C):
            out_1[i, :, c] = img_aug[idx:idx + kernel_width, :, c].transpose(0, 1).mv(weights_H[i])

    # Process W dimension with symmetric padding
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    # Apply convolution for W dimension
    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for c in range(in_C):
            out_2[:, i, c] = out_1_aug[:, idx:idx + kernel_width, c].mv(weights_W[i])

    return out_2.numpy()


def GenerateMod(sourcedir, savedir):
    """
    Generates mod-cropped HR images and downscaled LR images.
    Args:
        sourcedir (str): Directory containing the original images.
        savedir (str): Directory to save the processed HR and LR images.
    """
    # Set parameters
    scales = [2, 4]  # Upsampling scales

    saveHRpath = os.path.join(savedir, 'HR')
    saveLRpath = os.path.join(savedir, 'LR_bicubic')

    print("Save HR Dir: ", saveHRpath)
    print("Save LR Dir: ", saveLRpath)

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will overwrite ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will overwrite ' + str(saveLRpath))

    # Create directories for different scales
    for scale in scales:
        saveLRpath_X = os.path.join(saveLRpath, 'X' + str(scale))
        if not os.path.isdir(saveLRpath_X):
            os.mkdir(saveLRpath_X)
        else:
            print('It will overwrite ' + str(saveLRpath_X))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.jpg')]  # Image file type
    filepaths = sorted(filepaths)
    
    # Prepare data with augmentation
    for filename in tqdm(filepaths, desc=f"Processing {os.path.basename(sourcedir)}"):
        # Read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        # Calculate dimensions for modcrop
        width = int(np.floor(image.shape[1] / max(scales)))
        height = int(np.floor(image.shape[0] / max(scales)))
        
        # Apply modcrop
        if len(image.shape) == 3:
            image_HR = image[0:max(scales) * height, 0:max(scales) * width, :]
        else:
            image_HR = image[0:max(scales) * height, 0:max(scales) * width]

        # Save the HR image
        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)

        # Generate and save LR images for each scale
        for scale in scales:
            # Downscale
            image_LR = ImResizeNp(image_HR.astype(np.float32) / 255.0, 1 / scale, True)
            image_LR = np.clip(image_LR * 255.0, 0, 255.0).round().astype(np.uint8)
            # Save
            cv2.imwrite(os.path.join(saveLRpath, 'X' + str(scale), filename[:-4] + 'x' + str(scale) + '.jpg'), image_LR)

if __name__ == "__main__":
    # --- Configuration ---
    # Path to the original dataset directory. 
    # This directory should contain subdirectories for each dataset (e.g., /RS_Dataset/Set5, /RS_Dataset/Set14)
    # Please check the image type (e.g., .png or .jpg) and modify the `GenerateMod` function if needed.
    dataDir = '/dataPath'  
    # Path to the directory where the results will be saved.
    saveDir = '/savePath'
    
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)

    # Process each subdirectory in the dataDir
    for dirs in os.listdir(dataDir):
        print('Processing dataset:', dirs)
        source_path = os.path.join(dataDir, dirs)
        target_path = os.path.join(saveDir, dirs)
        if os.path.isdir(source_path):
            GenerateMod(source_path, target_path)
