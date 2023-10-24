#######preprocessing for SETD model########     
from scipy import ndimage
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import glob
import numpy as np

def maskcroppingbox(data, mask):
    mask_2 = np.argwhere(mask)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = mask_2.min(axis=0)-1, mask_2.max(axis=0) + 1
    zstart = max(0, zstart)
    ystart =max(0, ystart)
    xstart = max(0, xstart)
    zstop = min(data.shape[0], zstop)
    ystop = min(data.shape[1], ystop)
    xstop = min(data.shape[2], xstop)
    roi_image = data[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_mask = mask[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_image[roi_mask < 1] = 0
    return roi_image

dataPath = '/your/path/to/separated_masks'
savepath ='/your/path/to/saved_processed'
featureDict = {}
dictkey = {}
for imgPath in glob.glob('/your/path/to/separated_masks/*org.nii.gz'):
    fileName = os.path.basename(imgPath)
    labelPath = os.path.join(dataPath, fileName.replace('org.nii.gz', 'seg.nii.gz'))   
    sitkImage = sitk.ReadImage(imgPath)
    data = sitk.GetArrayFromImage(sitkImage)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(labelPath))
    data = maskcroppingbox(data, mask)
    data = sitk.GetImageFromArray(data)
    savePath = os.path.join(savepath, fileName.replace('org.nii.gz', 'rorg.nii.gz'))
    sitk.WriteImage(data, savePath)
