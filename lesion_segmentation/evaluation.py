# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:32:46 2022
3D reconstruction and evaluation of 2D prediction results in three directions
select the best model in three directions
@author: Administrator
"""

from evaluation_util import *
import pandas as pd
from options import SegmentationOptions
options = SegmentationOptions()
opts = options.parse()
resroot = opts.res_root
trainroot = opts.data_root

###################################################
# Record the size of the input picture in three directions
# The first one represents the size of the original 2D slice
# the second represents the size of the input to network after padding and cropping
# the third one represents the reconstruction axis
###################################################

size = {'X': [(145, 121, 1), (144, 128), 2],
        'Y': [(121, 121, 1), (128, 128), 1],
        'Z': [(121, 145, 1), (128, 144), 0]}

indexs = ['dice', 'precision', 'sensitivity', 'specificity', 'AD', 'HD']

# create record
col = []
for i in size.keys():
    for j in indexs:
        col.append(i + '_' + j)
record = pd.DataFrame(columns=col, index=opts.ptid_list, dtype=object)


# 3D reconstruction and evaluation
for direction, v in size.items():
    directionRes = os.path.join(resroot, direction, 'predMask')  #The folder where the predicted slice stored
    path_true = os.path.join(trainroot, direction, 'train', 'MASK') #The folder where the groundtrth slice stored
    Dindexs = list(direction + '_' + i for i in indexs)
    for val in os.listdir(directionRes):
        valimgRes = os.path.join(directionRes, val)
        ptids = val.split('_')
        sizes = v[1]
        axis = v[2]
        for ptid in ptids:
            sliceList, prd, SliceNum, slice2Dindex = reconstructionPRE(ptid, valimgRes, sizes, axis)
            prdBin = binarys(prd)
            gt = reconstructionMASK(sliceList, sizes, axis, path_true)
            # Evaluate and save the  results
            dices = dice(prdBin, gt)
            precisions = precision(prdBin, gt)
            sensitivitys = sensitivity(prdBin, gt)
            specificitys = specificity(prdBin, gt)
            HD, AD = ADHD(prdBin, gt)
            value = [dices, precisions, sensitivitys, specificitys,HD, (AD[0] + AD[1]) / 2]
            for i in range(len(value)):
                record.loc[ptid, Dindexs[i]] = value[i]




