# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 09:53:48 2022
@author: Administrator
"""
import cv2 as cv
import os
import numpy as np
from copy import deepcopy
import surface_distance as surfdist

def reconstructionPRE(ptid, valimgRes, size, axis):
    allPatientSlice = os.listdir(valimgRes)
    sliceList = [allPatientSlice[i] for i in range(len(allPatientSlice)) if ptid in allPatientSlice[i]]
    sliceList.sort(key=lambda info: (info[0:2], int(info.split('.')[0].split('_')[1])))
    sliceLists = [int(i.split('.')[0].split('_')[1]) for i in sliceList]
    slice0 = reconstruction(sliceList, axis, size, valimgRes)
    return sliceList, slice0, len(sliceList), sliceLists

def reconstruction(sliceList, axis, size, valimgRes):
    slice0 = base(axis, size)
    a = deepcopy(slice0)
    for img in sliceList:
        if '-' not in img:
            path_img = os.path.join(valimgRes, img)
            _slice = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
            _slice = np.resize(_slice, a.shape)
            _slice = _slice / 255
            slice0 = np.concatenate((slice0, _slice), axis=axis)
    slice0 = mydrop(axis, slice0)
    return slice0

def mydrop(axis, slice0):
    if axis == 0:
        slice0 = slice0[1:, :, :]
    if axis == 1:
        slice0 = slice0[:, 1:, :]
    if axis == 2:
        slice0 = slice0[:, :, 1:]
    return slice0

def base(axis, size):
    if axis == 0:
        _slice = np.zeros([1, size[0], size[1]])
    if axis == 1:
        _slice = np.zeros([size[0], 1, size[1]])
    if axis == 2:
        _slice = np.zeros([size[0], size[1], 1])
    return _slice

def binarys(_slice):
    _slice[_slice >= 0.5] = 1
    _slice[_slice < 0.5] = 0
    return _slice

def reconstructionMASK(sliceList, size, axis, path_true):
    groundTruth = reconstruction(sliceList, axis, size, path_true)
    return groundTruth


def dice(pd, gt):
    return 2 * np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())


def precision(pd, gt):
    return np.logical_and(pd, gt).sum() / pd.sum()


def sensitivity(pd, gt):
    return np.logical_and(pd, gt).sum() / gt.sum()


def specificity(pd, gt):
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    return np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()


def ADHD(pd, gt):
    pd = np.asarray(pd, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    surface_distances = surfdist.compute_surface_distances(
        gt, pd, spacing_mm=(1.5, 1.5, 1.5))
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    return hd_dist_95, avg_surf_dist


