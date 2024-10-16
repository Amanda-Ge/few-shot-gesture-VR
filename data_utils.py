import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from random import randint,shuffle
import numpy as np

# plt.switch_backend('agg')

import os

def denoise_all(stroke_all, input_dim):
    """
    smoothing filter to mitigate some artifacts of the data collection
    """
    stroke_new_all = []
    for coords in stroke_all:
        stroke = denoise(coords, input_dim)
        stroke_new_all.append(stroke)
    return stroke_new_all

def interpolate_all(stroke_all, max_x_length, input_dim):
    """
    interpolates strokes using cubic spline
    """
    coords_all = []
    for stroke in stroke_all:
        coords = interpolate(stroke, max_x_length, input_dim)
        coords_all.append(coords)
    return coords_all

def interpolate(stroke, max_x_length, input_dim):
    coords = np.zeros([input_dim, max_x_length], dtype=np.float32)
    if len(stroke) > 3:
        for j in range(input_dim):
            f_x = interp1d(np.arange(len(stroke)), stroke[:, j], kind='linear')
            xx = np.linspace(0, len(stroke) - 1, max_x_length)
            # xx = np.random.uniform(0,len(stroke)-1, max_x_length)
            x_new = f_x(xx)
            coords[j, :] = x_new
    coords = np.transpose(coords)
    return coords


def multiplier(data,label,multi):
    data_ = data
    label_ = label
    for i in range(multi):
        data_ = np.concatenate((data_, data))
        label_ = np.concatenate((label_, label))

    data = data_
    label = label_

    return data,label

def denoise(coords, input_dim):
    stroke = savgol_filter(coords[:, 0], 7, 3, mode='nearest')
    for i in range(1, input_dim):
        x_new = savgol_filter(coords[:, i], 7, 3, mode='nearest')
        stroke = np.hstack([stroke.reshape(len(coords), -1), x_new.reshape(-1, 1)])
    return stroke


def shuffle(data,label):
    data = np.asarray(data,dtype = object)
    shuffled_indexes = np.random.permutation(np.shape(data)[0])
    data = data[shuffled_indexes]
    label = label[shuffled_indexes]
    return data, label


def relative_track_batch(length, input_dim, data):
    ### do this every iteration to make first step replacement=0
    temp = data[:, 0:input_dim]
    lastplace = np.zeros([length, 1])
    # x
    lastplace[1: length, 0] = temp[0: length - 1, 0]
    lastplace[0, 0] = temp[0, 0]
    temp[0: length, 0] -= lastplace[:, 0]
    # y
    lastplace[1: length, 0] = temp[0: length - 1, 1]
    lastplace[0, 0] = temp[0, 1]
    temp[0: length, 1] -= lastplace[:, 0]
    # z
    lastplace[1: length, 0] = temp[0: length - 1, 2]
    lastplace[0, 0] = temp[0, 2]
    temp[0: length, 2] -= lastplace[:, 0]
    temp = np.reshape(temp, [length, input_dim])
    return temp


def nortowrist(data, xlist):
    ###### normalize to wrist
    ylist = np.add(xlist, 1)
    zlist = np.add(xlist, 2)

    xc = data[:, 0]
    yc = data[:, 1]
    zc = data[:, 2]
    data[:, xlist] -= np.tile(
        np.reshape(xc, [-1, 1]), (1, len(xlist)))
    data[:, ylist] -= np.tile(
        np.reshape(yc, [-1, 1]), (1, len(ylist)))
    data[:, zlist] -= np.tile(
        np.reshape(zc, [-1, 1]), (1, len(zlist)))

    return data


def preprocess(data, xlist,nor_to_wrist=False, relative=False):
    # relative replacement or not
    # use center as track or not, if not, use wrist as track
    length = np.shape(data)[0]
    input_dim = np.shape(data)[1]

    if relative:
        if nor_to_wrist:
            data = nortowrist(data, xlist)
            data = relative_track_batch(length, input_dim, data)
        else:
            data = relative_track_batch(length, input_dim, data)
    elif not relative:
        if nor_to_wrist:
            data = nortowrist(data, xlist)
        else:
            data = data

    return data
