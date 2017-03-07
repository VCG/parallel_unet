from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

import caffe


class SliceVolumeLayer(caffe.Layer):
    """
    Splits a 3D volume into three stack of 2D images, 
    resulting into XY, YZ, ZX volumes
    """

    def setup(self, bottom, top):
        # check input pair
        bot_shape = bottom[0].shape

        if len(bottom) != 1:
            raise Exception("Need one bottom to split data")

    def reshape(self, bottom, top):
        bot_shape = bottom[0].shape

        # reshape the tops to the faces of the 3D volume
        top[0].reshape( bot_shape[-3], 1, bot_shape[-2], bot_shape[-1] )
        top[1].reshape( bot_shape[-2], 1, bot_shape[-3], bot_shape[-1] )
        top[2].reshape( bot_shape[-1], 1, bot_shape[-3], bot_shape[-2] )


    def forward(self, bottom, top):

        axes = [-3,-2,-1]
        for i in range(len(top)):
            axis = axes[i]

            for s in range(top[i].shape[0]):
                if axis == -3:
                    top[i].data[0,0,:,:] = bottom[0].data[0,0,s,:,:]
                elif axis == -2:
                    top[i].data[0,0,:,:] = bottom[0].data[0,0,:,s,:]
                elif axis == -1:
                    top[i].data[0,0,:,:] = bottom[0].data[0,0,:,:,s]
        
    def backward(self, top, propagate_down, bottom):
        pass

