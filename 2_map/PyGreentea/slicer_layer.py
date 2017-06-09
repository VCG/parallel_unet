from __future__ import print_function
import sys, os, math
import json
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

import caffe


class SlicerLayer(caffe.Layer):
    """
    Splits a 3D volume into three stack of 2D images, 
    resulting into XY, YZ, ZX volumes
    """

    def setup(self, bottom, top):

        json_object = json.loads( self.param_str )
        self.offsets = json_object['offsets']
        self.sizes = json_object['sizes']
        self.index = 0

        # check input pair
        print('#bottoms:', len(bottom))
        print('#tops:', len(top))
        print('bot shape:', *bottom[0].shape)
        print('top shape:', *top[0].shape)
        print('sizes:', self.sizes)
        print('offsets:', self.offsets)

        if len(bottom) != 1:
            raise Exception("Need one bottom to split data")


    def reshape(self, bottom, top):

        #print('VolumeSlicer.reshape')
        bot_shape = bottom[0].shape

        # reshape the tops to the faces of the 3D volume
        for i in range(len(top)):
            top[i].reshape( 1, 1, bot_shape[-2], bot_shape[-1] )

    def forward(self, bottom, top):

        #print('SlicerLayer # %d/%d:'%(self.index, self.sizes[0]) )       
        #print('sizes:', self.sizes)
        #print('offsets:', self.offsets)
        #print(*bottom[0].shape)
        #print(bottom[0].data)

        offset = self.index + self.offsets[0]
        top[0].data[0,0,:,:] = bottom[0].data[0,0,offset,:,:] # z or xy
        top[1].data[0,0,:,:] = bottom[0].data[0,0,:,offset,:] # y or zx
        top[2].data[0,0,:,:] = bottom[0].data[0,0,:,:,offset] # x or yz
        self.index += 1

        if self.index == self.sizes[0]:
            #print('resetting index to zero...')
            self.index = 0
        
    def backward(self, top, propagate_down, bottom):
        pass
