from __future__ import print_function
import sys, os, math
import json
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

import caffe


class TransformerLayer(caffe.Layer):
    """
    Splits a 3D volume into three stack of 2D images, 
    resulting into XY, YZ, ZX volumes
    """
    def setup(self, bottom, top):

        json_object = json.loads( self.param_str )
        self.offsets = json_object['offsets']
        self.sizes = json_object['sizes']
        self.index = 0

        shape = bottom[0].shape
        self.data = np.zeros((3, shape[-1], shape[-1], shape[-1] ))

        # check input pair
        print('#bottoms:', len(bottom))
        print('#tops:', len(top))
        print('bot shape:', *bottom[0].shape)
        print('top shape:', *top[0].shape)
        print('sizes:', self.sizes)
        print('offsets:', self.offsets)

        if len(bottom) != 3:
            raise Exception("Need three bottoms (z, y, x) ")


    def reshape(self, bottom, top):
        #print('Transformer.reshape')
        bot_shape = bottom[0].shape

        # reshape the tops to the faces of the 3D volume
        for i in range(len(top)):
            top[i].reshape( 1, 3, bot_shape[-1], bot_shape[-1], bot_shape[-1]) #self.sizes[0], self.sizes[1], self.sizes[2])

    def forward(self, bottom, top):
        #print('TransformerLayer # %d/%d:'%(self.index, self.sizes[0]) )       
        #print('sizes:', self.sizes)
        #print('offsets:', self.offsets)
        #print(*bottom[0].shape)
        #print(bottom[0].data)
        if self.index == 0:
            self.data[...] = 0.0

        self.data[0,self.index,:,:] = bottom[0].data[0,0,:,:]
        self.data[1,self.index,:,:] = bottom[1].data[0,0,:,:]
        self.data[2,self.index,:,:] = bottom[2].data[0,0,:,:] 
        self.index += 1

        enable = 0
        if self.index == self.data.shape[-1]:  #self.sizes[0]:
            enable = 1
            self.index = 0
            top[0].data[0,0,:,:,:] = self.data[0,:,:,:]
            top[0].data[0,1,:,:,:] = self.data[1,:,:,:]
            top[0].data[0,2,:,:,:] = self.data[2,:,:,:]

            #top[0].data[...] = top[0].data[...]/self.sizes[0]

        if len(top) > 1:
            top[1].data[0] = enable

        
    def backward(self, top, propagate_down, bottom):
        #print('TransformerLayer.backward')
        z = top[0].diff[0,0,0,:,:]
        y = top[0].diff[0,1,0,:,:]
        x = top[0].diff[0,2,0,:,:]
        for i in range(top[0].shape[-1]):
            z += top[0].diff[0,0,i,:,:]
            y += top[0].diff[0,1,i,:,:]
            x += top[0].diff[0,2,i,:,:]
        bottom[0].diff[0,0,:,:] = z/top[0].shape[-1]
        bottom[1].diff[0,0,:,:] = y/top[0].shape[-1]
        bottom[2].diff[0,0,:,:] = x/top[0].shape[-1]
