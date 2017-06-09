from __future__ import print_function
import sys, os, math
import json
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

import caffe


class FusionLayer(caffe.Layer):
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

        shape = bottom[0].shape
        self.data = np.zeros((3, 2, shape[-1], shape[-1], shape[-1] ))

        '''
        print('bot shape:', *bottom[0].shape)
        print('top shape:', *top[0].shape)
        print('sizes:', self.sizes)
        print('offsets:', self.offsets)
        '''

        if len(bottom) != 3:
            raise Exception("Need three bottoms (z, y, x) ")


    def reshape(self, bottom, top):
        #print('Fusion.reshape')
        bot_shape = bottom[0].shape

        #print('#bottoms:', len(bottom))
        #print('#tops:', len(top))
        
        # reshape the tops to the faces of the 3D volume
        for i in range(3):
            top[i].reshape( 1, 2, bot_shape[-1], bot_shape[-1], bot_shape[-1])
            #print('shape:', *top[i].shape)

        if len(top) > 3:
            top[-1].reshape( 1 )
            #print('flag shape:', *top[-1].shape)


    def forward(self, bottom, top):
        if self.index == 0:
            self.data[...] = 0.0

        #    01      01      01
        # z (xy), y (zx), x (yz)
        self.data[0,0,:,self.index,:] = bottom[1].data[0,0,:,:] #z1 y[0]
        self.data[0,1,:,:,self.index] = bottom[2].data[0,1,:,:] #z2 x[1]

        self.data[1,0,self.index,:,:] = bottom[0].data[0,1,:,:] #y1 z[1]
        self.data[1,1,:,:,self.index] = bottom[2].data[0,0,:,:] #y2 x[0]

        self.data[2,0,self.index,:,:] = bottom[0].data[0,0,:,:] #x1 z[0]
        self.data[2,1,:,self.index,:] = bottom[1].data[0,1,:,:] #x2 y[1]

        self.index += 1
        enable = 0
        if self.index == self.data.shape[-1]:  #self.sizes[0]:

            # enable loss computation
            enable = 1

            # reset index
            self.index = 0

            # copy the combined data top next layer blobs
            top[0].data[0,0,...] = self.data[0,0,...] #z1
            top[0].data[0,1,...] = self.data[0,1,...] #z2

            top[1].data[0,0,...] = self.data[1,0,...] #y1
            top[1].data[0,1,...] = self.data[1,1,...] #y2

            top[2].data[0,0,...] = self.data[2,0,...] #x1
            top[2].data[0,1,...] = self.data[2,1,...] #x2

            #for i in range(3):
            #    print('----', i)
            #    print(top[i].data)

        if len(top) > 3:
            top[-1].data[0] = enable

        
    def backward(self, top, propagate_down, bottom):
        multi_pass = True
        if multi_pass:
            bottom[0].diff[0,0,:,:] = top[2].diff[0,0,self.index,:,:] #x1
            bottom[0].diff[0,1,:,:] = top[1].diff[0,0,self.index,:,:] #y1

            bottom[1].diff[0,0,:,:] = top[0].diff[0,0,:,self.index,:] #z1
            bottom[1].diff[0,1,:,:] = top[2].diff[0,1,:,self.index,:] #x2

            bottom[2].diff[0,0,:,:] = top[1].diff[0,1,:,:,self.index] #y2
            bottom[2].diff[0,1,:,:] = top[0].diff[0,1,:,:,self.index] #z2

            self.index = self.index + 1
            if self.index == self.data.shape[-1]:
                self.index = 0
            return
        else:

            for i in range(3):
                bottom[i].diff[...] = 0.0

            n = top[0].shape[-1]
            for i in range(n):
                bottom[0].diff[0,0,:,:] += top[2].diff[0,0,:,:,i] #x1
                bottom[0].diff[0,1,:,:] += top[1].diff[0,0,:,i,:] #y1

                bottom[1].diff[0,0,:,:] += top[0].diff[0,0,i,:,:] #z1
                bottom[1].diff[0,1,:,:] += top[2].diff[0,1,:,:,i] #x2

                bottom[2].diff[0,0,:,:] += top[1].diff[0,1,:,i,:] #y2
                bottom[2].diff[0,1,:,:] += top[0].diff[0,1,i,:,:] #z2

            for i in range(3):
                bottom[i].diff[...] = bottom[i].diff[...]/n
