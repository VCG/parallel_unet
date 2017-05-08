from __future__ import print_function
import sys, os, math
import json
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb
import caffe

class TransformNetLayer(caffe.Layer):
    """
    Transforms 2D Networks into 3D
    """

    def setup(self, bottom, top):
        json_object = json.loads( self.param_str )
        self.sizes = json_object['sizes']
        self.index = 0

        # storage of data and diffs for 2 feature maps of each of the three 2D network arms
        self.data = [ np.zeros( (2,self.sizes[0],self.sizes[1],self.sizes[2]) ) for _ in range(3)]
        self.diff = [ np.zeros( (2,self.sizes[0],self.sizes[1],self.sizes[2]) ) for _ in range(3)]

        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three bottoms to transform")

        if len(top) < 1:
            raise Exception("Need atleast one top")

    def reshape(self, bottom, top):

        # reshape the tops
        # local storage of 2D network arm predictions
        top[0].reshape( 1, 3, self.sizes[-3], self.sizes[-2], self.sizes[-1] ) # 3 maps

        # gate flag
        if len(top) > 1:
            top[1].reshape( 1 )

    def forward(self, bottom, top):

        #print('TransformNetLyaer.forward() -  index:', self.index)#, '#slices:', self.n_slices)

        # accumulate predictions (to be average during loss pass)
        # 2 36 36 36 
        # self.data[0] = z (xy)
        # self.data[1] = y (zx)
        # self.data[2] = x (yz)

        # n is number of slices
        # bottom is 3 arrays of (1, 2, n, n)
        # self.data is 3 arrays of (2, n, n, n)
        # zz = yz[0] and yz[1]
        self.data[0][0,self.index,:,:] = bottom[1].data[0,0,:,:]
        self.data[0][1,self.index,:,:] = bottom[2].data[0,1,:,:]

        # yy = xy[1] and yz[0]
        self.data[1][0,self.index,:,:] = bottom[0].data[0,1,:,:]
        self.data[1][1,self.index,:,:] = bottom[2].data[0,0,:,:]

        # x = yz
        self.data[2][0,self.index,:,:] = bottom[0].data[0,0,:,:]
        self.data[2][1,self.index,:,:] = bottom[1].data[0,1,:,:]

        self.index += 1

        stackFinished = (self.index == self.data[0].shape[-1])

        # average the data if we reached the end of the stack
        if stackFinished:
            self.index = 0

            # n is number of slices
            # top is one array of (1, 3, n, n, n)
            for i in range( len(self.data) ):
                top[0].data[0,i,:,:,:] = (self.data[i][0] + self.data[i][1])/2
                #print('top i:', i)
                #print(top[0].data[0,i,:,:,:])

        if len(top) > 1:
            top[1].data[0] = 1 if stackFinished else 0

    def backward(self, top, propagate_down, bottom):

        average = True

        if self.index == 0:
            #if True:

            # store diffs locally to be propagated one index at a time
            # n is number of slices
            # top is one array of (1, 3, n, n, n)
            # self.diff is 3 arrays of (2, n, n, n) 
            # 1 3 12 12 12 top  (order is z=(xy), y=(zx), x=(yz)
            #   2 12 12 12 diff
            for j in range(top[0].shape[-1]):
                # z=xy
                self.diff[0][0,j,:,:] = top[0].diff[0,2,j,:,:]
                self.diff[0][1,j,:,:] = top[0].diff[0,1,j,:,:]
                # y=zx
                self.diff[1][0,j,:,:] = top[0].diff[0,0,j,:,:]
                self.diff[1][1,j,:,:] = top[0].diff[0,2,j,:,:]
                # x=yz
                self.diff[2][0,j,:,:] = top[0].diff[0,1,j,:,:]
                self.diff[2][1,j,:,:] = top[0].diff[0,0,j,:,:]

            if average: 
                for i in range(len(bottom)):
                    f1 = self.diff[i][0,0,:,:]
                    f2 = self.diff[i][1,0,:,:]
                    for j in range(1, self.diff[0].shape[-1]):
                        f1 += self.diff[i][0,j,:,:]
                        f2 += self.diff[i][1,j,:,:]
                    bottom[i].diff[0,0,:,:] = f1/self.diff[0].shape[-1]
                    bottom[i].diff[0,1,:,:] = f2/self.diff[0].shape[-1]
                    #print('bottom:', i)
                    #print(bottom[i].diff)
                self.index = 0
       
        if not average: 
            # set the diffs of the bottom
            for i in range(len(bottom)):
                bottom[i].diff[0,0,:,:] = self.diff[i][0,self.index,:,:] 
                bottom[i].diff[0,1,:,:] = self.diff[i][1,self.index,:,:] 
            self.index += 1
        

        if self.index == self.data[0].shape[-1]:
            self.index = 0
