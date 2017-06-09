from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

import caffe


class MergeDataLayer(caffe.Layer):
    """
    Splits a 3D volume into three stack of 2D images
    """

    def setup(self, bottom, top):
        print('MergeDataLayer.setup')
        print(type(bottom))
        print(type(top))
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three bottoms to merge")

    def reshape(self, bottom, top):
        # check input dimensions match
        print('MergeDataLayer.reshape')

        for i in range(len(bottom)):
            print('bottom-----')
            for j in range(len(bottom[i].shape)):
                print(bottom[i].shape[j])

        shape = bottom[0].shape
        # what about anisotrophic
        top[0].reshape( 1, len(bottom), shape[2], shape[2], shape[3] )

    def forward(self, bottom, top):
        print('MergeDataLayer.forward')
        #self.diff[...] = bottom[0].data - bottom[1].data
        #top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

        for i in range(len(bottom)):
            print('bottom-----', i, *bottom[i].shape)

        for i in range(len(top)):
            print('top-----', i,*top[0].shape)

        # first dimension of source and destination are expected to be 1
        #for i in range(len(bottom)):
        #    top[0].data[0,i,:,:,:] = bottom[i].data[0,:,:,:]

        # --------
        # transform the data so its in the form of 
        # (feature_maps, num_slices, width, height)
        for i in range(len(bottom)):

            b = self.to_featuremap_based( bottom[0].data )
            print('bot:', *b.shape)

            b = self.crop( b )
            print('cropped:', *b.shape)

            #yoffset = (bottom[0].shape[0] - bottom[0].shape[-2])/2;
            #xoffset = (bottom[0].shape[0] - bottom[0].shape[-1])/2;
            #print( 'yoffset:',yoffset, 'xoffset:', xoffset )
            '''
            for f in range(top[0].shape[-3]):
                for s in range(top[0].shape[-4]):
                    top[0].data[0,f,s,:,:] = bottom[i].data[0,f,:,:]
            '''

        #exit(1)
        #---------
        # Compute the coordinates to extract the 12,12,12 volume which 
        # will be send to the loss layer for detection.


    def backward(self, top, propagate_down, bottom):
        print('MergeDataLayer.backward')
        print ('#top:', len(top))
        print ('#bot:', len(bottom))
        #exit(1)

    def crop(self, data):
        shape = data.shape
        size = min([data.shape[-3], data.shape[-2], data.shape[-1]])
        new = np.zeros((shape[0], size, size, size))

        offsets = [0,0,0]
        offsets[0] = (shape[1] - size)/2
        offsets[1] = (shape[2] - size)/2
        offsets[2] = (shape[2] - size)/2

        for i in range(shape[0]):
            new[i,:,:,:] = data[i,offsets[0]:offsets[0]+size, offsets[1]:offsets[1]+size, offsets[2]:offsets[2]+size]

        return new

    def to_featuremap_based(self, data):
        shape = data.shape
        new = np.zeros((shape[1], shape[0], shape[2], shape[3] ))
        for i in range(shape[1]):
            for j in range(shape[0]):
                new[i,j,:,:] = data[j,i,:,:]
        return new

