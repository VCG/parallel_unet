from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

import caffe


class TransformVolumesLayer(caffe.Layer):
    """
    Transforms 2D volumes from batch-based to feature-map based
    """

    def setup(self, bottom, top):
        print('TransformVolumesLayer.setup')
        print(type(bottom))
        print(type(top))
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three bottoms to transform")

    def reshape(self, bottom, top):
        # check input dimensions match
        print('TransformVolumesLayer.reshape')

        shape = bottom[0].shape
        size = min([shape[-4], shape[-2], shape[-1]])

        print('#tops:', len(top))
        print('#bots:', len(bottom))
        print(*shape)

        # difference is shape of inputs
        self.diff = [ np.zeros_like(bottom[i].data, dtype=np.float32) for i in range(len(bottom))]

        # reshape the tops
        top[0].reshape( 1, shape[-3], size, size, size )
        top[1].reshape( 1, shape[-3], size, size, size )
        top[2].reshape( 1, shape[-3], size, size, size )

        for i in range(len(top)):
            print('top i:', i, 'shape:', *top[i].shape)

    def forward(self, bottom, top):
        print('TransformVolumesLayer.forward')
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

            # transformed output to featuremap-based
            # <1, featuremap, z, y, x>
            b = self.to_featuremap_based( bottom[0].data )
            print('bot:', *b.shape)

            # crop the output
            self.crop_and_set( b, top[i] )

        # re-arrange output (xs together, ys together, zs together)
        # indices of the tops whose data need to be group
        top_indices = [(0,2), (0,1), (2, 1)] 

        # indices of the feature maps to be grouped
        fmap_indices = [(0,1), (1,0), (0,1)]

        # group the feature maps for each top
        for i in range( len(top) ):
            
            t = top_indices[i]
            f = fmap_indices[i]
           
            data1 = np.copy( top[ t[0] ].data[:,f[0],:,:,:] )
            data2 = np.copy( top[ t[1] ].data[:,f[1],:,:,:] )

            top[ i ].data[:,0,:,:,:] = data1
            top[ i ].data[:,1,:,:,:] = data2 
   
    def backward(self, top, propagate_down, bottom):
        print('TransformVolumesLayer.backward')

        # indices of the diffs to copy to 
        diff_indices = [(0,1), (1,2), (2,0)]

        # indices of the feature maps copy to
        fmap_indices = [(0,1), (1,0), (0,1)]

        diffs = []
        # ungroup the planes
        for i in range(len(top)):
            t = diff_indices[i]
            f = fmap_indices[i]

            shape = top[i].diff.shape
            print('diffshape:', *shape)
            diffs.append( np.zeros( (shape[-4], shape[-3], shape[-2], shape[-1]) ) )
      
            print('#diffs:', len(diffs))
 
            # propate the errors to the correct position in the 2D networks 
            diffs[i][0,:,:,:] = np.copy( top[ t[0] ].diff[0,f[0],:,:,:] )
            diffs[i][1,:,:,:] = np.copy( top[ t[1] ].diff[0,f[1],:,:,:] )
        
            print('i:', i, 'shape:', *diffs[i].shape)            

        offsets = [0,0,0]
       
        # copy data diffs 
        for d in range( len(diffs) ):
            shape = self.diff[0].shape
            size = min([shape[-4], shape[-2], shape[-1]])

            offsets[0] = (shape[-4] - size)/2
            offsets[1] = (shape[-2] - size)/2
            offsets[2] = (shape[-1] - size)/2
            print('offsets:', *offsets)
            print('shape:',*shape)
            print('size:', size)

            #from: 2, 12, 12, 12
            #to: 100,2,12,12
            shape = diffs[d].shape
            for i in range(shape[1]): # i=0..11
                print('i:', i, *diffs[d].shape)
                for j in range(shape[0]): # j=0..1
                    self.diff[d][i+offsets[0],j,:,:] = diffs[d][j,i,:,:]

    def crop_and_set(self, data, t):
        shape = data.shape
        size = min([data.shape[-3], data.shape[-2], data.shape[-1]])
        new = np.zeros((shape[0], size, size, size))

        offsets = [0,0,0]
        offsets[0] = (shape[1] - size)/2
        offsets[1] = (shape[2] - size)/2
        offsets[2] = (shape[2] - size)/2

        print('offsets:', *offsets)
        for i in range(shape[0]):
            t.data[:,i,:,:,:] = data[i,offsets[0]:offsets[0]+size, offsets[1]:offsets[1]+size, offsets[2]:offsets[2]+size]


    def to_featuremap_based(self, data):
        shape = data.shape
        #from 100,2,12,12
        #to   2, 100,12,12
        new = np.zeros((shape[1], shape[0], shape[2], shape[3] ))
        for i in range(shape[1]):
            for j in range(shape[0]):
                new[i,j,:,:] = data[j,i,:,:]
        return new

