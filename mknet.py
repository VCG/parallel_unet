from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = './PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt


# Create configuration structure for the networks
#-----------------------------------------------------------------------
netconf = pygt.netgen.NetConf()
netconf.ignore_conv_buffer = True
netconf.use_batchnorm = False
netconf.dropout = 0.0
netconf.fmap_start = 24
netconf.u_netconfs[0].unet_fmap_inc_rule = lambda fmaps: int(math.ceil(fmaps * 3))
netconf.u_netconfs[0].unet_fmap_dec_rule = lambda fmaps: int(math.ceil(fmaps / 3))
netconf.u_netconfs[0].unet_downsampling_strategy = [[2,2],[2,2],[2,2]]
netconf.u_netconfs[0].unet_depth = 3

#netconf.input_shape = [124,124,124]
#netconf.output_shape = [36,36,36]
#netconf.input_shape = [100,100,100]
#netconf.output_shape = [12,12,12]
netconf.input_shape = [204,204,204]
netconf.output_shape = [116,116,116]
#netconf.input_shape = [404,404,404]
#netconf.output_shape = [316,316,316]

# enable which networks to generate for training
train = {'malis':False, 'euclid':True}

# Create malis parallel network
#-----------------------------------------------------------------------
if train['malis']:
    netconf.loss_function = 'malis'
    train_net_conf = pygt.netgen.create_train_net( netconf )
    with open('net_train_malis.prototxt', 'w') as f:
        print(train_net_conf, file=f)

# Create euclidean parallel network
#-----------------------------------------------------------------------
if train['euclid']:
    netconf.loss_function = 'euclid'
    train_net_conf = pygt.netgen.create_train_net( netconf )
    with open('net_train_euclid.prototxt', 'w') as f:
        print(train_net_conf, file=f)

# Create test network
#-----------------------------------------------------------------------
test_net_conf = pygt.netgen.create_test_net( netconf )
with open('net_test.prototxt', 'w') as f:
    print(test_net_conf, file=f)


# CREATE BIGGER NETWORK FOR TESTING
#-----------------------------------------------------------------------
netconf.input_shape = [204,204,204]
netconf.output_shape = [116,116,116]
#netconf.input_shape = [124,124,124]
#netconf.output_shape = [36,36,36]


# Workaround to allow any size (train net unusably big)
netconf.mem_global_limit = 200 * 1024 * 1024 * 1024
netconf.mem_buf_limit = 200 * 1024 * 1024 * 1024
# Define some loss function (irrelevant for testing though)
netconf.loss_function = "euclid"

# Generate the nework, store it
test_net_big_conf = pygt.netgen.create_test_net( netconf )
with open('net_test_big.prototxt', 'w') as f:
    print(test_net_big_conf, file=f)
