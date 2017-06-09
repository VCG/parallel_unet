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

# Create the network we want
netconf = pygt.netgen.NetConf()
netconf.ignore_conv_buffer = True
netconf.use_batchnorm = False
netconf.dropout = 0.0
netconf.fmap_start = 24
netconf.u_netconfs[0].unet_fmap_inc_rule = lambda fmaps: int(math.ceil(fmaps * 3))
netconf.u_netconfs[0].unet_fmap_dec_rule = lambda fmaps: int(math.ceil(fmaps / 3))
netconf.fmap_output = 2
netconf.fmap_output3d = 3

netconf.input_shape = [132,132]
netconf.output_shape = [44, 44]

netconf.input_shape3d = [132,132,132]
netconf.output_shape3d = [44, 44, 44]

print ('Input shape: %s' % netconf.input_shape)
print ('Output shape: %s' % netconf.output_shape)
print ('Feature maps: %s' % netconf.fmap_start)

netconf.loss_function = "euclid"
train_net_conf_euclid, test_net_conf = pygt.netgen.create_nets(netconf)
netconf.loss_function = "malis"
train_net_conf_malis, test_net_conf = pygt.netgen.create_nets(netconf)

with open('net_train_euclid.prototxt', 'w') as f:
    print(train_net_conf_euclid, file=f)
with open('net_train_malis.prototxt', 'w') as f:
    print(train_net_conf_malis, file=f)
with open('net_test.prototxt', 'w') as f:
    print(test_net_conf, file=f)

#### Make a big test proto
# Biggest possible network for testing on 12 GB
netconf.ignore_conv_buffer = True
netconf.mem_global_limit = 8 * 1024 * 1024 * 1024
mode = pygt.netgen.caffe_pb2.TEST
shape_min = [100,100,100]
shape_max = [520,520,520]
constraints = [None, lambda x: x[0], lambda x: x[1]]

inshape,outshape,fmaps = pygt.netgen.compute_valid_io_shapes(netconf,mode,shape_min,shape_max,constraints=constraints)

# We choose the maximum that still gives us 20 fmaps:
index = [n for n, i in enumerate(fmaps) if i>=netconf.fmap_start][-1]
print("Index to use: %s" % index)

# Some patching to allow our new parameters
netconf.input_shape3d = inshape[index]
netconf.output_shape3d = outshape[index]
netconf.input_shape = netconf.input_shape[1:]
netconf.output_shape = netconf.output_shape[1:]

netconf.input_shape = [204,204]
netconf.output_shape = [116,116]
netconf.input_shape3d = [204,204,204]
netconf.output_shape3d = [116,116,116]

# Workaround to allow any size (train net unusably big)
netconf.mem_global_limit = 200 * 1024 * 1024 * 1024
netconf.mem_buf_limit = 200 * 1024 * 1024 * 1024
# Define some loss function (irrelevant for testing though)
netconf.loss_function = "euclid"

# Generate the nework, store it
train_net_big_conf, test_net_big_conf = pygt.netgen.create_nets(netconf)

with open('net_test_big.prototxt', 'w') as f:
    print(test_net_big_conf, file=f)
