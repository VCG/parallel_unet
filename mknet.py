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

def gen_net(loss_function):

    # Create the network we want
    netconf = pygt.netgen.NetConf()
    netconf.ignore_conv_buffer = True
    netconf.use_batchnorm = False
    netconf.dropout = 0.0
    netconf.fmap_start = 24
    netconf.u_netconfs[0].unet_fmap_inc_rule = lambda fmaps: int(math.ceil(fmaps * 3))
    netconf.u_netconfs[0].unet_fmap_dec_rule = lambda fmaps: int(math.ceil(fmaps / 3))
    #netconf.u_netconfs[0].unet_downsampling_strategy = [[1,2,2],[1,2,2],[1,2,2]]
    netconf.u_netconfs[0].unet_downsampling_strategy = [[2,2],[2,2],[2,2]]
    netconf.u_netconfs[0].unet_depth = 3
    netconf.loss_function = loss_function

    netconf.input_shape = [100,100,100]
    netconf.output_shape = [12,12,12]

    train_net_conf, test_net_conf = pygt.netgen.create_xyz_nets(netconf)

    net_name = 'net_train_%s.prototxt'%(loss_function)

    with open(net_name, 'w') as f:
        print(train_net_conf, file=f)

    with open('net_test.prototxt', 'w') as f:
        print(test_net_conf, file=f)

# generate networks
gen_net('malis')
gen_net('euclid')
