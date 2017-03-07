 
from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = './PyGreentea'

sys.path.append(pygt_path)
import PyGreentea as pygt

# Load the datasets
path = '/home/caffe/sources/research/data'

## Training datasets
train_dataset = []
train_dataset.append({})
dname = 'ecs-ruobin'
train_dataset[-1]['name'] = dname
train_dataset[-1]['nhood'] = pygt.malis.mknhood3d()
train_dataset[-1]['data'] =  np.array(h5py.File(join(path,dname,'im_uint8.h5'),'r')['main'],dtype=np.float32)/(2.**8)
#train_dataset[-1]['components'] = np.array(h5py.File(join(path,dname,'seg_groundtruth.h5'),'r')['main'])
train_dataset[-1]['components'] = np.array(h5py.File(join(path,dname,'ac4_groundtruth1.h5'),'r')['main'])
train_dataset[-1]['label'] = pygt.malis.seg_to_affgraph(train_dataset[-1]['components'],train_dataset[-1]['nhood'])
train_dataset[-1]['transform'] = {}
train_dataset[-1]['transform']['scale'] = (0.8,1.2)
train_dataset[-1]['transform']['shift'] = (-0.2,0.2)

print('Training set contains ' + str(len(train_dataset)) + ' volumes')
#pdb.set_trace()
print('Running the simple dataset augmenter...')
train_dataset = pygt.augment_data_simple(train_dataset,trn_method='affinity')
print('Training set now contains ' + str(len(train_dataset)) + ' volumes')

for iset in range(len(train_dataset)):
    train_dataset[iset]['data'] = train_dataset[iset]['data'][None,:] # add a dummy dimension
    #train_dataset[iset]['label'] = train_dataset[iset]['label'][None,:]
    train_dataset[iset]['components'] = train_dataset[iset]['components'][None,:]
    print(train_dataset[iset]['name'] + str(iset) + ' shape:' + str(train_dataset[iset]['data'].shape))

## Testing datasets
test_dataset = []
for dname in ['ecs-tst-normalized']:
	test_dataset.append({})
	test_dataset[-1]['name'] = dname
	test_dataset[-1]['data'] =  np.array(h5py.File(join(path,dname,'im_uint8.h5'),'r')['main'],dtype=np.float32)/(2.**8)

for iset in range(len(test_dataset)):
    test_dataset[iset]['data'] = test_dataset[iset]['data'][None,:]
    print(test_dataset[iset]['name'] + str(iset) + ' shape:' + str(test_dataset[iset]['data'].shape))

    
# Set train options
class TrainOptions:
    loss_function = "euclid"
    loss_output_file = "log/loss.log"
    test_output_file = "log/test.log"
    test_interval = 2000
    scale_error = 3 #True
    training_method = "affinity"
    recompute_affinity = True
    train_device = 0
    test_device = 0
    test_net='net_test.prototxt'
    max_iter = int(1.0e4)
    #snapshot = int(2000)
    #loss_snapshot = int(2000)
    snapshot = int(100)
    loss_snapshot = int(100)
    snapshot_prefix = 'net'


options = TrainOptions()

# Set solver options
print('Initializing solver...')
solver_config = pygt.caffe.SolverParameter()
solver_config.train_net = 'net_train_euclid.prototxt'

#solver_config.base_lr = 1e-3
#solver_config.momentum = 0.99
#solver_config.weight_decay = 0.000005
#solver_config.lr_policy = 'inv'
#solver_config.gamma = 0.0001
#solver_config.power = 0.75

solver_config.type = 'Adam'
solver_config.base_lr = 1e-4
solver_config.momentum = 0.99
solver_config.momentum2 = 0.999
solver_config.delta = 1e-8
solver_config.weight_decay = 0.000005
solver_config.lr_policy = 'inv'
solver_config.gamma = 0.0001
solver_config.power = 0.75

solver_config.max_iter = options.max_iter
solver_config.snapshot = options.snapshot
solver_config.snapshot_prefix = options.snapshot_prefix
solver_config.display = 1

# Set devices
print('Setting devices...')
pygt.caffe.enumerate_devices(False)
# pygt.caffe.set_devices((options.train_device, options.test_device))
pygt.caffe.set_devices(tuple(set((options.train_device, options.test_device))))

#-----------------------
# EUCLID
#-----------------------
# First training method

print('===>creating solver....', solver_config.snapshot_prefix)
solverstates = pygt.getSolverStates(solver_config.snapshot_prefix);
print('===>done creating solver....', solver_config.snapshot_prefix)
if True and (len(solverstates) == 0 or solverstates[-1][0] < solver_config.max_iter):
    print('=====> initializing solver....')
    solver, test_net = pygt.init_solver(solver_config, options)
    print('=====> finish initializing solver....')
    print('===>#states:', len(solverstates))
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])

    pygt.train(solver, test_net, train_dataset, test_dataset, options)  
    print('done training method one')

# Second training method
solverstates = pygt.getSolverStates(solver_config.snapshot_prefix);

#-----------------------
# MALIS
#-----------------------

print ('starting MALIS training...')
if True or len(solverstates) == 0 or (solverstates[-1][0] >= solver_config.max_iter):
    # Modify some solver options
    solver_config.max_iter = int(3e5)
    solver_config.train_net = 'net_train_malis.prototxt'
    options.loss_function = 'malis'
    # Initialize and restore solver
    solver, test_net = pygt.init_solver(solver_config, options)
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])
    pygt.train(solver, test_net, train_dataset, test_dataset, options)
