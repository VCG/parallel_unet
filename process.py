from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import time
# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt

# model files
##modelfile = 'result_31_3/net_iter_48000.caffemodel'
##modelproto = 'result_31_3/net_test.prototxt'

#modelfile = '31-3/net_iter_100000.caffemodel'
#modelproto = '31-3/net_test.prototxt'

modelfile = 'net_iter_150000.caffemodel'
modelproto = 'net_test_big.prototxt'

# Load the datasets
path = '/home/paragt/install/examples/ac-data/'
# Test set
test_dataset = []

test_dataset.append({})
dname = 'ecs-ruobin'
test_dataset[-1]['name'] = dname
h5im = h5py.File(join(path,dname,'im_uint8.h5'),'r')
h5im_n = np.asarray(h5im[h5im.keys()[0]]).astype(float32)/255
test_dataset[-1]['data'] = h5im_n
test_dataset[-1]['data'] = test_dataset[-1]['data'][None,:]
#h5im_n = pygt.normalize(np.asarray(h5im[h5im.keys()[0]]).astype(float32), -1, 1)
#test_dataset[-1]['data'] = h5im_n

#test_dataset.append({})
#dname = 'tstvol-520-2-h5'
#test_dataset[-1]['name'] = dname
#h5im = h5py.File(join(path,dname,'img_normalized.h5'),'r')
#h5im_n = pygt.normalize(np.asarray(h5im[h5im.keys()[0]]).astype(float32), -1, 1)
#test_dataset[-1]['data'] = h5im_n



# Set devices
test_device = 0
print('Setting devices...')
pygt.caffe.set_mode_gpu()
pygt.caffe.set_device(test_device)
# pygt.caffe.select_device(test_device, False)

# Load model
print('Loading model...')
net = pygt.caffe.Net(modelproto, modelfile, pygt.caffe.TEST)


start_time = time.time()
# Process
print('Processing ' + str(len(test_dataset)) + ' volume(s)...')
preds = pygt.process(net,test_dataset)
#for i in range(len(test_dataset)):
    #print('Saving ' + test_dataset[i]['name'])
    #h5file = test_dataset[i]['name'] + '.h5'
    #outhdf5 = h5py.File(h5file, 'w')
    #outdset = outhdf5.create_dataset('main', preds[i].shape, np.float32, data=preds[i])
    #outhdf5.close()
  
end_time = time.time()

print("elapsed seconds {0} for z ,y ,x ={1}, {2}, {3}".format(end_time-start_time, preds[0].shape[1],preds[0].shape[2],preds[0].shape[3]))

for i in range(0,len(test_dataset),2):
    print('Saving ' + test_dataset[i]['name'])
    h5file = test_dataset[i]['name'] + '-pred.h5'
    outhdf5 = h5py.File(h5file, 'w')
    outdset = outhdf5.create_dataset('main', preds[i].shape, np.float32, data=preds[i])
    outhdf5.close()
    
    h5file = test_dataset[i]['name'] + '-mask.h5'
    outhdf5 = h5py.File(h5file, 'w')
    outdset = outhdf5.create_dataset('main', preds[i+1].shape, np.uint8 , data=preds[i+1])
    outhdf5.close()
