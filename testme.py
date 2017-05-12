import scipy.misc
import string
import skimage
import json
import sys, os, math
import h5py
import numpy as np
import numpy.random
from scipy.misc import imread
from skimage import color
from skimage import io



path = '500-distal-pred.h5'
name = 'main'
volume = np.array(h5py.File( path ,'r')[name],dtype=np.float32)/(2.**8)
image = volume[0,:,:]*255
print np.min(image), np.max(image)

