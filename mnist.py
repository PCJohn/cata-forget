# Module to load the MNIST dataset from file
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import os
import cv2
import random
import numpy as np

path = '/home/prithvi/dsets/MNIST/trainingSet/'
train_count = 3000
val_count = 1000

channel_first = True
col = False

def load(shuffle=True,train_count=3000,val_count=1000,channel_first=True,col=False,flatten=False):
    ds = []
    vds = []
    classes = os.listdir(path)
    unit = np.diag(np.ones(len(classes)))
    for n in os.listdir(path):
        n_path = os.path.join(path,n)
        lab = unit[int(n)]
        flist = os.listdir(n_path)
        random.shuffle(flist)
        for s in flist[:train_count]:
            if col == True:
                img = cv2.imread(os.path.join(n_path,s))
            else:
                img = cv2.imread(os.path.join(n_path,s),0)
                img = img[...,np.newaxis]
            if channel_first == True:
                img = img.transpose(2,1,0)
            img = np.float32(img)/255.
            if flatten == True:
                img = img.flatten()
            ds.append((img,lab))
        for s in flist[train_count:train_count+val_count]:
            if col == True:
                img = cv2.imread(os.path.join(n_path,s))
            else:
                img = cv2.imread(os.path.join(n_path,s),0)
                img = img[...,np.newaxis]
            if channel_first == True:
                img = img.transpose(2,1,0)
            img = np.float32(img)/255.
            if flatten == True:
                img = img.flatten()
            vds.append((img,lab))
    if shuffle == True:
        random.shuffle(ds)
        random.shuffle(vds)
    return (ds,vds)
