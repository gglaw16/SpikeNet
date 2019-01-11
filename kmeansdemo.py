# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 00:04:18 2018

@author: gwenda
"""
#import matplotlib.pyplot as plt
import numpy as np
import math
import random
import cv2
import pdb
from sklearn.cluster import KMeans





def load_mnist():
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/t10k-labels-idx1-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        test_labels = np.fromfile(fp, dtype=np.uint8, count=num)
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/train-labels-idx1-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        train_labels = np.fromfile(fp, dtype=np.uint8, count=num)
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/t10k-images-idx3-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        rows = fp.read(4)
        rows = int(rows.encode('hex'), 16)
        cols = fp.read(4)
        cols = int(cols.encode('hex'), 16)
        test_images = np.fromfile(fp, dtype=np.uint8, count=num*rows*cols)
        test_images = test_images.reshape(num, rows, cols)
    with open('/home/claw/Preston/cs574/p3_mnist/mnist/train-images-idx3-ubyte', 'rb') as fp:
        magic = fp.read(4)
        num = fp.read(4)
        num = int(num.encode('hex'), 16)
        rows = fp.read(4)
        rows = int(rows.encode('hex'), 16)
        cols = fp.read(4)
        cols = int(cols.encode('hex'), 16)
        train_images = np.fromfile(fp, dtype=np.uint8, count=num*rows*cols)
        train_images = train_images.reshape(num, rows, cols)
    #cv2.imwrite('images/mnist.png', test_images[0, ...])
    return train_images, train_labels, test_images, test_labels


kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
images = np.array(load_mnist()[0])
images = images.reshape(60000,784)
print(images.shape)
kmeans.fit(images)
centroids = kmeans.cluster_centers_
out = centroids.reshape(10,28,28)
for i in range(10): cv2.imwrite("images/image%d.png"%i,out[i])



