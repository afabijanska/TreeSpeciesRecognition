# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:24:56 2019

@author: an_fab
"""

import os
import h5py
import random
import pickle
import configparser
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from keras.datasets import cifar10

from helpers import  write_hdf5, mirrorImage, imagePreprocessing

#----------------------------------------------------------------------------
#read config file

config = configparser.RawConfigParser()
config.read('configuration.txt')

total_folds_no = int(config.get('general settings', 'total_folds_no'))
num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))
patches_per_species = int(config.get('data attributes','patches_per_species'))

test_fold_id = int(config.get('testing settings','test_fold_id'))

data_dir = config.get('data paths','data_dir')
main_dir = os.path.join(data_dir, 'org')

train_patches = config.get('data paths','train_patches')
train_labels = config.get('data paths','train_labels')
test_patches = config.get('data paths','test_patches')
test_labels = config.get('data paths','test_labels')

dictLabels = pickle.load(open('dictLabels.p', "rb"))
dictNames = pickle.load(open('dictNames.p', "rb"))

#----------------------------------------------------------------------------
# function: createDataFromDir

def createDataFromDir(dirPath, numClasses, samplesPerClass, patchSize):
        
    X_data = np.zeros((numClasses*samplesPerClass,patchSize,patchSize,3))
    Y_data = np.zeros((numClasses*samplesPerClass,1))
    
    totalNumSamples = 0
        
    for dirname, dirnames, files in os.walk(dirPath):
        
        for subdir in dirnames:
            
            print(subdir)
            
            f = os.listdir(os.path.join(dirname, subdir))
            
            numImages = len(f)
            samplesPerImage = int(samplesPerClass/numImages)
            samplesPerImage +=1
            samplesPerClassCounter = 0
            
            for file in f:
                
                img = io.imread(os.path.join(dirname,subdir, file))
                img = img[:,:,0:3]
                img = imagePreprocessing(img)
                           
                print(os.path.join(dirname,subdir, file))
                [rows, cols, nch] = img.shape
                
                if cols < patchSize:
                    img = mirrorImage(img)
                    cols = cols * 2
                
                samples = 0
                
                while samples < samplesPerImage:
                    
                    x = random.randint(0, cols - patchSize)
                    y = random.randint(0, rows - patchSize)
                    patch = img[y:y+patchSize,x:x+patchSize]
                        
                    X_data[totalNumSamples] = patch
                    Y_data[totalNumSamples] = dictNames[subdir]
                    samples+=1
                    totalNumSamples+=1
                    samplesPerClassCounter+=1
                
                    if samplesPerClassCounter >= samplesPerClass:
                        break
                
    return X_data, Y_data

#----------------------------------------------------------------------------
# get train patches from train folds 
# get test patches from test folds
    
  
samples_per_fold = int(patches_per_species / (total_folds_no - 1))
   
X_train = np.zeros((num_classes*patches_per_species, patch_size, patch_size, 3))
Y_train = np.zeros((num_classes*patches_per_species,1))

k = 0

for i in range(1, total_folds_no + 1):
    
    dir_path = os.path.join(data_dir,'fold_'+ str(i))
    
    if i != test_fold_id:
        
        x_train, y_train = createDataFromDir(dir_path, num_classes, samples_per_fold, patch_size)
        
        X_train[k:k+num_classes*samples_per_fold] = x_train
        Y_train[k:k+num_classes*samples_per_fold] = y_train
        k = k + num_classes*samples_per_fold 
    
    else:
        
        X_test, Y_test = createDataFromDir(dir_path, num_classes, samples_per_fold, patch_size)
                
#----------------------------------------------------------------------------
#save train data
print ("saving train patches")
write_hdf5(X_train, train_patches)
write_hdf5(Y_train, train_labels)

#save test data
print ("saving test patches")
write_hdf5(X_test, test_patches)
write_hdf5(Y_test, test_labels)

#----------------------------------------------------------------------------
# display sample patches

for i in range (1,10):
    plt.subplot(3,3,i)
    idx = random.randint(0,num_classes*patches_per_species)
    plt.tight_layout()
    plt.imshow(X_train[idx])
    plt.title(('{%d: %s}')%(int(Y_train[idx]), dictLabels[int(Y_train[idx])]))
    plt.xticks([])
    plt.yticks([])
plt.show()