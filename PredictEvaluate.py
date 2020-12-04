# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:23:50 2019

@author: an_fab
"""

# evaluate method with respect to sample test patches

import pickle
import random
import numpy as np
import configparser
import matplotlib.pyplot as plt

from helpers import load_hdf5, plot_confusion_matrix
from keras.models import model_from_json

#-----------------------------------------------------------------------------
#read config file

config = configparser.RawConfigParser()
config.read('configuration.txt')

num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))

test_patches = config.get('data paths','test_patches')
test_labels = config.get('data paths','test_labels')
best_weights = config.get('data paths','best_weights')

conf_mat_patches_saved = config.get('data paths','conf_mat_patches_saved')

#-----------------------------------------------------------------------------    
#load CNN model

model = model_from_json(open('model.json').read())
model.load_weights(best_weights)

#-----------------------------------------------------------------------------   
#load test data - patches

X_test = load_hdf5(test_patches)

# for i in range(0,X_test.shape[0]):
#      X_test[i] = rotate(X_test[i],90)

Y_test = load_hdf5(test_labels)

#-----------------------------------------------------------------------------   
#predict 

y_pred = model.predict(X_test, batch_size=200)
y_pred = np.argmax(y_pred, axis = 1) 

#-----------------------------------------------------------------------------   
#assess accuracy

total = 0
okays = 0

for i in range(y_pred.shape[0]):
    total += 1
    if (y_pred[i] == Y_test[i]):
        okays += 1
        
print("total acc: ", 100*okays/total)
print("correct classifications: ", okays)
print("errors: ", total - okays)

#-----------------------------------------------------------------------------   
#plot confusion matrix

confusionMatrix = np.zeros((num_classes, num_classes), dtype= 'int')

for i in range(0,Y_test.shape[0]-1):
    
    confusionMatrix[int(Y_test[i])][y_pred[i]] +=1

dict_labels = pickle.load(open('dictLabels.p', "rb"))

names = list()

for i in range(0, num_classes):
    names.append(dict_labels[i])

plot_confusion_matrix(confusionMatrix, normalize=False, target_names = names)
pickle.dump(confusionMatrix, open(conf_mat_patches_saved, "wb"))