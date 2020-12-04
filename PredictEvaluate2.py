# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:14:47 2019

@author: an_fab
"""

import os
import pickle

import time

import numpy as np
import configparser
from skimage import io

from keras.models import model_from_json
from helpers import plot_confusion_matrix, imagePreprocessing

#-----------------------------------------------------------------------------
#read config file

config = configparser.RawConfigParser()
config.read('configuration.txt')

num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))

test_patches = config.get('data paths','test_patches')
test_labels = config.get('data paths','test_labels')
best_weights = config.get('data paths','best_weights')

pred_dir = config.get('data paths', 'preds_dir')
test_dir = config.get('data paths', 'test_dir')

conf_mat_images_saved = config.get('data paths','conf_mat_images_saved')
#-----------------------------------------------------------------------------    
#load CNN model

model = model_from_json(open('model.json').read())
model.load_weights(best_weights)

#-----------------------------------------------------------------------------   
# predict image patch by patch
    
def predictImage(img, patchSize,numClasses):
    
    stride = 5
    
    model = model_from_json(open('model.json').read())
    model.load_weights(best_weights)
    
    [rows, cols, nch] = img.shape
        
    out = np.zeros((rows,cols), dtype = 'uint8')
    
    num = (1 + (rows - patchSize)  / stride) * (1 + (cols - patchSize)  / stride)  
    num = int(num+1)
    
    X_test = np.zeros((num, patchSize, patchSize, nch))
    probs = np.zeros((numClasses,1))
    
    y = 0
    patchId = 0;
    
    while y+patchSize < rows:
       
        x = 0
    
        while x+patchSize < cols:
                        
            patch = img[y:y+patchSize,x:x+patchSize]
            X_test[patchId] = patch
            patchId += 1
            x = x + stride

        y = y + stride
    
    #print('--- patches: %d ---' % X_test.shape[0])
    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1) 
            
    y = 0
    patchId = 0;
     
    while y+patchSize < rows:
       
        x = 0
    
        while x+patchSize < cols:
            
            x1 = int(x+patchSize/2)
            x2 = int(x+patchSize/2) + stride

            y1 = int(y+patchSize/2)
            y2 = int(y+patchSize/2) + stride
               
            out[y1:y2,x1:x2] = 10*y_pred[patchId]
            probs[y_pred[patchId]] +=1
            patchId += 1
            x = x + stride

        y = y + stride    
    
    probs = probs/patchId
    fin = np.argmax(probs) 
    
    return out, fin

#-----------------------------------------------------------------------------  
# predict files in test directory
    
dict_labels = pickle.load(open('dictLabels.p', "rb"))
dict_names = pickle.load(open('dictNames.p', "rb"))

confMat = np.zeros((num_classes,num_classes))

for dirname, dirnames, files in os.walk(test_dir):
    
    for subdir in dirnames:
        
        print(subdir)
            
        classId = dict_names[subdir]
            
        f = os.listdir(os.path.join(dirname, subdir))
        
        for file in f:
            
            img = io.imread(os.path.join(dirname,subdir, file))
            img = img[:,:,0:3]
            print(os.path.join(dirname,subdir, file))
            
            img = imagePreprocessing(img)
            
            start_time = time.time()
            pred, label = predictImage(img, patch_size, num_classes)
            #print("--- %s seconds ---" % (time.time() - start_time))
            #print("--- shape %d x %d pixels ---" % (pred.shape[0], pred.shape[1]))
            
            #print(pred.shape[0], pred.shape[1], (time.time() - start_time))
            
            path = os.path.join(pred_dir,subdir, file)
            io.imsave(path, pred)
            
            confMat[classId,label] +=1
            
            if classId != label:
                print('!!!!!!!!!!!!!')
                print(os.path.join(dirname,subdir, file))
                print(label)
                print('!!!!!!!!!!!!!')
                

names = list()
for i in range(0, num_classes):
    names.append(dict_labels[i])
    
plot_confusion_matrix(confMat, normalize=False, target_names = names)
plot_confusion_matrix(confMat, normalize=True, target_names = names)
pickle.dump(confMat, open(conf_mat_images_saved, "wb"))