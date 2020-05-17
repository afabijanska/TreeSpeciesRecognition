# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 01:02:12 2019

@author: an_fab
"""

import math
import pickle
import numpy as np
import configparser

import random
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from keras import layers

from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras import regularizers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, LocallyConnected2D, LeakyReLU

from helpers import load_hdf5, mirrorImage

#-----------------------------------------------------------------------------
# Exponential Decay of learning rate

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * math.exp(-k*epoch)
   return lrate

#-----------------------------------------------------------------------------
# step decay of learning rate

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.75
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

#-----------------------------------------------------------------------------
# define focal loss

def focal_loss(gamma=2., alpha=.25):
    
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

#-----------------------------------------------------------------------------
# define old (existing) model
    
def getCompetitiveModel(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=64, kernel_size=(5,5), strides = (1,1), activation='relu', padding='same')(inputs)
    pool1 = MaxPool2D(pool_size=(3,3), strides = (2,2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(5,5), strides = (1,1), activation='relu', padding='same')(pool1)
    pool2 = MaxPool2D(pool_size=(3,3), strides = (2,2))(conv2)
    conn3 = LocallyConnected2D(filters=32,kernel_size=(3,3))(pool2)
    conn4 = LocallyConnected2D(filters=32,kernel_size=(3,3))(conn3)
    flat5 = Flatten()(conn4) 
    dens5 = Dense(numClasses, activation = 'softmax')(flat5)
    
    model = Model(inputs=inputs, outputs=dens5)
    model.compile(optimizer=Adam(lr = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    model.summary()
    return model
#-----------------------------------------------------------------------------
# define model
    
def getSampleModel(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same', use_bias=False)(inputs)
    #conv1 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv1)
    #conv1 = Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv1)
    #bn1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(act1)
    #drop1 = Dropout(0.4)(pool1)
   
    conv2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', use_bias=False)(pool1)
    #conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv2)
    #conv2 = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv2)
    #bn2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPool2D(pool_size=(2,2))(act2)
    #drop2 = Dropout(0.4)(pool2)
   
    conv3 = Conv2D(filters=64, kernel_size=(3,3), padding='same', use_bias=False)(pool2)
    #conv3 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv3)
    #conv3 = Conv2D(filters=128, kernel_size=(5,5), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv3)
    #bn3 = BatchNormalization()(conv3)
    act3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPool2D(pool_size=(2,2))(act3)
    #drop3 = Dropout(0.4)(pool3)
    
    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same', use_bias=False)(pool3)
    #conv4 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv4)
    #conv4 = Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding='same', kernel_regularizer = regularizers.l2(l = 0.001))(conv4)
    #bn4 = BatchNormalization()(conv4)
    act4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPool2D(pool_size=(2,2))(act4)
    #drop4 = Dropout(0.4)(pool4)   
    
    flat1 = Flatten()(pool4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(lr = 0.001, decay = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr = 0.01), loss=focal_loss(gamma=2., alpha=.50), metrics = ['accuracy'])
    model.summary()
    
    return model

#-----------------------------------------------------------------------------

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', use_bias=False)(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(y)
    #y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', use_bias=False)(shortcut)
        #shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

#-----------------------------------------------------------------------------
# define residual model
    
def getSampleResidualModel(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(act1)

    res1 = residual_block(pool1, 64) ;  
    pool2 = MaxPool2D(pool_size=(2,2))(res1)
    
    res2 = residual_block(pool2, 64)
    pool3 = MaxPool2D(pool_size=(2,2))(res2)
    
    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(pool3)

    act4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPool2D(pool_size=(2,2))(act4)
    
    flat1 = Flatten()(pool4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(lr = 0.001, decay = 0),loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr = 0.01), loss=focal_loss(gamma=2., alpha=.50), metrics = ['accuracy'])
    model.summary()
    
    return model

#-----------------------------------------------------------------------------
# define residual model
    
def getSampleResidualModel2(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    res1 = residual_block(act1, 64);  
    pool1 = MaxPool2D(pool_size=(2,2))(res1)
    
    res2 = residual_block(pool1, 64);  
    pool2 = MaxPool2D(pool_size=(2,2))(res2)
        
    res3 = residual_block(pool2, 64)
    pool3 = MaxPool2D(pool_size=(2,2))(res3)
    
    res4 = residual_block(pool3, 64)
    pool4 = MaxPool2D(pool_size=(2,2))(res4)
    
    flat1 = Flatten()(pool4)
    dens1 = Dense(256, activation='relu')(flat1)
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(lr = 0.001, decay = 0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr = 0.01), loss=focal_loss(gamma=2., alpha=.50), metrics = ['accuracy'])
    model.summary()
    
    return model

#-----------------------------------------------------------------------------
#read config file

config = configparser.RawConfigParser()
config.read('configuration.txt')

num_epochs = int(config.get('training settings', 'num_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

num_classes = int(config.get('data attributes','num_classes'))
patch_size = int(config.get('data attributes','patch_size'))

train_patches = config.get('data paths','train_patches')
train_labels = config.get('data paths','train_labels')
best_weights = config.get('data paths','best_weights')

history_saved = config.get('data paths','history_saved')

augment = config.get('training settings', 'augment')

#-----------------------------------------------------------------------------
#load and transform data

X_train = load_hdf5(train_patches)
Y_train = load_hdf5(train_labels)
Y_train = to_categorical(Y_train)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

#get model and train it
#model = getSampleModel(num_classes, (patch_size, patch_size, 3))
model = getSampleResidualModel2(num_classes, (patch_size, patch_size, 3))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
checkpointer = ModelCheckpoint(best_weights, verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased
patienceCallBack = EarlyStopping(monitor='val_loss',patience=100)
learningRateCallBack = LearningRateScheduler(exp_decay ,verbose = 1)
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)

batch_size = int(batch_size/2)


if augment == 'True':
    
    print('Train with augmentation')
    
    datagen = ImageDataGenerator(
                            rotation_range=10,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')
    
    validation_split = 0.2
    indx = list(range(0,len(X_train)))
    random.shuffle(indx)
    random.shuffle(indx)

    k = int(0.2*len(X_train))

    X_val = X_train[indx[0:k]]
    Y_val = Y_train[indx[0:k]]

    X_train = X_train[indx[k:len(X_train)]]
    Y_train = Y_train[indx[k:len(Y_train)]] 

    history = model.fit_generator(datagen.flow(x = X_train, y = Y_train, batch_size = batch_size),
                                  validation_data = datagen.flow(x = X_val, y = Y_val, batch_size = batch_size),
                                  steps_per_epoch = len(X_train)/batch_size,
                                  epochs = num_epochs,
                                  callbacks = [checkpointer,tbCallBack,patienceCallBack])                  
else:
    
    print('No augmentation')
    
    history = model.fit(x=X_train, 
                    y=Y_train, 
                    validation_split=0.2, 
                    epochs = num_epochs, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    callbacks = [checkpointer,tbCallBack,patienceCallBack])

#-----------------------------------------------------------------------------
#plot train history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

pickle.dump(history, open(history_saved, "wb"))