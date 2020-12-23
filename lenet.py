import os, sys

import numpy as np
from SpatialPyramidPooling import SpatialPyramidPooling

from sys import argv, exit

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import to_categorical

from util import *
import tensorflow as tf


class LeNet:
    model = None

    #@staticmethod
    def __init__(self, numChannels, imgRows, imgCols, numClasses, lr, activation='relu', weightsPath=None):

        self.model = Sequential()
        input_shape = (imgRows, imgCols, numChannels)

        if (K.image_data_format() == 'channels_first'):
            input_shape = (numChannels, imgRows, imgCols)

        self.model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding='same'))
        self.model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        self.model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.model.add(SpatialPyramidPooling([1, 2, 4]))
        #self.model.add(Flatten())
        self.model.add(Dense(84, activation='tanh'))
        #self.model.add(Dense(84, activation='softmax'))
        self.model.add(Dense(numClasses, activation='softmax'))
        self.model.add(Activation('softmax'))

        if (not weightsPath == None):
            self.model.load_weights(weightsPath)

        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = SGD(lr=lr),
            metrics = ['accuracy'])

def read_images(img_path,apps,train_dim,test_dim):
    import os
    import cv2

    X_train = []
    X_test  = []
    y_train = []
    y_test  = []

    max = 999
    for app in apps:
        path = img_path + app + '_' + train_dim + '/'
        f = os.listdir(path)
        if (len(f) < max): max = len(f)
    max = 200

    for app in apps:
        path = img_path + app + '_' + train_dim + '/'
        for d, r, f in os.walk(path):
            for file in f[:max]:
                if (not file.endswith('.png')): continue
                img = cv2.imread(path+file,cv2.IMREAD_GRAYSCALE)
                if (not img is None):
                    X_train.append(img)
                    y_train.append(apps.index(app))

        path = img_path + app + '_' + test_dim + '/'
        for d, r, f in os.walk(path):
            for file in f[:100]:
                if (not file.endswith('.png')): continue
                img = cv2.imread(path+file,cv2.IMREAD_GRAYSCALE)
                if (not img is None):
                    X_test.append(img)
                    y_test.append(apps.index(app))

    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    #y_train = np.asarray(y_train)
    #y_test  = np.asarray(y_test)

    y_train = to_categorical(np.asarray(y_train))
    y_test  = to_categorical(np.asarray(y_test))

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

    return X_train, X_test, y_train, y_test

# 1. Main function
def main():
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []

    train_dim = get_definitions("Dataset","train_dim")
    test_dim  = get_definitions("Dataset","test_dim")

    # 1.2 Get the dataset path
    img_path = get_definitions("Dataset","path")
    apps = get_applications()
    # 1.3 Parse all class lines provided in definitions file
    X_train, X_test, y_train, y_test = read_images(img_path,apps,train_dim,test_dim)
    number_of_train_classes = len(apps)
    lr = float(get_definitions("NNParams","lr"))
    epochs = int(get_definitions("NNParams","epochs"))

    #lenet = LeNet(1, int(train_dim[0]), int(train_dim[0]), number_of_train_classes)
    lenet = LeNet(1, None, None, number_of_train_classes, lr)
    print(lenet.model.summary())

    lenet.model.fit(
        X_train,
        y_train,
        batch_size = 28,
        epochs = epochs,
        verbose = 1)

    (loss, accuracy) = lenet.model.evaluate(
        X_test,
        y_test,
        batch_size = 28,
        verbose = 1)

    dir_results = get_definitions("Results","path")
    file_results = get_definitions("Results","file")

    if (not os.path.exists(dir_results)):
        os.mkdir(dir_results)

    print(accuracy)
    with open(dir_results+file_results,'a') as f:
        f.write('Tr: ' + train_dim + ' Ts: ' + test_dim + ' Ep: ' + str(epochs) +'\n')
        str_apps = 'Apps: '
        for app in apps:
            str_apps += app + ' '
        f.write(str_apps + '\n')
        f.write('Acc: ' + str(accuracy) + '\n\n')
    f.close()

if (__name__ == '__main__'):
    main()
