#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: srinivas
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 is number of feature maps we are creating
# (3,3) is the  no of rows and columns of fearure detector
#(64,64,3) -> (64,64) is dimentioin of 2d array and 3 is because of colour image RGB
#activation ='relu'  is used to have non linearity in the model
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
#Pooling is done to reduce the size of feature maps and reduce complexity of the model
#maxpooiling is used to extract the high value pixels
#(2,2) is the size of pooling window
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
#Flattening is done to create a huge single vector
classifier.add(Flatten())


# Step 4 - Full connection
#units is number of outputs in the hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))

#Output layer
# Adding the output layer
#sigmoid activation functions gives binary output
#if here are more than two categories then use activation = 'softmax'
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
#optimizer is adam as it is one of the stochastic gradient decent optimizer
"""loss = 'binary_crossentropy' it is used because output is the binary classifiaction output. if the output has more that
two classification then use loss = 'categorical_crossentropy'  """
#metrics = ['accuracy'] used to evaluate a model and accordingly update the weights to get the more accuaracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
#rescale = 1./255 is udsed to bring alll the images in the same format of pixels of pixel values between 0 and 1
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#target size is (64,64) because in the convolution layer input size is (64,64)
#batch_size = 32 means after passing of 32 images through the model weights will be updated
#class_mode is binary as we have cats and dogs as the class 
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#steps_per_epoch = 8000 is number of images in the training set
#validation_steps = 2000 is the number of images in the test set
#epochs = 25 is 25 times the model will be fitted 
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

#Accuracyof the test set was 75%

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image) # To convert the dimentions from (64,64) to (64,64,3) because it is the color image
test_image = np.expand_dims(test_image, axis = 0) #CNN always expect input in the form of batch. This extra dimention is for the batch
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'