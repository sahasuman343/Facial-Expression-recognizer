# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:25:19 2020

@author: Suman
"""
# Importing the libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
import numpy as np
tf.__version__

# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('images/train',
                                                 target_size = (48, 48),
                                                 batch_size = 32,
                                                 color_mode='grayscale',
                                                 class_mode = 'categorical')

# Creating the Test set
valid_set = test_datagen.flow_from_directory('images/validation',
                                            target_size = (48,48),
                                            batch_size = 32,
                                            color_mode='grayscale',
                                            class_mode = 'categorical')

# number of possible label values
n = 7
#print(dir(trainig_set))

# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(32,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(64,(5,5), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(128,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(256,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(n, activation='softmax'))


#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model=keras.models.load_model("best_model.h5")
#callbacks

cb1=keras.callbacks.ModelCheckpoint("models/best_model1.h5",save_best_only=True,verbose=1)
cb2=keras.callbacks.EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

#model hostory
history=model.fit_generator(training_set,
                            steps_per_epoch=training_set.n//training_set.batch_size,
                            epochs=10,
                            validation_data=valid_set,
                            validation_steps=valid_set.n//valid_set.batch_size,
                            callbacks=[cb1,cb2]
                            )