# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:00:38 2020

@author: aaron
"""
#Loading in the directories
# Directories for training, validation, and test sets
train_dir = './train'
valid_dir ='./validation'
test_dir = './test'

import os

# Directory with the training cat and dog pictures
train_grass_dir = os.path.join(train_dir, 'Grass')
train_water_dir = os.path.join(train_dir, 'Water')


# Directory with the validation cat and dog pictures
valid_grass_dir = os.path.join(valid_dir, 'Grass')
valid_water_dir = os.path.join(valid_dir, 'Water')

# Directory with the test cat and dog pictures
test_grass_dir = os.path.join(test_dir, 'Grass')
test_water_dir = os.path.join(test_dir, 'Water')

print('Total training grass images:', len(os.listdir(train_grass_dir)))
print('Total training water images:', len(os.listdir(train_water_dir)))
print('Total validation grass images:', len(os.listdir(valid_grass_dir)))
print('Total validation water images:', len(os.listdir(valid_water_dir)))
print('Total test grass images:', len(os.listdir(test_grass_dir)))
print('Total test water images:', len(os.listdir(test_water_dir)))

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
print('\n')
print('Preprocess the training set')
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

train_generator = train_datagen.flow_from_directory(
        directory = train_dir,      # The target directory
        target_size=(150, 150),     # Being resized to 150x150
        batch_size=32,
        class_mode='binary',        # Binary classification
        seed = 5
        )

# Each batch has 20 samples, and each sample is an 150x150 RGB image 
# (shape 150,150,3) and binary labels.
print('\n')
print('In the first batch')
(data_batch, labels_batch) = train_generator[0]
print('Data batch shape:', data_batch.shape)
print('Labels batch shape:', labels_batch.shape)

# preprocess the validation set
print('\n')
print('Preprocess the validation set')
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
        directory = valid_dir,      
        target_size=(150, 150),     
        batch_size=20,
        class_mode='binary',        
        seed = 5
        )

# preprocess the test set
print('\n')
print('Preprocess the test set')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        directory = test_dir,      
        target_size=(150, 150),     
        batch_size=1,
        class_mode='binary',  
        shuffle = False,
        seed = 5
        )

# initiate a pre-trained convolutional base VGG16
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
conv_base.summary()

# build the network
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False
model.summary()

# configure the model
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

# train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,     
      epochs=50,
      validation_data=valid_generator,
      validation_steps=50       # 1000/20
      )

model.save('grass_and_water_pretrained.h5')

# plot the training and validation scores
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

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

# evaludate the model
eval_test = model.evaluate_generator(generator=test_generator, steps=44)
print("The test score (accuracy) is {}%".format(eval_test[1]*100))

# output the predictions compared to the targets
import numpy as np
import pandas as pd
test_generator.reset()
pred=model.predict_generator(test_generator, steps=44, verbose=1)
predicted_class_indices= np.where(pred > 0.5, 1, 0)
predicted_class_indices=predicted_class_indices.reshape(-1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results_pretrained.csv",index=False)