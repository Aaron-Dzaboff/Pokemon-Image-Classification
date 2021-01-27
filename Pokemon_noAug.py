# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:10:45 2020

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

# Preprocess the images: read the image files; 
# decode the jpg to RBG grids of pixels (150x150); 
# convert to float point tensors; rescale the pixel values

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
print('\n')
print('Preprocess the training set')
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        directory = train_dir,      # The target directory
        target_size=(150, 150),     # Being resized to 150x150
        batch_size=20,
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

# build a CNN
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))  
model.add(layers.MaxPooling2D((2, 2))) # stride 2 (downsampled by a factor of 2)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten()) # Flatten the 3D outputs to 1D before adding a few Dense layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) # binary classificaiton
model.summary()

# configure the model
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,      # 2000/20
      epochs=20,
      validation_data=valid_generator,
      validation_steps=50       # 1000/20
      )

model.save('grass_and_water_noAug.h5')

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
results.to_csv("results-water_grass_noAug.csv",index=False)