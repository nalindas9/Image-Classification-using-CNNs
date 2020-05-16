"""
Dog - Cat Classification using CNN

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True
      
# Function to train the CNN on the dataset
def train(train_generator, validation_generator):
  callback = myCallback()
  try:
    # Defining the CNN model here
    model = keras.Sequential([
      # First Convolution 
      keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
      keras.layers.MaxPooling2D(2,2),
      # Second Convolution
      keras.layers.Conv2D(64, (3,3), activation='relu'),
      keras.layers.MaxPooling2D(2,2),
      # Third Convolution 
      keras.layers.Conv2D(128, (3,3), activation='relu'),
      keras.layers.MaxPooling2D(2,2),
      # Fourth Convolution
      keras.layers.Conv2D(128, (3,3), activation='relu'),
      keras.layers.MaxPooling2D(2,2),
      # Flatten the convoluted data
      keras.layers.Flatten(),
      keras.layers.Dropout(0.5),
      # Hidden dense layer
      keras.layers.Dense(512, activation = 'relu'),
      keras.layers.Dense(2, activation = 'softmax')
    ])
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    model.summary()
    
    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, callbacks=[callback])
  finally:
    model.save("trainedCNN_weights.h5")
    
def main():
  # Prepare dataset for target shape
  TRAINING_DIR = "/home/nalindas9/Documents/courses/spring_2020/enpm673-perception/datasets/train_data"
  training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

  train_generator = training_datagen.flow_from_directory(
	  TRAINING_DIR,
	  target_size=(150,150),
	  class_mode='categorical',
    batch_size=126
  )

  # Preparing testing dataset    
  VALIDATION_DIR = "/home/nalindas9/Documents/courses/spring_2020/enpm673-perception/datasets/test1"  
  validation_datagen = ImageDataGenerator(rescale = 1./255)

  validation_generator = validation_datagen.flow_from_directory(
	  VALIDATION_DIR,
	  target_size=(150,150),
	  class_mode='categorical',
    batch_size=126
  )
  
  train(train_generator, validation_generator)
if __name__ == '__main__':
  main()
