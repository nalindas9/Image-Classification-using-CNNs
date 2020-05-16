"""
Dog - Cat Classification using CNN

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras import optimizers

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True
      
# Function to train the CNN on the dataset
def train(train_generator, validation_generator):
  callback = myCallback()
  try:
    # Defining the VGG-16 Architecture CNN model here
    model = Sequential()
 
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3),  padding="same"))
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3),  padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(128, (3,3), activation='relu',  padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(256, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(256, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(256, (3,3), activation='relu',  padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(512, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(512, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(512, (3,3), activation='relu',  padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(512, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(512, (3,3), activation='relu',  padding="same"))
    model.add(Conv2D(512, (3,3), activation='relu',  padding="same"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    
    lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

    opt = optimizers.SGD(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy', 'categorical_crossentropy'], optimizer=opt)
    model.summary()
    
    history = model.fit(train_generator, epochs=50, steps_per_epoch=781, validation_data = validation_generator, verbose = 1, callbacks=[callback])
    
    acc = history.history['accuracy']
    loss = history.history['categorical_crossentropy']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.title('Training accuracy vs Epochs')
    plt.legend(loc=0)
    plt.show()
    plt.plot(epochs, loss, 'r', label='Loss')
    plt.title('Loss vs Epochs')
    plt.legend(loc=0)
    plt.show()
    
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
 	  batch_size=32
  )

  # Preparing testing dataset    
  VALIDATION_DIR = "/home/nalindas9/Documents/courses/spring_2020/enpm673-perception/datasets/test1"  
  validation_datagen = ImageDataGenerator(rescale = 1./255)

  validation_generator = validation_datagen.flow_from_directory(
	  VALIDATION_DIR,
	  target_size=(150,150),
	  class_mode='categorical',
	  batch_size=32  
  )
  
  train(train_generator, validation_generator)
if __name__ == '__main__':
  main()
