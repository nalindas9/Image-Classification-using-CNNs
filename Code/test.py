"""
Dog - Cat Classification using CNN - testing

Authors:
Nalin Das (nalindas9@gmail.com)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

from keras.models import load_model
import keras_preprocessing
from keras_preprocessing import image
import glob
import numpy as np
import os
import csv

def append_list_as_row(file_name, list_of_elem):
  # Open file in append mode
  with open(file_name, 'a+', newline='') as write_obj:
    # Create a writer object from csv module
    csv_writer = csv.writer(write_obj)
    # Add contents of list as last row in the csv file
    csv_writer.writerow(list_of_elem)
        
def test():
  model = load_model('trainedCNN_weights.h5',  compile=False)
  fields = ['id', 'label'] 
  csvname = "cnn_test.csv"
  with open(csvname, 'a') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
        
  for filename in sorted(glob.glob('/home/nalindas9/Documents/courses/spring_2020/enpm673-perception/datasets/test1/test1/*.jpg')):
    frame = filename.split("test1/")
    img = image.load_img(filename, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])/255
    prediction = model.predict(images)
        
    if prediction[0][0] >= 0.5:
      print('Image:', frame[-1], 'Prediction:', 0, 'Its a cat!')
      rows = [[frame[-1][:-4], 0]] 
      # writing to csv file 
      with open(csvname, 'a') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
          
        # writing the data rows 
        csvwriter.writerows(rows)
    else:
      print('Image:', frame[-1], 'Prediction:', 1, 'Its a dog!')
      rows = [[frame[-1][:-4], 1]] 
      with open(csvname, 'a') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the data rows 
        csvwriter.writerows(rows)
def main():
  test()
if __name__ == '__main__':
  main()
