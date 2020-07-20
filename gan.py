import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from os.path import join

image_dir = './data/images/'
img_paths = [join(image_dir, filename) for filename in os.listdir()
image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)