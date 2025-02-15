import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt

class image_prep():

    def hms_string(sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)

    def preprocess_image(generate_res = 1, channels = 3,   data_path='..\data_images',buffer_size=60000,batch_size=32):
        gen_square = 32 * generate_res
        print(f'Generating {gen_square}px square images')
        training_binary_path = os.path.join(data_path,f'training_data_{gen_square}_{gen_square}.npy')
        if not os.path.isfile(training_binary_path):
            start = time.time()
            print("Loading training images...")
            training_data = []
            images_path = os.path.join(data_path,'jpegs')
            for filename in tqdm(os.listdir(images_path)):
                path = os.path.join(images_path,filename)
                image = Image.open(path)
                width, height = image.size
                image = image.crop((height/1.363,0,height/1.363,0))
                image = image.resize((gen_square,gen_square),Image.ANTIALIAS)
                training_data.append(np.asarray(image))
            training_data = np.reshape(training_data,(-1,gen_square,gen_square,channels))
            training_data = training_data.astype(np.float32)
            training_data = training_data / 127.5 - 1.
            
            print("Saving training image binary...")
            np.save(training_binary_path,training_data)
            elapsed = time.time()-start
            print (f'Image preprocess time: {image_prep.hms_string(elapsed)}')
        else:
            print("Loading previous training data...")
            training_data = np.load(training_binary_path)
        return train_dataset 

if (__name__)==('__main__'):
    image_prep.preprocess_image()