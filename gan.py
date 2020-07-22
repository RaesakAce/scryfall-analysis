import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt

class dcgan ():
    def build_generator(seed_size, channels,generate_res=2):
        model = Sequential()

        model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
        model.add(Reshape((4,4,256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
   
        # Output resolution, additional upsampling
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        if generate_res>1:
            model.add(UpSampling2D(size=(generate_res,generate_res)))
            model.add(Conv2D(128,kernel_size=3,padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        # Final CNN layer
        model.add(Conv2D(channels,kernel_size=3,padding="same"))
        model.add(Activation("tanh"))

        return model


    def build_discriminator(image_shape):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, 
                        padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def save_images(cnt,noise, generate_res = 2, rows = 4 , cols = 7, margin = 16):
        gen_square = 32 * generate_res
        image_array = np.full(( 
            margin + (rows * (gen_square+margin)), 
            margin + (cols* (gen_square+margin)), 3), 
            255, dtype=np.uint8)
  
        generated_images = generator.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        image_count = 0
        for row in range(rows):
            for col in range(cols):
                r = row * (gen_square+16) + margin
                c = col * (gen_square+16) + margin
                image_array[r:r+gen_square,c:c+gen_square] = generated_images[image_count] * 255
                image_count += 1

          
        output_path = os.path.join(DATA_PATH,'output')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
  
        filename = os.path.join(output_path,f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)

    def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)  