import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from image_prep import image_prep
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt

class train():
    def optimizer():
        generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)

    @tf.function
    def train_step(images):
        seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(seed, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
    

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return gen_loss,disc_loss

    def train(dataset, epochs):
        fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
        start = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for image_batch in dataset:
                t = train_step(image_batch)
                gen_loss_list.append(t[0])
                disc_loss_list.append(t[1])

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            epoch_elapsed = time.time()-epoch_start
            print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
                ' {image_prep.hms_string(epoch_elapsed)}')
            save_images(epoch,fixed_seed)

        elapsed = time.time()-start
        print (f'Training time: {image_prep.hms_string(elapsed)}')