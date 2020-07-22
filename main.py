from data_cleaning import dataCleaning as clean
from data_download import dataDownload as download
import urllib3
import numpy
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt
from PIL import Image
import io
from train import train
from gan import dcgan
from image_prep import image_prep


data_path='.\data\oracle-cards.json'
image_path='.\data\images.json'
img_path='..\data_images'

download.update(data_path,'bulk')
print('data updated')

download.images(image_path,'img')
print('images downloaded')

clean_path=clean.data_clean(data_path)

print('data cleaned')

data=pd.read_json(clean_path)

print ('data in dataframe')
generate_res=2
channels=3
seed_size=100
epochs=10000
train_dataset=image_prep.preprocess_image()
generator = dcgan.build_generator(seed_size, channels)

noise = tf.random.normal([1, seed_size])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])
gen_square = 32 * generate_res
image_shape = (gen_square,gen_square,channels)

discriminator = dcgan.build_discriminator(image_shape)
decision = discriminator(generated_image)
print (decision)
train(train_dataset, epochs)
generator.save(os.path.join(img_path,"face_generator.h5"))
