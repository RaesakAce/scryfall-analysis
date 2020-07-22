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
img_path='..\data_images\jpegs'

download.update(data_path,'bulk')
print('data updated')

download.images(image_path,'img')
print('images downloaded')

clean_path=clean.data_clean(data_path)

print('data cleaned')

data=pd.read_json(clean_path)

print ('data in dataframe')

