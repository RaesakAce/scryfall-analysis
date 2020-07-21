from data_cleaning import dataCleaning as clean
from data_download import dataDownload as download
import urllib3
import numpy
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt
from PIL import Image
import io


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

#devo lavorare su questa roba perchè non funziona
imgs=pd.read_json(image_path)
keys=['art_crop','normal','large','png','small','border_crop']
val=keys[0]
imgs[val] = pd.Series([imgs['image_uris'][i][val] for i in imgs['image_uris'].index])

print(imgs['art_crop'][74])
print(imgs['image_uris'][74]['art_crop'])
print ('data in dataframe')

sb.relplot(x='edhrec_rank',y='eur',hue='reserved',size='cmc',size_norm=(0,10),data=data)
plt.show()