import urllib3
import pandas as pd
import os
import time
import io
import numpy
from PIL import Image

class dataDownload():
    def api_request(data_type):
        print('requesting...')
        request_type={'img':1,'bulk':0}
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        r = http.request('GET','https://api.scryfall.com/bulk-data')
        if (r.status == 200):
            print('accessing API')
            uri_data = pd.read_json(r.data)['data'][request_type[data_type]]['download_uri']
            r = http.request('GET',uri_data)
            print('request succesful!')
        else: print('r.status')
        return r
    
    def save_json(r,data_path):
        if (r.status == 200):
            print('saving raw data...')
            pd.read_json(r.data).to_json(data_path)
            print ('raw data saved!')
    
    def update(data_path,data_type):
        if os.path.isfile(data_path):
            last_update=(time.time()-os.stat(data_path).st_mtime)/3600
            if (last_update < 24):
                print(f'The data has been updated {round(last_update,2)} hours ago')
                return data_path
        dataDownload.save_json(dataDownload.api_request(data_type),data_path)
        
    
    def save_images(r,data_path):
        img_path='..\data_images\jpegs'
        imgs=pd.read_json(r.data)
        #Select the columns which I want to keep in the dataFrame
        useful_columns=['image_uris','artist']
        #Copy only the part that I'm interested in of the df

        imgs = imgs[useful_columns]
        img_list=[]
        for i in imgs['image_uris'].index:
            if type(imgs['image_uris'][i]) == type({}):
                img_list.append(imgs['image_uris'][i]['art_crop'])
            else:
                img_list.append(i)
        imgs['art_crop'] = pd.Series(img_list) 

        imgs = imgs[imgs.image_uris.map(type) ==  type({})]

        #Then I save the img json file
        imgs.to_json(data_path)
        #now I download the img files
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        for i in imgs['art_crop'].index:
            r = http.request('GET',imgs['art_crop'][i])
            img = Image.open(io.BytesIO(r.data))
            img.save(f'{img_path}\{i}.jpg')
        return img_path

    def images(data_path,data_type):
        if os.path.isfile(data_path):
            print('images already downloaded')
        else: dataDownload.save_images(dataDownload.api_request(data_type),data_path)

if (__name__ == '__main__') :
    data_path='.\data\oracle-cards.json'
    image='.\data\images.json'
    dataDownload.update(data_path,'bulk')
    dataDownload.images(image,'img')