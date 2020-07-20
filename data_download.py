import urllib3
import pandas as pd
import os
import time

class dataDownload():
    def api_request():
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        r = http.request('GET','https://api.scryfall.com/bulk-data')
        if (r.status == 200):
            uri_data = pd.read_json(r.data)['data'][0]['download_uri']
            r = http.request('GET',uri_data)
        return r
    
    def save_json(r,data_path):
        if (r.status == 200):
            pd.read_json(r.data).to_json(data_path)
    
    def update(data_path):
        last_update=(time.time()-os.stat(data_path).st_mtime)/3600
        if (last_update > 24):
            dataDownload.save_json(dataDownload.api_request(),data_path)
        else: print(f'The data has been updated {round(last_update,2)} hours ago')

if (__name__ == '__main__') :
    data_path='.\data\oracle-cards.json'
    dataDownload.update(data_path)
