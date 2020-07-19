import urllib3
import pandas as pd

class dataDownload:
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
    
    def main(data_path):
        data_download.save_json(data_download.api_request(),data_path)

if (__name__ == '__main__') :
    data_path='.\data\oracle-cards.json'
    data_download.main(data_path)