from data_cleaning import dataCleaning as clean
from data_download import dataDownload as download
data_path='.\data\oracle-cards.json'

request=download.api_request()

download.save_json(request,data_path)

clean.data_clean(data_path)