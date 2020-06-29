import pandas as pd
import seaborn as sb

data_path='.\data\oracle-cards-20200629050643.json'
data=pd.read_json(data_path)

useful_columns=['id','name','mana_cost','cmc', 'type_line', 'oracle_text', 'colors', 'color_identity', 'keywords', 'legalities', 'games', 'reserved', 'foil', 'nonfoil','promo', 'reprint','set', 'border_color', 'frame', 'full_art', 'textless','power', 'toughness']
print (data.head())