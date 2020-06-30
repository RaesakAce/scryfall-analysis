import pandas as pd
import seaborn as sb
import pyautogui as gui

data_path='.\data\oracle-cards-20200629050643.json'
data=pd.read_json(data_path)

prices=data['prices']
prices=prices['eur']
print(prices)
useful_columns=['id','name','mana_cost','cmc', 'type_line', 'oracle_text', 'colors', 'color_identity', 'keywords', 'legalities', 'games', 'reserved', 'foil', 'nonfoil','promo', 'reprint','set', 'border_color', 'frame', 'full_art', 'textless','power', 'toughness']
print(data['edhrec_rank'])
print(data.columns)
#sb.relplot(x="edhrec_rank", y="prices",data=data);
gui.alert('everything was printed')
