import pandas as pd
import seaborn as sb
import pyautogui as gui

data_path='.\data\oracle-cards-20200629050643.json'
data=pd.read_json(data_path)


useful_columns=['id'
,'name'
,'mana_cost'
,'cmc'
, 'type_line'
, 'oracle_text'
, 'colors'
, 'color_identity'
, 'keywords'
, 'legalities'
, 'games'
, 'reserved'
, 'foil'
, 'nonfoil'
,'promo'
, 'reprint'
,'set'
, 'border_color'
, 'frame'
, 'full_art'
, 'textless'
,'power'
, 'toughness']
print(data['edhrec_rank'])
print(type(data['prices']))
print(data.columns)
sb.barplot(x="edhrec_rank",data=data);
gui.alert('everything was printed')
