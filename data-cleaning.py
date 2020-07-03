import pandas as pd
import seaborn as sb
import pyautogui as gui
import matplotlib.pyplot as plt
sb.set(style="whitegrid")

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

keys=['usd','usd_foil','eur','tix']

data_copy = data

for val in keys:
    data_copy[val] = pd.Series([price[val] for price in data_copy['prices']])
data_copy.to_csv('.\data\oracle-cards-clean.csv')

sb.relplot(x="usd", y="eur",sizes=(400, 400), alpha=.5, height=6, data=data_copy)
plt.show()
