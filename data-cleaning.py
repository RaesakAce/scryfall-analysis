import os
import re
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
, 'toughness'
,'prices'
,'edhrec_rank']

keys=['usd','usd_foil','eur','tix']

clean = data[useful_columns].copy()

tokens = [num for num in range(21634) if type(clean['colors'][num]) is not list]
print(tokens)

for val in keys:
    clean[val] = pd.Series([price[val] for price in clean['prices']])

clean.to_json('.\data\oracle-cards-clean.json')

sb.relplot(x="usd", y="eur",sizes=(0.4, 0.4), alpha=.5, height=6, data=clean)
plt.show()
