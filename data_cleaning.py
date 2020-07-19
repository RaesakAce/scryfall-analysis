import os
import re
import pandas as pd


class dataCleaning:
    
    def data_clean(data_path):
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

    if (__name__ == '__main__') :
        data_path='.\data\oracle-cards.json'
        data_clean(data_path)