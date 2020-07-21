import os
import re
import pandas as pd


class dataCleaning():
    def data_clean(data_path):
        print ('begin cleaning')
        clean_path='.\data\oracle-cards-clean.json'
        #Read with pandas the raw json data and put it in a dataFrame
        data=pd.read_json(data_path)
        #Select the columns which I want to keep in the dataFrame
        useful_columns=['id'
        ,'layout'
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

        #Copy only the part that I'm interested in of the df
        clean = data[useful_columns].copy()

        #'prices' is a dictionary, I want to separate in different columns the values
        keys=['usd','usd_foil','eur','tix']
        for val in keys:
            clean[val] = pd.Series([price[val] for price in clean['prices']])

        #with a condition the dataFrame gets rid of cards that
        #due to strange layout would be difficult or meaningless to analyze
        clean = clean[clean.layout == 'normal']

        #drop useless columns
        #clean=clean.drop(columns=['layout','prices'])
        print('saving clean data')

        #Then I save the clean json file
        clean.to_json(clean_path)
        return clean_path

    if (__name__ == '__main__') :
        data_path='.\data\oracle-cards.json'
