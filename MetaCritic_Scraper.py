import pandas as pd
import time
from bs4 import BeautifulSoup
import requests
import numpy as np
import re

df = pd.read_csv('/media/ravi/Extra_50/tmdb-5000-movie-dataset/tmdb_5000_movies.csv')
header1 = {'User-Agent':"Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:63.0) Gecko/20100101 Firefox/63.0"}

df['metacritic_metascore'] = np.NaN
df['metacritic_metascore1'] = np.NaN

url = "https://www.metacritic.com/movie/"

for i in range(len(df['original_title'])):
    movie_title = re.sub(' +', ' ',df.loc[i,'original_title'])
    movie_title1 = re.sub('[^a-zA-z0-9 -]','',movie_title)
    url_metacritic = url+movie_title1.replace(' ','-').lower()
    if url_metacritic.strip() != '':
        url_meta=''
        while not url_meta:
            try:
                url_meta=requests.get(url_metacritic,headers=header1)
                soup=BeautifulSoup(url_meta.content,'html.parser')
                Score_Div = soup.find('a',class_="metascore_anchor")
                if Score_Div:
                    if re.sub('[^0-9]','',Score_Div.get_text()).isnumeric():
                        df.loc[i,'metacritic_metascore']=int(re.sub('[^0-9]','',Score_Div.get_text()))
                break
            except requests.exceptions.ConnectionError:
                time.sleep(5)
                continue

for i in range(len(df['title'])):
    movie_title = re.sub(' +', ' ',df.loc[i,'title'])
    movie_title1 = re.sub('[^a-zA-z0-9 -]','',movie_title)
    url_metacritic = url+movie_title1.replace(' ','-').lower()
    if url_metacritic.strip() != '':
        url_meta=''
        while not url_meta:
            try:
                url_meta=requests.get(url_metacritic,headers=header1)
                soup=BeautifulSoup(url_meta.content,'html.parser')
                Score_Div = soup.find('a',class_="metascore_anchor")
                if Score_Div:
                    if re.sub('[^0-9]','',Score_Div.get_text()).isnumeric():
                        df.loc[i,'metacritic_metascore1']=int(re.sub('[^0-9]','',Score_Div.get_text()))
                break
            except requests.exceptions.ConnectionError:
                time.sleep(5)
                continue

df.to_csv('/media/ravi/Extra_50/tmdb-5000-movie-dataset/tmdb_metacritic.csv')
