from bs4 import BeautifulSoup
import pandas as pd
import requests

#URl
wine_url = 'https://top100.winespectator.com/lists/'

html = requests.get(url)

# Create BeautifulSoup object; parse with 'html.parser'
soup = BeautifulSoup(html.text, 'html.parser')

wine_data_table = soup.find('table',{'class':'tablesorter brk tablesorter-default'})
wine_list = wine_data.find_all('span',{'class':'sort-text'})
