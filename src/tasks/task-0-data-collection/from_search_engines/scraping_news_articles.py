import time
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


def get_query(base_words_query, samples=10):
    
    max_samples = np.product([len(l) for l in base_words_query])
    if samples>max_samples:
        samples = max_samples
    
    list_valid_arrays = []
    i = 0
    while i<samples:
        list_query_tmp = []
        for list_words in base_words_query:
            index =  np.random.randint(0, len(list_words), 1)                                       
            list_query_tmp.append(list_words[index[0]])            
                    
        if ' '.join(list_query_tmp) not in list_valid_arrays:
            list_valid_arrays.append(' '.join(list_query_tmp))
            i += 1            

    return list_valid_arrays


def get_news_links(list_queries, driver, n_pages=5):
    list_head_news = [] 
    enable_to_scrap = True
    for query in list_queries:

        print('query:', query)
        for page in range(1, n_pages+1):
            url = "http://www.google.com/search?q=" + query + "&start=" + str((page - 1) * 10)
            #delay = int(np.median(np.random.random(10)*10))
            delay = int(np.quantile(np.random.random(30)*30, np.random.choice([0.25, 0.5, 0.75], 1)))
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # soup = BeautifulSoup(r.text, 'html.parser')
        
            print('page:', page, 'delay:',delay)
            
            general_search = soup.find_all('div', class_="v7W49e")
            if len(general_search) > 0:
                for item in list(general_search[0].children):
                    content = item.find('div', class_ = 'yuRUbf')
                    if content is not None:
                        title_tmp = content.find('a').text
                        link_tmp = content.find('a')['href']

                        date_tmp = item.find('span', class_ = 'MUxGbd wuQ4Ob WZ8Tjf')    
                        if date_tmp is not None:
                            list_span = []
                            for span in date_tmp.find_all('span'):
                                list_span.append(span.text)

                            list_head_news.append({'query':query, 'title':title_tmp, 'link':link_tmp, 'date':' '.join(list_span)})
                            
            if len(list_head_news) == 0:
                print('\n Google have detected unusual traffic from your computer network, change IP or try later')
                enable_to_scrap = False
                break
            
        if not enable_to_scrap:
            break
        
    return list_head_news

def get_news_links_from_bing(list_queries, headers, n_pages=5):
    list_head_news = [] 
    enable_to_scrap = True
    for query in list_queries:

        print('query:', query)
        for page in range(1, n_pages+1):
            
            url = "http://www.bing.com/search?q=" + query + "&first=" + str((page - 1) * 10)

            delay = int(np.quantile(np.random.random(30)*30, np.random.choice([0.25, 0.5, 0.75], 1)))
            
            r = requests.get(url, headers=headers)            
            soup = BeautifulSoup(r.text, 'html.parser')

            print('page:', page, 'delay:',delay)

            general_search = soup.find_all('ol', {'id':'b_results'})
            link_tmp = ''
            if len(general_search) > 0:
                for item in list(general_search[0].children):
                    itema = item.find('a')
                    title_tmp = itema.text
                    if 'href' in itema.attrs.keys():
                        link_tmp = itema['href']

                    date_tmp = item.find('span', class_ = 'news_dt')    
                    if date_tmp is not None:    
                        date_tmp = date_tmp.text

                    list_head_news.append({'query':query, 'title':title_tmp, 'link':link_tmp, 'date':date_tmp})

            if len(list_head_news) == 0:
                print('\n Bing have detected unusual traffic from your computer network, change IP or try later')
                enable_to_scrap = False
                break

        if not enable_to_scrap:
            break

    return list_head_news

def scraping_text(df, link_column='link'):
    
    list_news = []
    
    for i in np.arange(df.shape[0]): 
    
        dict_data = df.iloc[i].to_dict()
        delay = int(np.median(np.random.random(5)*5))
        time.sleep(delay)

        try:
            html_tmp = requests.get(dict_data[link_column]).text        
            not_error = True
        except Exception as e:
            not_error = False
            print('Error in GET request')

        if not_error:
            try:
                soup_tmp = BeautifulSoup(html_tmp, 'html.parser')  
                not_error = True
            except Exception as e:
                not_error = False
                print('Error parsing HTML page')

        if not_error:
            list_paragraph = []
            for paragraph in list(soup_tmp.find_all(['p'])):        
                list_paragraph.append(paragraph.text)
            dict_data['text'] = ' '.join(list_paragraph)
            list_news.append(dict_data)

        print(i, 'delay:', delay)
                
    return pd.DataFrame(list_news)

def save_data(df, text_column, name_file):
    df[text_column] = df[text_column].str.encode('utf-8')
    df.to_csv(name_file)
