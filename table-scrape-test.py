# # thanks to https://www.datahen.com/blog/web-scraping-using-python-beautiful-soup/

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd

# url = 'https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue'
# # https://dmtsms.scotiabank.com/api/rates/fxr
# try:
#     page = requests.get(url)
# except Exception as e:
#     print('Error downloading page: ',e)

# soup = BeautifulSoup(page.text, 'html.parser')

# # Find all tables in the webpage
# tables = soup.find_all('table')

# # Iterate over each table and convert to DataFrame
# dfs = []  # This will contain all dataframes
# for table in tables:
#     dfs.append(pd.read_html(str(table))[0])

# top_revenue_companies = dfs[0]

# top_revenue_companies.columns = top_revenue_companies.columns.get_level_values(0)

# top_revenue_companies = top_revenue_companies.drop(columns=["State-owned", "Ref."])

# top_revenue_companies.to_csv('exchange_rates.csv', index=False)

# thanks to https://stackoverflow.com/a/55732636, https://stackoverflow.com/a/76398418

import pandas as pd
import wikipedia as wp
import requests
from bs4 import BeautifulSoup

import yake
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import collections
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams

def most_common(lst):
    return max(set(lst), key=lst.count)

# return the most basic form a word based on part of speech
def basicForm(word):
    lemmatizer = WordNetLemmatizer()

    if word[0][1] in ['V','VD','VG','VN']:
        return lemmatizer.lemmatize(word, "v")
    else:
        return lemmatizer.lemmatize(word)

html = wp.page("List of national capitals", auto_suggest=False).html().encode("UTF-8")
try: 
    df = pd.read_html(html, extract_links = "body")[1]  # Try 2nd table first as most pages contain contents table first
except IndexError:
    df = pd.read_html(html, extract_links = "body")[0]
print(df['City/Town'].to_string())

# df.to_csv('national_capitals.csv', index=False)

# get what we want from the pages
# city name, keywords, & cluster label
city_df = df['City/Town']
for index, row in df.iterrows():
    # follow the link to that city's page
    # url='https://en.wikipedia.org' + df.iloc[index]['City/Town'][1]
    # print(url)
    # response = requests.get(url=url)
    # soup = BeautifulSoup(response.content, 'html.parser')
    # # title = soup.find('h1')
    # # print(title.string)
    # # print([item.get_text() for item in soup.select("mw-page-title-main")])
    # # city_page = wp.page(df.iloc[index]['City/Town'][0], auto_suggest=False).html().encode("UTF-8")
    # # print(city_page.summary)
    # title = soup.find('h1')
    # print(title)
    # print(title.string)
    city_page = wp.page(df.iloc[index]['City/Town'][1][6:].replace('_',' '), auto_suggest=False)
    # print(city_page.summary)

    # get the keywords of this page's content
    text = city_page.content

    # using NLTK
    english_stops = set(stopwords.words('english'))
    words = word_tokenize(text)
    words_list = [word for word in words if word not in english_stops]
    # print(words_list)
    important_words = ' '.join(words_list)
    # print(important_words)

    # NLTK information extraction
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    best_parts = []
    for sent in tagged_sentences:
        # print(nltk.ne_chunk(sent))
        best_parts_of_sent = [t[0] for t in sent if (t[1] == "NN" or t[1] == "NNS" or t[1] == "JJ")]
        for word in best_parts_of_sent:
            best_parts.append(word)
    # print(best_parts)
    # print(len(best_parts))
    best_parts_str = ' '.join(best_parts)
    # print(best_parts_str)

    # print the top keyphrases out of those words
    # using YAKE!
    kw_extractor = yake.KeywordExtractor(n=1, top=500, stopwords=english_stops)
    keyphrases = kw_extractor.extract_keywords(best_parts_str)
    # print("new keywords!")
    # for kw, v in keyphrases:
    #     print("Keyphrase: ",kw, ": score", v)