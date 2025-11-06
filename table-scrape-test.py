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

from urllib.parse import unquote

from multiprocessing import Pool
from joblib import parallel_backend

def most_common(lst):
    return max(set(lst), key=lst.count)

# return the most basic form a word based on part of speech
def basicForm(word):
    lemmatizer = WordNetLemmatizer()

    if word[0][1] in ['V','VD','VG','VN']:
        return lemmatizer.lemmatize(word, "v")
    else:
        return lemmatizer.lemmatize(word)

if __name__ == '__main__': # for Windows compatibility

    html = wp.page("List of national capitals", auto_suggest=False).html().encode("UTF-8")
    try: 
        df = pd.read_html(html, extract_links = "body")[1]  # Try 2nd table first as most pages contain contents table first
    except IndexError:
        df = pd.read_html(html, extract_links = "body")[0]
    print(df['City/Town'].to_string())

    # df.to_csv('national_capitals.csv', index=False)

    # get what we want from the pages
    # city name, keywords, & cluster label
    city_df = df[['City/Town', 'Country/Territory']]
    # city_df.rename({'City/Town': 'Name', 'Country/Territory' : 'Keywords'}, axis=1)
    city_df.rename(columns={'City/Town': 'Name', 'Country/Territory': 'Keywords'})
    for index, row in df.iterrows():
        # record the city's name in the new dataframe
        city_name = unquote(str(df.iloc[index]['City/Town'][1][6:].replace('_',' '))) # thanks to https://stackoverflow.com/a/16566128
        print(city_name)
        city_df.loc[index, 'Name'] = city_name

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
        city_page = wp.page(city_name, auto_suggest=False)
        # print(city_page.summary)

        # get the keywords of this page's content
        text = city_page.content.lower().strip()

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

        # store the keywords in that city's Keywords column
        city_df.loc[index, 'Keywords'] = ' '.join([kw for kw, v in keyphrases])
    city_df = city_df.drop(['City/Town', 'Country/Territory'], axis=1) # we don't need these columns now
    city_df.drop_duplicates(subset=['Name'], keep='last') # dropping duplicates in case a city is mentioned twice or more in the table that we scraped

    # make clusters of the cities!
    # thanks to https://www.kaggle.com/code/ronnahshon/unsupervised-clustering-with-us-census-tracts

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import json
    import requests
    import numpy as np
    import pandas as pd
    from tqdm.notebook import tqdm
    # from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer

    # # Convert dataframe into numpy array (allows for faster computation)
    # X = city_df[['Keywords']].values
    # thanks to https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    X = vectorizer.fit_transform(city_df['Keywords']).toarray()

    # # Define range of clusters to check
    # inertia_scores = []
    # silhouette_scores = []
    # no_of_clusters = range(2, 22)

    # # Calculate intertia & silhouette average for each cluster
    # for cluster in tqdm(no_of_clusters):
    #     kmeans = KMeans(n_clusters=cluster, init="k-means++", random_state=42)
    #     kmeans = kmeans.fit(X)
        
    #     inertia = kmeans.inertia_
    #     silhouette_avg = silhouette_score(X, kmeans.labels_)
        
    #     inertia_scores.append(round(inertia))
    #     silhouette_scores.append(silhouette_avg)

    # # Interia scree plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.plot(range(1, len(no_of_clusters)+1), inertia_scores, marker="o", linewidth=2, linestyle="--")
    # plt.xticks(range(1, len(no_of_clusters)+1), no_of_clusters, fontsize=12, fontweight="bold")
    # ax.set_yticklabels(["{:,.0f}".format(x/1000) + "K" for x in ax.get_yticks()])
    # plt.yticks(fontsize=12, fontweight="bold")
    # plt.xlabel("# of Clusters", fontsize=16, fontweight="bold")
    # plt.ylabel("Inertia", fontsize=16, fontweight="bold")
    # plt.title("Inertia Scree Plot per Cluster", fontsize=20, fontweight="bold")
    # ax.title.set_position([.5, 1.025])
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()

    # slopes = [0]
    # slopes_pct_change = []
    # inertia_df = pd.DataFrame()
    # inertia_df["inertia"] = inertia_scores
    # inertia_df["n_clusters"] = inertia_df.index + 2

    # def derivative_calc(df, x_field, y_field):
    #     x_values = df[x_field].values
    #     y_values = df[y_field].values
    #     for i in range(1, len(x_values)):
    #         (x1, y1) = (x_values[i-1], y_values[i-1])
    #         (x2, y2) = (x_values[i], y_values[i])
    #         slope = round((y2 - y1) / (x2 - x1), 4)
    #         slopes.append(slope)
    #         slopes_pct_change.append((abs(slopes[i-1]) - abs(slopes[i])) / abs(slopes[i-1]))
    #     df["slopes"] = slopes
    #     df["slopes_pct_change"] = slopes_pct_change + [0]

    # # Define optimal number of clusters
    # derivative_calc(inertia_df, "n_clusters", "inertia")
    # n_clusters_kmeans = int(inertia_df.loc[inertia_df["slopes_pct_change"].idxmax()]["n_clusters"])
    # print("# of Clusters for KMeans Algorithm:", n_clusters_kmeans)

    kmeans = KMeans(n_clusters=20, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    def cluster_cnts(predictions, algorithm):
        
        unique, counts = np.unique(predictions, return_counts=True)
        cluster_cnts_df = pd.DataFrame(counts)
        cluster_cnts_df["ratio"] = round(100 * cluster_cnts_df[0] / cluster_cnts_df[0].sum(), 4)
        cluster_cnts_df = cluster_cnts_df.reset_index()
        cluster_cnts_df.columns = ["cluster", "count", "ratio"]
        
        print(f"Breakdown of Cities in Each {algorithm} Cluster")
        return cluster_cnts_df

    print(cluster_cnts(y_kmeans, "KMeans"))

    silhouette_kmeans_euclidean = round(silhouette_score(X, y_kmeans, metric="euclidean"), 4)
    silhouette_kmeans_manhattan = round(silhouette_score(X, y_kmeans, metric="manhattan"), 4)
    print("Silhouette Score KMeans Euclidean:", silhouette_kmeans_euclidean, "\nSilhouette Score KMeans Manhattan:", silhouette_kmeans_manhattan)

    # store cluster number for each city
    # city_df['Kmeans'] = None
    city_df.insert(loc=1, column='Kmeans', value=['' for i in range(city_df.shape[0])]) # initialize a new empty column to store cluster numbers for this particular algorithm
    for index, row in city_df.iterrows():
        # store the number in that city's cluster number column
        city_df.loc[index, 'Kmeans'] = y_kmeans[index]

    # save the data
    city_df.to_csv('national_capitals.csv', index=False)