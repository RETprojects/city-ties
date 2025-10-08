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

html = wp.page("List of national capitals", auto_suggest=False).html().encode("UTF-8")
try: 
    df = pd.read_html(html, extract_links = "body")[1]  # Try 2nd table first as most pages contain contents table first
except IndexError:
    df = pd.read_html(html, extract_links = "body")[0]
print(df['City/Town'].to_string())

# get what we want from the pages
# city name, keywords, & cluster label
# city_df = df['City/Town']
for index, row in df.iterrows():
    # follow the link to that city's page
    url='https://en.wikipedia.org' + df.iloc[index]['City/Town'][1]
    print(url)
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # title = soup.find('h1')
    # print(title.string)
    print([item.get_text() for item in soup.select("h2 .mw-headline")])
    # city_page = wp.page(df.iloc[index]['City/Town'][0], auto_suggest=False).html().encode("UTF-8")
    # print(city_page.summary)