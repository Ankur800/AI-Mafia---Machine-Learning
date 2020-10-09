from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout

warnings.filterwarnings("ignore")

data = pd.read_json("./data/tops_fashion.json")
# print(data)

# print(data.shape[0], data.shape[1])

# print(data.columns)

data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]

# print(data.head())

# print(data['product_type_name'].describe())

# print(data['product_type_name'].unique())

product_type_count = Counter((data['product_type_name']))
# print(product_type_count.most_common(10))

# print(data['brand'].describe())

brand_count = Counter((data['brand']))
# print(brand_count.most_common(10))

# print(data['color'].describe())

color_count = Counter(data['color'])
# print(color_count.most_common(10))

# print(data['formatted_price'].describe())

price_count = Counter(data['formatted_price'])
# print(price_count.most_common(10))

# print(data['title'].describe())

data.to_pickle("./data/180k_apparel_data")

data = data.loc[~data['formatted_price'].isnull()]
# print(data.shape[0])

data.to_pickle('./data/28k_apparel_data')

data = pd.read_pickle('./data/28k_apparel_data')
(data.duplicated('title'))
# print(sum(data.duplicated('title')))

# print(data.head())

# Remove all products with very few words in title
data_sorted = data[data['title'].apply(lambda x : len(x.split()) > 4)]
# print(data_sorted.shape)

# Sort the whole data based on title (alphabetical order of title)
data_sorted.sort_values('title', inplace=True, ascending=False)

indices = []
for i,row in data_sorted.iterrows():
    indices.append(i)

# print(indices)

import itertools

stage1_dedupe_asins = []
i = 0
j = 0
num_data_points = data_sorted.shape[0]
while i < num_data_points and j < num_data_points:
    previous_i = i

    a = data['title'].loc[indices[i]].split()

    j = i + 1

    while j < num_data_points:

        b = data['title'].loc[indices[j]].split()

        length = max(len(a), len(b))

        count = 0;

        for k in itertools.zip_longest(a, b):
            if (k[0] == k[1]):
                count += 1

        if (length - count) > 2:

            stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])

            if j == num_data_points - 1:
                stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[j]])

            i = j
            break

        else:
            j += 1

    if previous_i == i:
        break

data = data.loc[data['asin'].isin(stage1_dedupe_asins)]

# print(data.shape)

data.to_pickle('./data/17k_apperal_data')

data = pd.read_pickle('./data/17k_apperal_data')

indices = []
for i, row in data.iterrows():
    indices.append(i)

stage2_dedupe_asins = []
while len(indices) != 0:
    i = indices.pop()
    stage2_dedupe_asins.append(data['asin'].loc[i])

    a = data['title'].loc[i].split()

    for j in indices:

        b = data['title'].loc[j].split()

        length = max(len(a), len(b))

        count = 0

        for k in itertools.zip_longest(a, b):
            if (k[0] == k[1]):
                count += 1

        if (length - count) < 3:
            indices.remove(j)

data = data.loc[data['asin'].isin(stage2_dedupe_asins)]

print(data.shape)

data.to_pickle('./data/16k_apperal_data')

# Text Processing
data = pd.read_pickle('./data/16k_apperal_data')

