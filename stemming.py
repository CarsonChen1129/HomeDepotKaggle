import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer


df_train = pd.read_csv('data/train.csv', encoding='ISO-8859-1')
df_test = pd.read_csv('data/test.csv', encoding='ISO-8859-1')
df_desc = pd.read_csv('data/product_descriptions.csv')
df_attr = pd.read_csv('data/attributes.csv')

stemmer = SnowballStemmer('english')

def str_stemmer(s):
    '''
    Stem all description with the following step: lower() -> split() -> stemmer.stem -> join
    :param s: input string
    :return: stemmed string
    '''
    return " ".join([stemmer.stem(word) for word in str(s).lower().split()])

print('stemming train search term')
df_train['search_term'] = df_train['search_term'].map(lambda x: str_stemmer(x))
print('stemming train product title')
df_train['product_title'] = df_train['product_title'].map(lambda x: str_stemmer(x))
df_train.to_csv('stemmed_data/train.csv')

print('stemming test search term')
df_test['search_term'] = df_test['search_term'].map(lambda x: str_stemmer(x))
print('stemming test product title')
df_test['product_title'] = df_test['product_title'].map(lambda x: str_stemmer(x))
df_test.to_csv('stemmed_data/test.csv')

print('stemming product description')
df_desc['product_description'] = df_desc['product_description'].map(lambda x: str_stemmer(x))
df_desc.to_csv('stemmed_data/product_descriptions.csv')

print('stemming attrubutes')
df_attr['value'] = df_attr['value'].map(lambda x: str_stemmer(x))
df_attr.to_csv('stemmed_data/attributes.csv')