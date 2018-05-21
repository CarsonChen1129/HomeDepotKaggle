from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
import csv
import numpy as nup
import pandas as pds
import matplotlib.pyplot as mpl
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


# Common Words Calculation
def common(s1, s2):
    return sum(int(s2.find(word)>=0) for word in s1.split())

# Merge attribute to find terms in query
def merge(attr):
    names = attr["name"]
    values = attr["value"]
    merge = []
    for name, value in zip(names, values):
        merge.append(" ".join((name, value)))
    return " ".join(merge)


if __name__ == '__main__':
    # Load data with pandas
    test_data = pds.read_csv("test.csv", encoding="ISO-8859-1")
    train_data = pd.read_csv("train.csv", encoding="ISO-8859-1")
    attribute_data = pd.read_csv("attributes.csv", encoding="ISO-8859-1")
    product_description_data = pd.read_csv("product_descriptions.csv", encoding="ISO-8859-1")

    train_data['search_term'] = train_data['search_term'].map(lambda x:str_stemmer(x))
    test_data['search_term'] = test_data['search_term'].map(lambda x:str_stemmer(x))
    train_data['product_title'] = train_data['product_title'].map(lambda x:str_stemmer(x))
    test_data['product_title'] = test_data['product_title'].map(lambda x:str_stemmer(x))

    # Join tables
    new_train = pds.merge(train_data, product_description_data, how='left', on='product_uid')
    new_test = pds.merge(test_data, product_description_data, how='left', on='product_uid')

    new_train['product_description'] = new_train['product_description'].map(lambda x:str_stemmer(x))
    new_test['product_description'] = new_test['product_description'].map(lambda x:str_stemmer(x))

    new_train['query_len'] = new_train['search_term'].map(lambda x:len(x.split())).astype(nup.int64)
    new_test['query_len'] = test['search_term'].map(lambda x:len(x.split())).astype(nup.int64)

    new_train['info'] = new_train['search_term']+"\t"+new_train['product_title']+"\t"+new_train['product_description']
    new_test['info'] = new_test['search_term']+"\t"+new_test['product_title']+"\t"+new_test['product_description']


    new_train['common_title'] = new_train['info'].map(lambda x:common(x.split('\t')[0],x.split('\t')[1]))
    new_train['common_descriptoin'] = new_train['info'].map(lambda x:common(x.split('\t')[0],x.split('\t')[2]))

    test['common_title'] = test['info'].map(lambda x:common(x.split('\t')[0],x.split('\t')[1]))
    test['common_descriptoin'] = test['info'].map(lambda x:common(x.split('\t')[0],x.split('\t')[2]))


    attribute_data.dropna(how="all", inplace=True)
    attribute_data["product_uid"] = attribute_data["product_uid"].astype(int)
    attribute_data["value"] = attribute_data["value"].astype(str)
    product_attributes = attribute_data.groupby("product_uid").apply(merge)
    product_attributes = product_attributes.reset_index(name="product_attributes")

    new_train = pd.merge(train, product_attributes, how="left", on="product_uid")

    new_train['product_attributes_x'] = new_train['product_attributes'].fillna('')
    new_train['info_attr'] = new_train['search_term']+"\t"+new_train['product_attributes']
    new_train['common_attributes'] = new_train['info_attr'].map(lambda x:common(x.split('\t')[0],x.split('\t')[1]))
    new_train = new_train.drop(['search_term','product_title','product_description','description_title', 'product_attributes', 'info', 'info_attr', ],axis=1)


    new_test = pd.merge(new_test, product_attributes, how="left", on="product_uid")
    new_test['product_attributes'] = new_test['product_attributes'].fillna('')
    new_test['info_attr'] = new_test['search_term']+"\t"+test['product_attributes']
    new_test['common_attributes'] = new_test['info_attr'].map(lambda x:common(x.split('\t')[0],x.split('\t')[1]))
    new_test = new_test.drop(['search_term','product_title','product_description','description_title', 'product_attributes', 'info', 'info_attr', ],axis=1)

    new_train = new_train.drop(['search_term','product_title','product_description','description_title', 'product_attributes_x', 'info', ],axis=1)
    new_test.to_csv('test_feature.csv')
    new_train.to_csv('train_feature.csv')

    new_train = new_train.drop(['search_term','product_title','product_description', 'info', 'product_uid'],axis=1)
    new_test = new_test.drop(['search_term','product_title','product_description', 'info', 'product_uid'],axis=1)

    train.to_csv('train_feature_base.csv')
    test.to_csv('test_feature_base.csv')




