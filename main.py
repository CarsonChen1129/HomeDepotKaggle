# import necessary library
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
import os
import Levenshtein
from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
from scipy import spatial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Step 1: read data

# Step 1.1: read the data
df_train = pd.read_csv('data/train.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding = "ISO-8859-1")
df_desc = pd.read_csv('data/product_descriptions.csv')

# Step 1.2: Concat training dataset and testing dataset for pre-processing
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Step 1.3: Add product description
df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

# Step 2: Pre-processing
# Step 2.1: import snowball stemmer
stemmer = SnowballStemmer('english')


def str_stemmer(s):
    '''
    Stem all description with the following step: lower() -> split() -> stemmer.stem -> join
    :param s: input string
    :return: stemmed string
    '''
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


def str_common_word(str1, str2):
    '''
    Count common words in two strings
    :param str1: string
    :param str2: string
    :return: number of common words
    '''
    return sum(int(str2.find(word) >= 0) for word in str1.split())


# Step 2.2: Stem all data
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

# Step 3: Calculate text feature
# Step 3.1: Calculate Levenshtein distance
# Levenshtein distance between search terms and product title
df_all['dist_in_title'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)
# Levenshtein distance between search term and product description
df_all['dist_in_desc'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)
# Combine product title and product description as all texts
df_all['all_texts'] = df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '

# Generate a dictionary of all text words
dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in df_all['all_texts'].values)


class MyCorpus(object):
    '''
    Convert dictionary to be bag of words representation.
    '''
    def __iter__(self):
        for x in df_all['all_texts'].values:
            yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))

# new an Corpus instance
corpus = MyCorpus()
# Calculate TF-IDF on the bag of words vectors
tfidf = TfidfModel(corpus)


def to_tfidf(text):
    '''
    calculate TF-IDF on bag of words vector
    :param text: input text
    :return:
    '''
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
    return res


def cos_sim(text1, text2):
    '''
    Calculate cosine similarity between two texts
    :param text1: input string
    :param text2: input string
    :return: cosine similarity
    '''
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])


# Calculate similarity between search terms and product title
df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
# Calculate similarity between search terms and product description
df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)
# Load nltk tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Convert all texts into a list of sentences, and then convert to be a list of words
sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
sentences = [y for x in sentences for y in x]
# Apply tokenizer
w2v_corpus = [word_tokenize(x) for x in sentences]
# Train model
model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)


def get_vector(text):
    '''
    Get the vector representation of input text.
    :param text: input string
    :return:
    '''
    res = np.zeros([128])
    count = 0
    for word in word_tokenize(text):
        res += model[word]
        count+=1
    return res/count


def w2v_cos_sim(text1, text2):
    '''
    Calculate cosine similarity between two word vectors.
    :param text1: input string
    :param text2: input string
    :return: cosine similarity
    '''
    try:
        w2v1 = get_vector(text1)
        w2v2 = get_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)


# calculate cosine similarity between word vector and title
df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
# calculate cosine similarity between word vector and description
df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)
# drop unnecessary columns
df_all = df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)

# Get the training dataset and testing dataset
df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]
# Record testing id
test_ids = df_test['id']
# Split another training dataset and drop unnecessary columns
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis = 1).values

# use models from sklearn: RandomForest <-- change to other model from here
params = [1,3,5,6,7,8,9,10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

# generate predictions
rf = RandomForestRegressor(n_estimators=30, max_depth=6)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/RF_outputs.csv',index=False)


