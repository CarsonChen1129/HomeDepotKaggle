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
df_train = pd.read_csv('../data/train.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('../data/test.csv', encoding = "ISO-8859-1")
df_desc = pd.read_csv('../data/product_descriptions.csv')

# Step 1.2: Concat training dataset and testing dataset for pre-processing
# df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)

# Step 1.3: Add product description
# df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
df_train = pd.merge(df_train, df_desc, how='left', on='product_uid')
df_test = pd.merge(df_test, df_desc, how='left', on='product_uid')

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
# df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
# df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
# df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))
df_train['search_term'] = df_train['search_term'].map(lambda x: str_stemmer(x))
df_train['product_title'] = df_train['product_title'].map(lambda x: str_stemmer(x))
df_train['product_description'] = df_train['product_description'].map(lambda x: str_stemmer(x))
df_test['search_term'] = df_test['search_term'].map(lambda x: str_stemmer(x))
df_test['product_title'] = df_test['product_title'].map(lambda x: str_stemmer(x))
df_test['product_description'] = df_test['product_description'].map(lambda x: str_stemmer(x))

# Step 3: Calculate text feature
# Step 3.1: Calculate Levenshtein distance
# Levenshtein distance between search terms and product title
# df_all['dist_in_title'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)
# # Levenshtein distance between search term and product description
# df_all['dist_in_desc'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)
# # Combine product title and product description as all texts
# df_all['all_texts'] = df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '
df_train['dist_in_title'] = df_train.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)
# Levenshtein distance between search term and product description
df_train['dist_in_desc'] = df_train.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)
# Combine product title and product description as all texts
df_train['all_texts'] = df_train['product_title'] + ' . ' + df_train['product_description'] + ' . '
df_test['dist_in_title'] = df_test.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)
# Levenshtein distance between search term and product description
df_test['dist_in_desc'] = df_test.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)
# Combine product title and product description as all texts
df_test['all_texts'] = df_test['product_title'] + ' . ' + df_test['product_description'] + ' . '

# Generate a dictionary of all text words
# dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in df_all['all_texts'].values)
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in df_all['all_texts'].values)
# dict_train_list = Dictionary(list(tokenize(x, errors='ignore')) for x in df_train['all_texts'].values)
# dict_test_list = Dictionary(list(tokenize(x, errors='ignore')) for x in df_test['all_texts'].values)


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
# df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
# # Calculate similarity between search terms and product description
# df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)
# # Load nltk tokenizer
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# # Convert all texts into a list of sentences, and then convert to be a list of words
# sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
# sentences = [y for x in sentences for y in x]
# # Apply tokenizer
# w2v_corpus = [word_tokenize(x) for x in sentences]
# # Train model
# model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)

df_train['tfidf_cos_sim_in_title'] = df_train.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
df_test['tfidf_cos_sim_in_title'] = df_test.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)
# Calculate similarity between search terms and product description
df_train['tfidf_cos_sim_in_desc'] = df_train.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)
df_test['tfidf_cos_sim_in_desc'] = df_test.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)
# Load nltk tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Convert all texts into a list of sentences, and then convert to be a list of words
sentences = []
for x in df_train['all_texts'].values:
    sentences.append(tokenizer.tokenize(x))

for x in df_test['all_texts'].values:
    sentences.append(tokenizer.tokenize(x))
# sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]
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
# df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
# # calculate cosine similarity between word vector and description
# df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)
# # drop unnecessary columns
# df_all = df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)
df_train['w2v_cos_sim_in_title'] = df_train.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
df_test['w2v_cos_sim_in_title'] = df_test.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
# # calculate cosine similarity between word vector and description
df_train['w2v_cos_sim_in_desc'] = df_train.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)
df_test['w2v_cos_sim_in_desc'] = df_test.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)
# # drop unnecessary columns
# df_all = df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)
df_train = df_train.drop(['search_term','product_title','product_description','all_texts'],axis=1)
df_test = df_test.drop(['search_term','product_title','product_description','all_texts'],axis=1)



# Get the training dataset and testing dataset
# df_train = df_all.loc[df_train.index]
# df_test = df_all.loc[df_test.index]

# df_all.to_pickle('df_all.pkl')
# df_train.to_pickle('df_train.pkl')
# df_test.to_pickle('df_test.pkl')

# df_all = pd.read_pickle('df_all.pkl')
df_train = pd.read_pickle('df_train.pkl')
df_test = pd.read_pickle('df_test.pkl')
# df_test = pd.read_csv('../data/test.csv', encoding = "ISO-8859-1")
# df_test = df_all.loc[df_test.index]
# df_train = df_all.loc[df_train.index]
# df_test = df_all.loc[df_test.index]

# Split another training dataset and drop unnecessary columns
y_train = df_train['relevance'].values
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id'], axis = 1).values


xgb_params0={'colsample_bytree': 1, 'silent': 1, 'nthread': 8, 'min_child_weight': 10,
    'n_estimators': 300, 'subsample': 1, 'learning_rate': 0.09, 'objective': 'reg:linear',
    'seed': 10, 'max_depth': 7, 'gamma': 0.}
xgb_params1={'colsample_bytree': 0.77, 'silent': 1, 'nthread': 8, 'min_child_weight': 15,
    'n_estimators': 500, 'subsample': 0.77, 'learning_rate': 0.035, 'objective': 'reg:linear',
    'seed': 11, 'max_depth': 6, 'gamma': 0.2}

# use models from sklearn: RandomForest <-- change to other model from here
params = [1,3,5,6,7,8,9,10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

# generate predictions
# Record testing id
# df_test = pd.read_csv('../data/test.csv', encoding = "ISO-8859-1")
test_ids = df_test['id']
# rf = RandomForestRegressor(n_estimators=128, max_depth=15)
rf = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=6, max_features=0.9, min_samples_split=1.0, n_jobs=-1, random_state=2014)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/RF_outputs.csv',index=False)

# AdaBoost
from sklearn.ensemble import AdaBoostRegressor
ada_scores = []
for param in params:
    clf = AdaBoostRegressor(n_estimators=30, learning_rate=1.0, loss='linear')
    ada_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    ada_scores.append(np.mean(ada_score))

print(ada_scores)
test_ids = df_test['id']
ab = AdaBoostRegressor(n_estimators=300, learning_rate=1.0, loss='linear')
ab.fit(X_train, y_train)
y_pred = ab.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/AB_outputs.csv',index=False)

# Bagging Regression, Extra Trees Regression, Gradient Boosting Regression
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
bagging_scores = []
extra_scores = []
gradient_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators=30, max_samples=1.0, max_features=1.0)
    bagging_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    bagging_scores.append(np.mean(bagging_score))
    clf = ExtraTreesRegressor(n_estimators=30, max_depth=param)
    extra_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    extra_scores.append(np.mean(extra_score))
    clf = GradientBoostingRegressor(n_estimators=30, max_depth=param)
    gradient_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    gradient_scores.append(np.mean(gradient_score))

print(bagging_scores)
print(extra_scores)
print(gradient_scores)

# bag = BaggingRegressor(n_estimators=300, max_samples=1.0, max_features=1.0)
import xgboost as xgb
bag = BaggingRegressor(base_estimator=xgb.XGBRegressor(**xgb_params1), n_estimators=10, random_state=np.random.RandomState(2016))
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/Bag_outputs.csv',index=False)

et = ExtraTreesRegressor(n_estimators=300, max_depth=16)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/ET_outputs.csv',index=False)

# gb = GradientBoostingRegressor(n_estimators=128, max_depth=16)
gb = GradientBoostingRegressor(n_estimators=500, max_depth=6, min_samples_split=2, min_samples_leaf=15, learning_rate=0.035, loss='ls',random_state=10)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
    if y_pred[i] < 1:
        y_pred[i] = 1
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/GB_outputs.csv',index=False)

# Linear regression
from sklearn.linear_model import LinearRegression
linear_scores = []
for param in params:
    clf = LinearRegression()
    linear_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    linear_scores.append(np.mean(linear_score))

print(linear_scores)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/LR_outputs.csv',index=False)


# SVM
from sklearn import svm
svm_scores = []
for param in params:
    clf = svm.SVR(kernel='linear')
    svm_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    svm_scores.append(np.mean(svm_score))

print(svm_scores)

svm = svm.SVR(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/SVM_outputs.csv',index=False)

# Ridge
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(cv=10)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/Ridge_outputs.csv',index=False)

# MLP
from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor(solver='lbfgs', alpha=1e-5)
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
    if y_pred[i] < 1:
        y_pred[i] = 1
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/MLP_outputs.csv',index=False)

# XGB Boost
from xgboost import XGBRegressor
xgb = XGBRegressor(**xgb_params1)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
    if y_pred[i] < 1:
        y_pred[i] = 1
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/XGB_outputs.csv',index=False)


# KNN
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(128,  weights="uniform", leaf_size=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
    if y_pred[i] < 1:
        y_pred[i] = 1
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/KNN_outputs.csv',index=False)

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dec = DecisionTreeRegressor(criterion='mse', splitter='random', max_depth=4, min_samples_split=7, min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_features='sqrt', random_state=None, max_leaf_nodes=None, presort=False)
dec.fit(X_train, y_train)
y_pred = dec.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 3:
        y_pred[i] = 3
    if y_pred[i] < 1:
        y_pred[i] = 1
pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('outputs/DecisionTree_outputs.csv',index=False)