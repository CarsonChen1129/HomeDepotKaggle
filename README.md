# HomeDepotKaggle
Home Depot Product Relevance Model - A Kaggle task 

[Home Depot Product Relevance Model Competition](https://www.kaggle.com/c/home-depot-product-search-relevance)

## Getting Started

This is a minor study(play around) for the Home Depot Product Relevance Model Kaggle Task. Our final score is not optimized to hit the top leaderboard, but our performance is considerable for only 6 features.

### Prerequisites

```
Python 3.6
Jupyter notebook
```

## Running the tests
Just simply download and run the script at your choice (python script or jupyter notebook)

## Features

Only 6 features extracted from the dataset to achieve the study goal. They are:
1. Levenshtein distance between search term and product title
2. Levenshtein distance between search term and product description
3. Cosine similarity between search term and product title (TF-IDF)
4. Cosine similarity between search term and product description (TF-IDF)
5. Cosine similarity between search term and product title (Word2Vec)
6. Cosine similarity between search term and product description (Word2Vec)

## Pipeline

1. Fill up all missing data with empty string value or default value
2. Merge the training data set and testing data set with description data set
3. Extract and parse different features
.. * Apply the following steps on each description: to_lowercase -> split -> apply Snowball stemmer -> aggregate
4. Refine training data set and testing data set
5. Test data with different models

## Models and score

### Random Forest

| Parameters                                                                                                                	| Private Score 	| Public Score 	|
|---------------------------------------------------------------------------------------------------------------------------	|---------------	|--------------	|
| n_estimators=30, max_depth=16                                                                                             	| 0.48779       	| 0.48956      	|
| n_estimators=100, max_depth=80                                                                                            	| 0.4943        	| 0.49739      	|
| n_estimators=128, max_depth=8                                                                                             	| 0.4856        	| 0.48769      	|
| n_estimators=128, max_depth=5                                                                                             	| 0.49004       	| 0.4913       	|
| n_estimators=128, max_depth=15                                                                                            	| 0.48734       	| 0.4899       	|
| n_estimators=500, max_depth=5, min_samples_leaf=6, max_features=0.9, min_samples_split=1.0, n_jobs= -1, random_state=2014 	| 0.48971       	| 0.4911       	|

### Ada Boost

| Parameters       	| Private Score 	| Public Score 	|
|------------------	|---------------	|--------------	|
| n_estimators=30  	| 0.50011       	| 0.50039      	|
| n_estimators=128 	| 0.49972       	| 0.49978      	|
| n_estimators=300 	| 0.50021       	| 0.50048      	|

### Boosting Regression

| Parameters       	| Private Score 	| Public Score 	|
|------------------	|---------------	|--------------	|
| n_estimators=30  	| 0.50007       	| 0.5034       	|
| n_estimators=128 	| 0.49383       	| 0.49643      	|
| n_estimators=300 	| 0.49295       	| 0.49592      	|

### Extra Tree Regression

| Parameters                     	| Private Score 	| Public Score 	|
|--------------------------------	|---------------	|--------------	|
| n_estimators=30, max_depth=16  	| 0.49062       	| 0.4918       	|
| n_estimators=128, max_depth=8  	| 0.48808       	| 0.48949      	|
| n_estimators=128, max_depth=16 	| 0.48659       	| 0.48817      	|
| n_estimators=300, max_depth=16 	| 0.48604       	| 0.48784      	|

### Gradient Boosting Regression

| Parameters                                                                                                                	| Private Score 	| Public Score 	|
|---------------------------------------------------------------------------------------------------------------------------	|---------------	|--------------	|
| n_estimators=30, max_depth=6                                                                                              	| 0.48533       	| 0.48728      	|
| n_estimators=128, max_depth=6                                                                                             	| 0.48505       	| 0.48733      	|
| n_estimators=128, max_depth=16                                                                                            	| 0.50459       	| 0.50796      	|
| n_estimators=500, max_depth=6, min_samples_split=1.0, min_samples_leaf=15, learning_rate=0.035, loss='ls',random_state=10 	| 0.48748       	| 0.4897       	|
| n_estimators=500, max_depth=6, min_samples_split=2, min_samples_leaf=15, learning_rate=0.035, loss='ls',random_state=10   	| 0.48476       	| 0.48694      	|

### Linear Regression

| Parameters 	| Private Score 	| Public Score 	|
|------------	|---------------	|--------------	|
|            	| 0.49104       	| 0.49301      	|

### Ridge Regression

| Parameters 	| Private Score 	| Public Score 	|
|------------	|---------------	|--------------	|
|            	| 0.49104       	| 0.49302      	|

### MLP Regression

| Parameters               	| Private Score 	| Public Score 	|
|--------------------------	|---------------	|--------------	|
| Solver=lbfgs, alpha=1e-5 	| 0.74715       	| 0.74326      	|
| Solver=adam,alpha=1e-3   	| 0.81787       	| 0.8206       	|

### eXtreme Gradient Boosting Regression (XGB)

| Parameters	Private Score	Public Score 	|                 	|                      	|                         	|                     	|              	|                     	|                         	|          	|              	|                         	|
|-------------------------------------	|-----------------	|----------------------	|-------------------------	|---------------------	|--------------	|---------------------	|-------------------------	|----------	|--------------	|-------------------------	|
| colsample_bytree: 1                 	| silent: 1       	| nthread: 8           	| min_child_weight: 10    	| 'n_estimators': 300 	| subsample: 1 	| learning_rate: 0.09 	| objective: 'reg:linear' 	| seed: 10 	| max_depth: 7 	| gamma: 0.	0.48663	0.48888 	|
| colsample_bytree: 0.77              	| silent: 1       	| nthread: 8           	| min_child_weight: 15    	| 0.48419	0.48659      	|              	|                     	|                         	|          	|              	|                         	|
| n_estimators: 500                   	| subsample: 0.77 	| learning_rate: 0.035 	| objective: 'reg:linear' 	|                     	|              	|                     	|                         	|          	|              	|                         	|
| seed: 11                            	| max_depth: 6    	| gamma: 0.2           	|                         	|                     	|              	|                     	|                         	|          	|              	|                         	|

### Bagging Regression + XGB

| Parameters	Private Score	Public Score 	|                 	|                             	|                                          	|                     	|              	|                     	|                         	|          	|              	|                           	|                                                       	|
|-------------------------------------	|-----------------	|-----------------------------	|------------------------------------------	|---------------------	|--------------	|---------------------	|-------------------------	|----------	|--------------	|---------------------------	|-------------------------------------------------------	|
| colsample_bytree: 1                 	| silent: 1       	| nthread: 8                  	| min_child_weight: 10                     	| 'n_estimators': 300 	| subsample: 1 	| learning_rate: 0.09 	| objective: 'reg:linear' 	| seed: 10 	| max_depth: 7 	| gamma: 0. n_estimators=10 	| random_state=np.random.RandomState(2018)	0.48459	0.4871 	|
| colsample_bytree: 0.77              	| silent: 1       	| nthread: 8                  	| min_child_weight: 15                     	| 0.48379	0.48613      	|              	|                     	|                         	|          	|              	|                           	|                                                       	|
| n_estimators: 500                   	| subsample: 0.77 	| learning_rate: 0.035        	| objective: 'reg:linear'                  	|                     	|              	|                     	|                         	|          	|              	|                           	|                                                       	|
| seed: 11                            	| max_depth: 6    	| gamma: 0.2  n_estimators=10 	| random_state=np.random.RandomState(2018) 	|                     	|              	|                     	|                         	|          	|              	|                           	|                                                       	|

### KNN

| Parameters                          	| Private Score 	| Public Score 	|
|-------------------------------------	|---------------	|--------------	|
| 128, weights="uniform", leaf_size=5 	| 0.52871       	| 0.52994      	|

### Decision Tree

| Parameters 	| Private Score 	| Public Score 	|
|------------	|---------------	|--------------	|
|            	| 0.52605       	| 0.52666      	|

## With Attribute (compared with same parameters)

| Models            	| Public Score without Attribute 	| Public Score with Attribute 	|
|-------------------	|--------------------------------	|-----------------------------	|
| Random Forest     	| 0.4911                         	| 0.53559                     	|
| AdaBoost          	| 0.49592                        	| 0.52592                     	|
| Bagging           	| 0.48613                        	| 0.51251                     	|
| Gradient Boosting 	| 0.48694                        	| 0.51322                     	|
| Linear Regression 	| 0.49301                        	| 0.5276                      	|
| Ridge             	| 0.49302                        	| 0.5276                      	|
| XGB               	| 0.48659                        	| 0.51272                     	|
| KNN               	| 0.52994                        	| 0.51984                     	|
| Decision Tree     	| 0.52666                        	| 0.52998                     	|

