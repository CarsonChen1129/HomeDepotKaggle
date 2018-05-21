from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
from pyspark.sql import SparkSession
# create SparkContext if sc doesn't exist.
try:
    sc
except NameError:
    sc =SparkContext()
from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import os


if __name__ == '__main__':
    spark = SparkSession.builder.appName("FinalPrj").getOrCreate()
    df = spark.read.csv("/home/yijiaj/FinalData/train_feature_base.csv",header=True,inferSchema=True)
    (trainingData, testData) = df.randomSplit([0.7, 0.3])