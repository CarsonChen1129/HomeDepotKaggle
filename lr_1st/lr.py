from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.linalg import SparseVector
from pyspark.sql.functions import udf


    
if __name__ == '__main__':
    vs = VectorAssembler(inputCols=["query_len", "common_title", "common_descriptoin"],outputCol='features')
    train_lr = vs.transform(trainingData)
    lr = LinearRegression(featuresCol='features', regParam=0.3, elasticNetParam=0.8,labelCol='relevance',maxIter=1000)
    model = lr.fit(train_lr)
    lr_summary = model.summary
    print("Coefficients: %s" % str(model.coefficients))
    print("Intercept: %s" % str(model.intercept))
    print("numIterations: %d" % lr_summary.totalIterations)
    print("objectiveHistory: %s" % str(lr_summary.objectiveHistory))
    print("RMSE: %f" % lr_summary.rootMeanSquaredError)
    print("r2: %f" % lr_summary.r2)
