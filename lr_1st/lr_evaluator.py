from pyspark.ml.evaluation import RegressionEvaluator
    
if __name__ == '__main__':
    evaluator = RegressionEvaluator(labelCol="relevance", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predict_lr)
    print("RMSE = %g" % rmse)