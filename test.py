#!/usr/bin/env python

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('final_project').getOrCreate()

from datetime import datetime
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import *
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from prepare_data import read_train_data, newSparkSession

model_file = 'final_project/models/model_20200511214059'
test_data_path = 'hdfs:/user/ab8690/bd_project/model_data/1_percent/test_interactions'
SEED = 10

print(datetime.now())
test_data = spark.read.parquet(test_data_path)

testusers = test_data.select('user_id').distinct()
actual_preds = test_data.select('user_id', 'book_id').orderBy(col('user_id'), expr('rating DESC')).groupBy(
    'user_id').agg(expr("collect_list(book_id) as books"))

als_model = ALSModel.load(model_file)
#als = ALS(rank=20, maxIter=10, regParam=0.001, userCol='user_id', itemCol='book_id', seed=SEED,
#          ratingCol='rating')
#als_model = als.fit(test_data)
print('done fit')

recs = als_model.recommendForUserSubset(testusers, 500)
pred = recs.select('user_id', 'recommendations.book_id')
pred_rdd = pred.join(actual_preds, 'user_id', 'inner').rdd.map(lambda row: (row[1], row[2]))

ranking_metrics = RankingMetrics(pred_rdd)
regression_metrics = RegressionMetrics(pred_rdd)

#evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
#rmse = evaluator.evaluate(pred_rdd)

print('Precision at 50 : {0}'.format(ranking_metrics.precisionAt(50)))
print("RMSE: {0}".format(regression_metrics.rootMeanSquaredError()))
print(datetime.now())
