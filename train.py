#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import sys, getopt
import time

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import *
from prepare_data import read_train_data, newSparkSession

# constants - replace netID
model_file = "final_project/models/model_{}"
SEED = 10


def train(trainingDataName):
    spark = newSparkSession()
    train, val, _ = read_train_data(trainingDataName)  # eg: trainingDataName = '5_percent'

    # select actual predictions by user
    users = val.select('user_id').distinct()
    #w = Window.partitionBy('user_id').orderBy(col('rating').desc())
    actual_preds = val.select('user_id', 'book_id').orderBy(col('user_id'), expr('rating DESC')).groupBy(
        'user_id').agg(expr("collect_list(book_id) as books"))

    # ---------------------
    # HYPERPARAMETER SEARCH
    # ---------------------
    rank = [20]
    regParam = [0.01]

    paramGrid = itertools.product(rank, regParam)

    for rank, regParam in paramGrid:
        print(rank, regParam)

        t_start = time.process_time()

        als = ALS(rank=rank, maxIter=10, regParam=regParam, userCol='user_id', itemCol='book_id', seed=SEED,
                  ratingCol='rating')

        als_model = als.fit(train)
        print('done fit')


        # predict
        recs = als_model.recommendForUserSubset(users, 500)

        pred = recs.select('user_id', 'recommendations.book_id')
        pred_rdd = pred.join(actual_preds, 'user_id', 'inner').rdd.map(lambda row: (row[1], row[2]))

        # metrics
        ranking_metrics = RankingMetrics(pred_rdd)
        precision50 = ranking_metrics.precisionAt(50)
        #precisionK = ranking_metrics.precisionAt(500)
        #map_met = ranking_metrics.meanAveragePrecision
        #ndcg = ranking_metrics.ndcgAt(500)

        t_end = time.process_time()

        time_diff = t_end - t_start

        modelFileName = saveModel(als_model)

        results = dict({'rank': rank,'regParam': regParam,'p50': precision50,
                        'subsample': trainingDataName,
                        'modelFileName': modelFileName,
                        'timeElapsed': time_diff})

        print(results)


# helper functions
def getTimeStamp():
    timestamp = time.strftime("%Y%m%d%H%M%S")
    return timestamp


def saveModel(model):
    fileName = model_file.format(getTimeStamp())
    model.save(fileName)
    return fileName


def main(argv):
    opts, args = getopt.getopt(argv, 't:')

    for option, arg in opts:
        if option == '-t':
            train(str(arg))


if __name__ == "__main__":
    main(sys.argv[1:])
