#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, getopt, random
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

# CONSTANTS -- replace NetID in path
source_path = 'hdfs:/user/ab8690/final_project/all_data/{}'
train_path = 'hdfs:/user/ab8690/final_project/train_data/{}/{}'
SEED = 10

def csv_to_parquet(sparkSession=None):
    ''' Reads source data and writes to user hdfs '''
    spark = sparkSession or newSparkSession()

    filePath = 'hdfs:/user/bm106/pub/goodreads'
    fileNames = ['goodreads_interactions.csv', \
                 'user_id_map.csv', \
                 'book_id_map.csv']

    for fileName in fileNames:
        table = spark.read.csv("{}/{}".format(filePath, fileName), header=True, inferSchema=True)
        table.write.parquet(source_path.format(fileName.split(".")[0]))


def save_train_data(name, train_interactions, val_interactions, test_interactions):
    train_interactions.write.parquet(train_path.format(name, 'train_interactions'))
    val_interactions.write.parquet(train_path.format(name, 'val_interactions'))
    test_interactions.write.parquet(train_path.format(name, 'test_interactions'))


def read_train_data(name, sparkSession=None):
    spark = sparkSession or newSparkSession()
    train_int = spark.read.parquet(
        train_path.format(name, 'train_interactions'))
    val_int = spark.read.parquet(
        train_path.format(name, 'val_interactions'))
    test_int = spark.read.parquet(
        train_path.format(name, 'test_interactions'))
    return (train_int, val_int, test_int)


def random_split_users_int(interactions, users, splitProportion):
    """ Returns a split of the interactions of the users received.
        Used to include interactions from validation and test users in the training set. """

    w = Window.partitionBy('user_id').orderBy('book_id')
    ranked_interactions = interactions.join(users, 'user_id', 'leftsemi').\
        select("user_id", "book_id", "rating", F.percent_rank().over(w).alias("percent_rank"))

    split1 = ranked_interactions.filter(ranked_interactions.percent_rank <= splitProportion).drop('percent_rank')
    split2 = ranked_interactions.filter(ranked_interactions.percent_rank > splitProportion).drop('percent_rank')

    return (split1, split2)


def train_val_test_split(user_ids, int):
    # data_file is parquet file name

    train_users, val_users, test_users = user_ids.randomSplit([0.6, 0.2, 0.2], seed=SEED)

    # create training dataset
    train_set = int.join(train_users, 'user_id', 'leftsemi')

    # validation set
    valid_train_int, valid_set = random_split_users_int(val_users, int)
    train_set = train_set.union(valid_train_int)

    # test set
    test_train_int, test_set = random_split_users_int(test_users, int)
    train_set = train_set.union(test_train_int)

    return (train_set, valid_set, test_set)


def generate_train_data(s_size=.01, sparkSession=None):

    # set random seed and start spark session
    random.seed(SEED)
    spark = sparkSession or newSparkSession()

    # read interactions data
    interactions = spark.read.parquet(source_path.format('goodreads_interactions'))
    interactions = interactions.filter('rating > 0').drop('is_read').drop('is_reviewed')

    # remove users with less than 10 interactions
    interactions = interactions.join(interactions.groupBy(int.user_id).agg({'book_id': 'count'}).
                                     filter('count(book_id) >= 10'), 'user_id', 'leftsemi')

    users = interactions.select('user_id').distinct()
    
    if s_size > 0:
        users = users.sample(False, s_size, seed=SEED)
        interactions = interactions.join(users, 'user_id', 'leftsemi')

    interactions_split = train_val_test_split(users, interactions)
    save_train_data("{}_percent".format(float(s_size * 100)), *interactions_split)


def newSparkSession():
    # return SparkSession.builder.getOrCreate()
    mem = "7GB"
    spark = (SparkSession.builder.appName("BD_Project")
             .master("yarn")
             .config("sparn.executor.memory", mem)
             .config("sparn.driver.memory", mem)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark



def main(argv):
    opts, args = getopt.getopt(argv, 'lt:r:')

    for option, arg in opts:
        if option == '-l':
            csv_to_parquet()
        if option == '-t':
            generate_train_data(s_size=float(arg))


if __name__ == "__main__":
    main(sys.argv[1:])



