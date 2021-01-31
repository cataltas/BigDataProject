import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import sklearn
from skopt import forest_minimize
import time
from prepare_data import read_train_data, get_user, newSparkSession


MODEL_PATH_TEMPLATE = "final_project/models/model_{}"
RESULTS_PATH_TEMPLATE = "/home/{}/final-project-bd-final-project/results/als_{}_{}.json"
SEED = 10


def sparse_train_test(data_train,data_test):
    data_train=data_train.select("*").toPandas()
    data_test = data_test.select("*").toPandas()
    train_df = pd.DataFrame(data_train,columns=["user_id","book_id","rating"])
    test_df = pd.DataFrame(data_test,columns=["user_id","book_id","rating"])
    print(train_df["user_id"].index)
    sparse_train= csr_matrix((np.array(train_df["rating"]), (np.array(train_df["user_id"].index), np.array(train_df["book_id"].index))))
    sparse_test = csr_matrix((np.array(test_df["rating"]), (np.array(test_df["user_id"].index), np.array(test_df["book_id"].index))),shape=(len(train_df["user_id"]),len(train_df["book_id"])))
    return sparse_train,sparse_test

def optimal(params):

    data_train_opt,data_val,_ = read_train_data("1_percent")
    train_opt,val = sparse_train_test(data_train,data_val)

    rank, alpha = params
    model = LightFM(loss='warp', learning_rate=learning_rate,item_alpha = alpha, user_alpha = alpha)
    model.fit(train_opt, epochs=20, verbose=True)
    precision = precision_at_k(model, val)
    precision_mean = np.mean(precision)
    return -precision_mean

def main():
    
    # find optimal learning rate and rank
    bounds = [(10**-4, 1.0, 'log-uniform'),(10**-6, 10**-1, 'log-uniform')]
    opt_params = forest_minimize(optimal, bounds, verbose=True)
    opt_lr,opt_rank=opt_params.x[0],opt_params.x[1]

    # times and precisions for 3 data sets with 
    data_1_train,_,data_1_test = read_train_data("0.1_percent")
    data_2_train,_, data_2_test = read_train_data("0.5_percent")
    data_3_train,_, data_3_test = read_train_data("1_percent")
    
    dataset= [[data_1_train,data_1_test],[data_2_train,data_2_test],[data_3_train,data_3_test]]
    times =[]
    precisions=[]
    for data in dataset:
        train,test = sparse_train_test(data[0],data[1])
        start = time.time()
        model = LightFM(loss='warp', learning_rate=opt_lr,no_components=opt_rank)
        model.fit(train, epochs=10, verbose=True)
        precision = precision_at_k(model, test, k=50).mean()
        sec = (time.time() - start)/60
        print("TIME:",sec,"PREC:", precision)
        times.append(sec)
        precisions.append(np.mean(precision))
    
    print("times:", times, "precisions", precisions)

if __name__ == "__main__":

    main()





