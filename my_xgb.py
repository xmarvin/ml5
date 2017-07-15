import pandas as pd
import numpy as np
import operator
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA
from save_module import *
from prepare_data import *
from my_fold import *
import sys

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def xgb_current_params():
  params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'colsample_bytree': 0.9,
    'eta': 0.025,
    'gamma': 0.6,
    'subsample': 0.75,
    'max_depth': 5,
    'silent': 1,
    'min_child_weight': 5.0,
    'tree_method': 'exact'
  }
  return(params)

def my_train(x_train, x_valid, y_train, y_valid, nfold, xgb_params=xgb_current_params()):
  d_train = xgb.DMatrix(x_train, label=y_train)
  d_valid = xgb.DMatrix(x_valid, label=y_valid)
  watchlist = [(d_train, 'train'), (d_valid, 'valid')]
  bst = xgb.train(xgb_params, d_train, 1000, watchlist, early_stopping_rounds=10, verbose_eval=100)
  return bst

def my_predict_proba(bst, d_test):  
  p_test = bst.predict(d_test, ntree_limit=bst.best_iteration)
  return p_test

def my_process_test(t):
  return xgb.DMatrix(t)

def run_second_cv():
  xgb_fl = pd.read_csv('out/xgb_train.csv',header=None)
  xgb_fl6 = pd.read_csv('out/xgb6_train.csv',header=None)
  knn_fl = pd.read_csv('out/knn_train.csv',header=None)
  train['xgb_fl'] = xgb_fl[0]
  train['xgb_fl6'] = xgb_fl6[0]
  train['knn_fl'] = knn_fl[0]
  print(train.head()) 
  print(cv_score2(train, train_y))

def my_handle_output(res):
  return res

def run_first_level():
  x_train1, y_train1, x_train2, y_train2 = get_first_split(train, train_y)
  res1 = predict_kfold(sys.modules[__name__], x_train1, y_train1, x_train2, 10)
  res2 = predict_kfold(sys.modules[__name__], x_train2, y_train2, x_train1, 10)
  pred1 = res2
  pred2 = res1
  res = np.concatenate((pred1, pred2), axis=0)
  save_prediction(res, 'out/xgb6_train.csv')


def hyperopt_score(params):
  bst = my_train(x_train, x_valid, y_train, y_valid, 0, params)
  dtest = my_process_test(x_valid)
  y_pred = my_predict_proba(bst, dtest)
  return log_loss(y_valid, y_pred)

#train, train_y = get_only_train()
#x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=42)

def run_hyperopt():
  space = {      
      'eta': hp.quniform('eta', 0.005, 0.1, 0.025),
      # A problem with max_depth casted to float instead of int with
      # the hp.quniform method.
      'max_depth':  hp.choice('max_depth', [4,5,6,7]),
      'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
      'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
      'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
      'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
      'eval_metric': 'logloss',
      'objective': 'binary:logistic',      
      'booster': 'gbtree',
      'tree_method': 'exact',
      'silent': 1,
      'seed': 0
  }
  # Use the fmin function from Hyperopt to find the best hyperparameters  

  best = fmin(hyperopt_score, space, algo=tpe.suggest, max_evals=1000)
  print(best)


if __name__ == "__main__":
  run_hyperopt()