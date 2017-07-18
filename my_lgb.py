import pandas as pd
import numpy as np
import operator
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA
import sys

from my_model import MyModel

class MyLgbModel(MyModel):

  def current_params(self):
    params = {
      'task': 'train',
      'boosting_type': 'gbdt',
      'objective': 'binary',
      'metric': 'binary_logloss',
      'bagging_freq': 4,
      'num_leaves': 25,
      'bagging_fraction': 0.9,
      'feature_fraction': 0.95,
      'learning_rate': 0.025,

      'verbose': 0
    }
    return(params)

  def train_with_params(self,x_train, x_valid, y_train, y_valid, nfold, params):
    d_train = lgb.Dataset(x_train.values, y_train)
    d_valid = lgb.Dataset(x_valid.values, y_valid, reference=d_train)  
    bst = lgb.train(params, d_train, num_boost_round = 10000, valid_sets=d_valid, early_stopping_rounds=10)
    return bst

  def my_predict_proba(self,bst, d_test):  
    p_test = bst.predict(d_test, num_iteration=bst.best_iteration)
    return p_test

  def my_process_test(self,t):
    return t.values

  def hyperopt_score(self,params):
    bst = my_train(x_train, x_valid, y_train, y_valid, 0, params)
    dtest = my_process_test(x_valid)
    y_pred = my_predict_proba(bst, dtest)
    return log_loss(y_valid, y_pred)

  def hyperopt_scope(self):
    space = {
      'task': 'train',
      'boosting_type': 'gbdt',
      'objective': 'binary',
      'metric': 'binary_logloss',
      'num_leaves': hp.choice('num_leaves', [10,20,25,30,31,35,40,45,50]),
      'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
      'feature_fraction': hp.quniform('feature_fraction', 0.5, 1, 0.05),
      'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1, 0.05),
      'bagging_freq': hp.choice('bagging_freq', [3,4,5,6,7,8,10])
    }
    return space