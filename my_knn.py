import pandas as pd
import numpy as np
import operator
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA

import sys

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from prepare_data import *
from sklearn.grid_search import GridSearchCV


def current_params():
  params = {       
       'n_neighbors': 30,
       'leaf_size': 20,
       'algorithm': 'auto',
       'n_jobs': -1
  }       
  return(params)

def my_train(x_train, x_val, y_train, y_val, nfold,rf_params=current_params()):
  clf = KNeighborsClassifier(**rf_params)
  clf.fit(x_train, y_train)
  return clf


def my_predict_proba(model, x):
  return model.predict_proba(x)

def my_process_test(t):  
  return t

def my_handle_output(res):
  return np.array([r[1] for r in res])

def hyperopt_score(rf_params):
  bst = my_train(x_train, x_valid, y_train, y_valid, 0, rf_params)
  dtest = my_process_test(x_valid)
  y_pred = my_predict_proba(bst, dtest)
  return log_loss(y_valid, y_pred)

def run_hyperopt():
  space = {       
       'n_neighbors': [5,10,15,20,25,30],
       'algorithm': ['auto', 'ball_tree', 'kd_tree'],
       'leaf_size': [20,30,40,50]       
       }
  train, train_y = get_only_train()  
  clf = KNeighborsClassifier({'random_state': 0})
  grid_search = GridSearchCV(clf, space, n_jobs=-1, cv=2)
  grid_search.fit(train, train_y)
  print(grid_search.best_params_)

#run_hyperopt()