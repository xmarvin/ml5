import pandas as pd
import numpy as np
import operator
from sklearn.linear_model import BayesianRidge

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

from my_model import MyModel

class MyRidgeModel(MyModel):

  def current_params(self):
    res = {
      'alpha_2': 148.40775713186514,
      'alpha_1': 0.13588328378990333,
      'n_iter': 400,
      'normalize': False
    }
    return res

  def train_with_params(self,x_train, x_val, y_train, y_val, nfold, params):  
    clf = BayesianRidge(**params)
    clf.fit(x_train, y_train)
    return clf

  def my_predict_proba(self,model, x):
    return model.predict(x)

  def my_process_test(self, t):  
    return t

  def my_handle_output(self, res):
    return res

  def hyperopt_space(self):
    space = {       
         'n_iter': hp.choice('n_iter', [100,200,300,400,500,800,1000,2000]),
         'normalize': hp.choice('normalize', [True, False]),
         'alpha_1': hp.loguniform('alpha_1', -7, 5),
         'alpha_2': hp.loguniform('alpha_2', -7, 5),
         }
    return space