import sys
import pandas as pd
import numpy as np
import pdb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import svm
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from my_model import MyModel

class MySvmModel(MyModel):

  def current_params(self):
    res = {
      'kernel':'rbf',
      'probability': True,
      'C': 3
    }
    return res

  def train_with_params(self, x_train, x_val, y_train, y_val, kfold, params):
    clf = svm.SVC(**params)
    clf.fit(x_train.as_matrix(), y_train) 
    return clf

  def my_predict(self, neigh, x_val):
    return neigh.predict(x_val)

  def my_predict_proba(self, neigh, x_val):
    return neigh.predict_proba(x_val)

  def my_process_test(self, t):
    return t.as_matrix()

  def my_handle_output(self, res):
    return np.array([r[1] for r in res])

  def hyperopt_space(self):
    space = {
       'C': hp.choice('C', [0.01,0.3,1,3,10]),
       'probability': True,
       'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
     }
    return space      