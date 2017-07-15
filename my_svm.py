import sys
import pandas as pd
import numpy as np
import pdb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import svm
from save_module import *
from prepare_data import *
from my_fold import *

train, train_y = get_train()
test = get_test()

def my_train(x_train, x_val, y_train, y_val, kfold, param1=3):
  clf = svm.SVC(kernel='rbf', probability=True, C=param1)
  clf.fit(x_train.as_matrix(), y_train) 
  return clf

def my_predict(neigh, x_val):
  return neigh.predict(x_val)

def my_predict_proba(neigh, x_val):
  return neigh.predict_proba(x_val)

def my_process_test(t):
  return t.as_matrix()

def my_handle_output(res):
  return np.array([r[1] for r in res])  