import pandas as pd
import numpy as np
import operator
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
import operator
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA
from save_module import *
from prepare_data import *
from my_fold import *
import my_xgb
import my_rf
import my_knn
import my_svm
import sys


def cv_score(train, train_y):
  x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=0)
  y_pred_xgb = predict_kfold(my_xgb, x_train, y_train, x_valid, 1)
  y_pred_rf = predict_kfold(my_rf, x_train, y_train, x_valid, 1)
  y_pred_svm = predict_kfold(my_svm, x_train, y_train, x_valid, 1)
  y_pred_knn = predict_kfold(my_knn, x_train, y_train, x_valid, 1)
  #y_pred_nnet = predict_kfold(my_nnet, x_train, y_train, x_valid, 5)
  save_prediction(y_pred_xgb, 'cv_y_pred_xgb.csv')
  save_prediction(y_pred_rf, 'cv_y_pred_rf.csv')
  save_prediction(y_pred_svm, 'cv_y_pred_svm.csv')
  save_prediction(y_pred_knn, 'cv_y_pred_knn.csv')

  options = [
              [0.25,0.25,0.25,0.25],
              [0.5,0.25,0.15,0.1],
              [0.6,0.15,0.15,0.1],
              [0.6,0.2,0.15,0.05],
              [0.7,0.1,0.15,0.05],
              [0.7,0.15,0.1,0.05],
              [0.8,0.1,0.05,0.05],
              [0.85,0.05,0.05,0.05],
              [0.9,0.05,0.025,0.025],
              [0.95,0.05,0,0]
            ]
  df = {}
  for kofs in options:
    kof1,kof2,kof3,kof4 = kofs
    y_pred = (y_pred_xgb * kof1) + (y_pred_rf * kof2) + (y_pred_svm * kof3) + (y_pred_knn * kof4)
    df["{0}-{1}-{2}-{3}".format(kof1,kof2,kof3,kof4)] = [log_loss(y_valid, y_pred)]

  df = sorted(df.items(), key=operator.itemgetter(1))
  return(df)

def run_cv():
  train, train_y = get_only_train()
  print(cv_score(train, train_y))

#run_cv()
train, train_y = get_only_train()
x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=0)
y_pred_xgb = pd.read_csv('cv_y_pred_xgb.csv',header=None)
y_pred_rf = pd.read_csv('cv_y_pred_rf.csv',header=None)
y_pred_svm = pd.read_csv('cv_y_pred_svm.csv',header=None)
y_pred_knn = pd.read_csv('cv_y_pred_knn.csv',header=None)
options = [            
            [0.8,0.1,0.05,0.05],
            [0.9,0.05,0.025,0.025],
            [0.95,0.05,0,0],
            [1,0,0,0]
          ]
df = {}
for kofs in options:
  kof1,kof2,kof3,kof4 = kofs
  y_pred = (y_pred_xgb * kof1) + (y_pred_rf * kof2) + (y_pred_svm * kof3) + (y_pred_knn * kof4)
  df["{0}-{1}-{2}-{3}".format(kof1,kof2,kof3,kof4)] = [log_loss(y_valid, y_pred)]

df = sorted(df.items(), key=operator.itemgetter(1))
print(df)