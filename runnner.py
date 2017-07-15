import sys
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from save_module import *
from prepare_data import *
from my_fold import *

import my_xgb
import my_lgb
#import my_nnet
import my_svm
import my_rf
import my_knn
import pdb

def run_first_level(module, fname):
  train, train_y = get_only_train()
  x_train1, y_train1, x_train2, y_train2 = get_first_split(train, train_y)
  res1 = predict_kfold(module, x_train1, y_train1, x_train2, 10)
  res2 = predict_kfold(module, x_train2, y_train2, x_train1, 10)
  pred1 = res2
  pred2 = res1
  res = np.concatenate((pred1, pred2), axis=0)
  save_prediction(res, fname)

def run_predict(module, fname):
  train, train_y, test = get_train_and_test()
  res = predict_kfold(module, train, train_y, test, 10)
  save_prediction(res, fname)

def run_cv(module):
  train, train_y = get_only_train()
  n = 10
  scores = []
  for i in range(n):
    x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=43+i)
    y_pred = predict_kfold(module, x_train, y_train, x_valid, 5)
    scores.append(log_loss(y_valid, y_pred))
  
  scores = np.array(scores)
  print("min = {0}, max = {1}, avg = {2}".format(scores.min(), scores.max(), scores.mean()))


def run_second_cv(module):
  train_df, train_y = get_only_train()
  xgb_fl = pd.read_csv('1lv/my_xgb_train.csv',header=None)
  rf_fl = pd.read_csv('1lv/my_rf_train.csv',header=None)
  knn_fl = pd.read_csv('1lv/my_knn_train.csv',header=None)
  nnet_fl = pd.read_csv('1lv/my_nnet_train.csv',header=None)

  train = pd.DataFrame()
  train['xgb_fl'] = xgb_fl[0]
  train['knn_fl'] = knn_fl[0]
  train['rf_fl'] = rf_fl[0]
  train['nnet_fl'] = nnet_fl[0]
  print(train.head()) 
  
  n = 10
  scores = []
  for i in range(n):
    x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=43+i)
    y_pred = predict_kfold(module, x_train, y_train, x_valid, 5)    
    scores.append(log_loss(y_valid, y_pred))
  
  scores = np.array(scores)
  print("2cv: min = {0}, max = {1}, avg = {2}".format(scores.min(), scores.max(), scores.mean()))

if __name__ == "__main__":
  module_name = sys.argv[1]
  module = sys.modules[module_name]
  print(module_name)
  if sys.argv[2] == 'predict':
    print("Predict:")
    name = "out/{0}_kfold{1}.csv".format(module_name, 10)
    run_predict(module, name)
  elif sys.argv[2] == '1lv':
    name = "1lv/{0}_train.csv".format(module_name)
    run_first_level(module, name)
  elif sys.argv[2] == '2cv':
    run_second_cv(module)
  else:
    print("CV:")    
    run_cv(module)