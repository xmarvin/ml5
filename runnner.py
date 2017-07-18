import sys
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from save_module import *
from prepare_data import *

from my_xgb import *
from my_lgb import *
#import my_nnet
from my_svm import *
from my_rf import *
from my_knn import *
from my_ridge import *
from my_lr import *
import pdb

from my_fold import *

def get_data_by_level(level):
  if level == 1:
    return get_only_train()
  elif level == 2:
    return get_only_train_2lv()  

def run_first_level(algorithm, fname):
  train, train_y = get_data_by_level(1)
  order_idx, x_train1, y_train1, x_train2, y_train2 = get_first_split(train, train_y)
  pred2 = algorithm.predict_with_kfold(x_train1, y_train1, x_train2, 5)
  pred1 = algorithm.predict_with_kfold(x_train2, y_train2, x_train1, 5)
  res = np.concatenate((pred1, pred2), axis=0)
  res = order_split_output(order_idx, res)
  save_prediction(res, fname)

def run_predict(algorithm, fname):
  train, train_y, test = get_train_and_test()
  res = algorithm.predict_with_kfold(train, train_y, test, 10)
  save_prediction(res, fname)

def run_cv(algorithm, level):
  train, train_y = get_data_by_level(level)
  print(train.head())
  n = 10
  scores = []
  for i in range(n):
    x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=43+i)
    y_pred = algorithm.predict_with_kfold(x_train, y_train, x_valid, 5)
    scores.append(log_loss(y_valid, y_pred))
  
  scores = np.array(scores)
  print("min = {0}, max = {1}, avg = {2}".format(scores.min(), scores.max(), scores.mean()))

def hyperopt(algorithm, level):
  train, train_y = get_data_by_level(level)  
  x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=42)
  algorithm.run_hyperopt(x_train, x_valid, y_train, y_valid)

if __name__ == "__main__":
  module_name = sys.argv[1]  
  algorithm = globals()[module_name]()
  print(module_name)
  if sys.argv[2] == 'predict':
    print("Predict:")
    name = "out/{0}_kfold{1}.csv".format(module_name, 10)
    run_predict(algorithm, name)
  elif sys.argv[2] == '1lv':
    print("1lv train:")
    name = "1lv/{0}_train.csv".format(module_name)
    run_first_level(algorithm, name)
  elif sys.argv[2] == 'cv2':
    run_cv(algorithm, 2)
  elif sys.argv[2] == 'hyper':
    print("Hyperopt:")
    hyperopt(algorithm, 1)
  elif sys.argv[2] == 'hyper2':
    print("Hyperopt - 2 level:")
    hyperopt(algorithm, 2)    
  else:
    print("CV:")    
    run_cv(algorithm, 1)