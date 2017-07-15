import pandas as pd
import numpy as np
import operator
from sklearn.ensemble import RandomForestClassifier
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
import my_rf
import sys

train = pd.read_csv('train.csv',sep=';')
test = pd.read_csv('test.csv',sep=';')

train.drop('cardio', axis=1, inplace=True)
train['alco'] = train['alco'].astype(str)
train['smoke'] = train['smoke'].astype(str)
train['active'] = train['active'].astype(str)
train = make_features(train)
test = make_features(test)

train_test = pd.concat((train,test))
train_test = process_data(train_test)


COLUMNS = ['age', 'weight', 'im','height', 'ap_lo', 'ap_hi', 'gender', 'gluc', 'age_range']
def fill_for_column(train_test, col):
  train_idx  = train_test[col] != 'None'
  test_idx  = train_test[col] == 'None'
  pdb.set_trace()
  res = predict_kfold(my_rf, train_test[train_idx][COLUMNS], train_test[train_idx][col].astype(int).values, train_test[test_idx][COLUMNS], 5)
  pred = [ int(a > 0.5) for a in res]
  train_test.loc[test_idx, col] = pred

fill_for_column(train_test, 'alco')
fill_for_column(train_test, 'smoke')
fill_for_column(train_test, 'active')

test_fil = train_test[train.shape[0]:]
test = pd.read_csv('test.csv',sep=';')
test['alco'] = test_fil['alco']
test['smoke'] = test_fil['smoke']
test['active'] = test_fil['active']
test['smoke'] = test['smoke'].astype(int)
test['alco'] = test['alco'].astype(int)
test['active'] = test['active'].astype(int)
test.to_csv('./stest.csv', index=False)


