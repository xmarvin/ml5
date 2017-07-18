import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import os
import glob

NORM_PRESSURE = {
    1: {0:[116,72], 1: [120, 75], 2: [127,80], 3: [137,84], 4: [144,85], 5: [159, 85]},
    2: {0:[123,76], 1: [126, 79], 2: [129,81], 3: [135,83], 4: [142,85], 5: [142, 80]},
  }

def normilize_pressure(val):
  res = abs(val)

  if res < 20:
    res = res * 10  

  if res >= 3000:
    res = res // 100

  if res >= 300:
    res = res // 10  

  return int(res)

def get_age_range(val):
  year = val // 365
  if year <= 20:
    return 0
  elif year < 30:
    return 1
  elif year < 40:
    return 2
  elif year < 50:
    return 3
  elif year < 60: 
    return 4
  else:
    return 5  

def get_norm(row):
  return NORM_PRESSURE[row['gender']][row['age_range']]

def get_norm_hi(row):
  return get_norm(row)[0]

def get_norm_lo(row):
  return get_norm(row)[1]

def norm_diff_lo(row):  
  norm = get_norm_lo(row)
  return(abs(norm - row['ap_lo']) / norm)

def norm_diff_hi(row):
  norm = get_norm_hi(row)
  return(abs(norm - row['ap_hi']) / norm)

def process_data(df):
  cols = ['age', 'weight', 'height', 'ap_lo','ap_hi', 'im', 'w_age']
  for col in cols:
    scale = StandardScaler()
    df[col] = scale.fit_transform(df[col])
  return df

def make_features(df):
  df['gender'] = df['gender'].astype(int)
  df['age_range'] = df['age'].map(get_age_range)
  df['im'] = df['weight'] / df['height'] / df['height']
  df['ap_hi'] = df['ap_hi'].map(normilize_pressure)
  df['ap_lo'] = df['ap_lo'].map(normilize_pressure)  

  df['cholesterol'] = (df['cholesterol'] - 1)
  df['gluc'] = (df['gluc'] - 1)  

  df['ap_lo'] = df['ap_lo'].astype(int)
  df['ap_hi'] = df['ap_hi'].astype(int)
  idx = df['ap_hi'] < df['ap_lo']
  df.loc[idx,['ap_hi','ap_lo']] = df.loc[idx,['ap_lo','ap_hi']].values
  
  df['norm_diff_hi'] = df.apply(norm_diff_hi, axis=1, raw=True)
  df['norm_diff_lo'] = df.apply(norm_diff_lo, axis=1, raw=True)

  df.loc[df['ap_hi'] < 60, 'ap_hi'] = 0
  df.loc[df['ap_lo'] < 30, 'ap_lo'] = 0

  df.loc[df['height'] < 100, 'height'] = df['height'].mean()  
  df.loc[(df['weight'] > 180) | (df['weight'] < 30), 'weight'] = df['weight'].mean()
  
  #df['diff_dx'] = (df['ap_hi'] - df['ap_lo'])
  df['w_age'] = df['weight'] * df['age'] / 365
  df['gender'] = (df['gender'] - 1)
  df.drop('id', axis=1,inplace=True)
  
  df = process_data(df)
  
  return(df)

def get_only_train():
  train, train_y = get_train()
  return (train, train_y)

def get_train_and_test():
  train, train_y = get_train()
  test = get_test()
  return (train, train_y, test)

def get_train():
  train = pd.read_csv('train.csv',sep=';')  
  train_y = train['cardio'].values
  train.drop('cardio', axis=1, inplace=True)
  train = make_features(train)
  return (train, train_y)

def get_test():
  test = pd.read_csv('test.csv',sep=';')
  test.loc[test['smoke'] == 'None', 'smoke'] = '0'
  test.loc[test['alco'] == 'None', 'alco'] = '0'
  test.loc[test['active'] == 'None', 'active'] = '1'
  test['smoke'] = test['smoke'].astype(int)
  test['alco'] = test['alco'].astype(int)
  test['active'] = test['active'].astype(int)
  test = make_features(test)
  return test

def get_only_train_2lv():
  train_df, train_y = get_only_train()
  train = pd.DataFrame()
  for fname in glob.glob("1lv/*.csv"):
    name = os.path.basename(fname).split('.')[0]
    fl = pd.read_csv(fname,header=None)
    train[name] = fl[0]

  return (train, train_y)