import pandas as pd
import numpy as np

import operator
import sys
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from operator import add
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import pdb
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from save_module import *
from prepare_data import *
from my_fold import *
import random

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

random.seed(0)

def my_process_test(t):  
  return t.as_matrix()

def data():
  train, train_y = get_only_train()
  train = train.as_matrix()
  x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=42)
  return (x_train, x_valid, y_train, y_valid)

def model(x_train, x_valid, y_train, y_valid):
  model = Sequential()
  model.add(Dense({{choice([100,200,300,400,500])}}, input_dim=16, kernel_initializer='normal', activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.2759433172288843))
  model.add(Dense({{choice([100,200,300,400,500])}}, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.32))
  model.add(Dense({{choice([100,200,300,400,500])}}, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.1))
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer={{choice(['rmsprop', 'adadelta', 'sgd'])}}, metrics=['accuracy'])
  model.fit(x_train, y_train,
    batch_size={{choice([64, 128, 256,512,1024])}},
    epochs=10,
    verbose=2,
    validation_data=(x_valid, y_valid))

  score, acc = model.evaluate(x_valid, y_valid, verbose=0)
  print('Test accuracy:', acc)
  return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def run_hyperopt():
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          rseed=0,
                                          trials=Trials())

    print("344234234324324")
    x_train, x_valid, y_train, y_valid = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_valid, y_valid))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

def nnet_model():
  model = Sequential()
  model.add(Dense(500, input_dim=16, kernel_initializer='normal', activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))
  model.add(Dense(500, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))
  model.add(Dense(500, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
  return model

def my_train(x_train, x_val, y_train, y_val, nfold):
  bst_model_path = "nnet_weights_full_{0}.h5".format(nfold)
  model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
  model = nnet_model()
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  model.fit(my_process_test(x_train), y_train, validation_data=(my_process_test(x_val), y_val),
    shuffle=True, batch_size=128, callbacks=[early_stopping, model_checkpoint], epochs=40)
  model.load_weights(bst_model_path)
  return model

def my_predict_proba(model, x):
  return model.predict_proba(x)

def my_handle_output(res):
  return np.array([r[0] for r in res])

def run_predict():
  train, train_y, test = get_train_and_test()
  res = predict_kfold(sys.modules[__name__], train, train_y, test, 5)
  save_prediction(res,'out/tt_nnet_kfold5.csv')

def run_cv(): #0.539100927473
  train, train_y = get_only_train()
  x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=42)
  y_pred = predict_kfold(sys.modules[__name__], x_train, y_train, x_valid, 5)
  print("cv: {0}".format(log_loss(y_valid, y_pred)))

run_predict()
#run_cv()
#run_hyperopt()