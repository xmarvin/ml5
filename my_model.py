from sklearn.metrics import log_loss
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from my_fold import *

class MyModel:
  def current_params(self):  
    raise Exception('Imlement me')

  def my_train(self,x_train, x_val, y_train, y_val, nfold, params = None):
    if params == None:
      params = self.current_params()

    return self.train_with_params(x_train, x_val, y_train, y_val, nfold, params)

  def train_with_params(self,x_train, x_val, y_train, y_val, nfold, params):
    return None

  def my_predict_proba(self,model, x):
    return model.predict(x)

  def my_process_test(self,t):  
    return t

  def my_handle_output(self,res):
    return res

  def predict_with_kfold(self, x_train, y_train, x_valid, nfolds, params = None):  
    if params == None:
      params = self.current_params()

    return predict_kfold(self, x_train, y_train, x_valid, nfolds, params)

  def hyperopt_score(self, params):
    y_pred = self.predict_with_kfold(self.x_train, self.y_train, self.x_valid, 1, params)
    return log_loss(self.y_valid, y_pred)    

  def hyperopt_space(self):
    raise Exception('Imlement me')

  def run_hyperopt(self, x_train, x_valid, y_train, y_valid):
    self.x_train = x_train
    self.x_valid = x_valid
    self.y_train = y_train
    self.y_valid = y_valid
    best = fmin(self.hyperopt_score, self.hyperopt_space(),
      algo=tpe.suggest, max_evals=1000)
    self.x_train = None
    self.x_valid = None
    self.y_train = None
    self.y_valid = None
    print(best)    