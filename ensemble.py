import pandas as pd
import numpy as np
import operator
from save_module import *

pred1 = pd.read_csv('out/my_xgb_lgb.csv',header=None)
pred2 = pd.read_csv('out/my_nnet.csv',header=None)
pred3 = pd.read_csv('out/my_rf_kfold10.csv',header=None)

final = pred1[0] * 0.55 + pred2[0] * 0.4 + pred3[0] * 0.05

save_prediction(final,"out/rf_nn_xgb.csv")