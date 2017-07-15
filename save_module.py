import pandas as pd
import numpy as np

def save_prediction(p1, name):
  sub = pd.DataFrame()
  sub['pred'] = p1
  sub.to_csv(name, index=False, header=False)