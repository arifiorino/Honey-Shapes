import numpy as np
from utils import *

name='A2'
def addTrainingData(context,actions,losses,train_x,train_y):
  train_x.append(context+actions)
  train_y.append([sum(losses)])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  test_x=allSequences(len(context),len(options))
  test_x=[context+[options[i] for i in a] for a in test_x]
  mean, stdev = GP(train_x, train_y, test_x)
  best_x = test_x[np.argmin(mean-ucb_mult*stdev)][len(context):]
  return best_x

