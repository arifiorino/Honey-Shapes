import numpy as np
from utils import *

name='A4'
def addTrainingData(context,actions,losses,train_x,train_y):
  K=len(actions)-len(losses)
  for i in range(len(losses)):
    train_x.append(context[i:i+K+1]+actions[i:i+K+1])
    train_y.append([losses[i]])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  K=len(train_x[0])//2-1
  L=len(context)-K
  N=len(options)
  test_x=allSequences(K+1,N)
  test_x=[context[:K+1]+[options[i] for i in a] for a in test_x]
  mean, stdev = GP(train_x, train_y, test_x)
  best_x=test_x[np.argmin(mean-ucb_mult*stdev)][K+1:]
  for i in range(K+1,len(context)):
    test_x=[context[i-K:i+1]+best_x[-K:]+[options[a]] for a in range(N)]
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(options[np.argmin(mean-ucb_mult*stdev)])
  return best_x

