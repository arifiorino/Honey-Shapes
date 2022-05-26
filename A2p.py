import numpy as np
from utils import *

name='A2p'
def addTrainingData(context,actions,losses,train_x,train_y):
  K=len(actions)-len(losses)
  for i in range(len(losses)):
    train_x.append(actions[i:i+K+1])
    train_y.append([losses[i]])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  K=len(train_x[0])
  L=len(context)-K
  N=len(options)
  test_x=allSequences(K,len(options))
  test_x=[[options[i] for i in a] for a in test_x]
  mean, stdev = GP(train_x, train_y, test_x)
  best_idx=None
  best=1e10;
  for idx in range(N**len(context)):
    ucb=sum([mean[(idx//N**i)%(N**K)]-ucb_mult*stdev[(idx//N**i)%(N**K)] for i in range(L)])
    if ucb<best:
      best=ucb
      best_idx=idx
  best_x=[(best_idx//N**i)%N for i in range(len(context))]
  return best_x

