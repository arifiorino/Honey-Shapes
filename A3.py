import numpy as np
from utils import *

name='A3'
def addTrainingData(context,actions,losses,train_x,train_y):
  K=len(actions)-len(losses)
  for i in range(len(losses)):
    train_x.append(context[i:i+K+1]+actions[i:i+K+1])
    train_y.append([losses[i]])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  K=len(train_x[0])//2-1
  L=len(context)-K
  N=len(options)
  seq=allSequences(len(context),N)
  seq=[[options[i] for i in a] for a in seq]
  test_x=[context[i:i+K+1]+a[i:i+K+1] for a in seq for i in range(L)]
  mean, stdev = GP(train_x, train_y, test_x)
  best_idx=None
  best=1e10;
  for idx in range(N**len(context)):
    ucb=np.sum(mean[idx*L:idx*L+L]-ucb_mult*stdev[idx*L:idx*L+L])
    if ucb<best:
      best=ucb
      best_idx=idx
  return seq[best_idx]

