import numpy as np
from utils import *

name='A4p'
def addTrainingData(context,actions,losses,train_x,train_y):
  K=len(actions)-len(losses)
  for i in range(len(losses)):
    train_x.append(context[i:i+K+1]+actions[i:i+K+1])
    train_y.append([losses[i]])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  K=len(train_x[0])//2-1
  A=len(context)
  L=A-K
  N=len(options)
  test_x=allSequences(K+1,N)
  test_x=[context[:K+1]+[options[i] for i in a] for a in test_x]
  mean, stdev = GP(train_x, train_y, test_x)
  best_x=test_x[np.argmin(mean-ucb_mult*stdev)][K+1:]
  for i in range(K+1,A):
    test_x=[context[i-K:i+1]+best_x[-K:]+[options[a]] for a in range(N)]
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(options[np.argmin(mean-ucb_mult*stdev)])
  for changeI in range(A):
    starts=range(max(0,changeI-K),min(A-1-K,changeI)+1)
    test_x=[context[start:start+K+1]+best_x[start:changeI]+[x]+best_x[changeI+1:start+K+1] for x in options for start in starts]
    mean, stdev = GP(train_x, train_y, test_x)
    ubcs=[sum((mean-ucb_mult*stdev)[i:i+len(starts)]) for i in range(0,len(test_x),len(starts))]
    best_x[changeI]=options[np.argmin(ubcs)]
  return best_x
