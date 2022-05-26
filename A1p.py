import numpy as np
name='A1p'
def addTrainingData(context,actions,losses,train_x,train_y):
  if len(train_x)==0:
    train_x.extend([[] for _ in range(len(losses))])
    train_y.extend([[] for _ in range(len(losses))])
  K=len(actions)-len(losses)
  for i in range(len(losses)):
    train_x[i].append([context[i+K],actions[i+K]])
    train_y[i].append([losses[i]])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  K=len(context)-len(train_x)
  best_x=[]
  for i in range(len(context)):
    test_x=[[context[i],x] for x in options]
    # no other option for 0 - (K-1), pick 0
    mean, stdev = GP(train_x[max(0,i-K)], train_y[max(0,i-K)], test_x)
    best_x.append(options[np.argmin(mean-ucb_mult*stdev)])
  return best_x

