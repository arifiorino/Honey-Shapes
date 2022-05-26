import numpy as np
name='A1'
def addTrainingData(context,actions,losses,train_x,train_y):
  K=len(actions)-len(losses)
  for i in range(len(losses)):
    train_x.append([context[i+K],actions[i+K]])
    train_y.append([losses[i]])

def findUCB(context,options,train_x,train_y,GP,ucb_mult):
  best_x=[]
  for i in range(len(context)):
    test_x=[[context[i],x] for x in options]
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(options[np.argmin(mean-ucb_mult*stdev)])
  return best_x

