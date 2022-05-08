import numpy as np
name='A1'
def addTrainingData(nCases,nDeaths,nLosses,gCases,caseI,last_in,last_out,train_x,train_y):
  for lossI in range(nLosses):
    train_x.append(gCases[int(caseI+lossI+(nDeaths-nLosses)):int(caseI+lossI+(nCases-nLosses+1))])
    train_x[-1].extend([last_in[lossI+(nDeaths-nLosses)]])
    train_y.append([last_out[lossI]])

def findUCB(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol,ucb_mult):
  best_x=[]
  for deathI in range(nDeaths):
    test_x=[]
    for x in bins:
      test_x.append(gCases[caseI+deathI:caseI+deathI+(nCases-nDeaths+1)]+[x])
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(bins[np.argmin(mean-ucb_mult*stdev)])
  return best_x

def findBest(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol):
  best_x=[]
  for deathI in range(nDeaths):
    test_x=[]
    for x in bins:
      test_x.append(gCases[caseI+deathI:caseI+deathI+(nCases-nDeaths+1)]+[x])
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(bins[np.argmin(mean)])
  return best_x
