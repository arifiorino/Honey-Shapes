import numpy as np, itertools
name='A4'
def addTrainingData(nCases,nDeaths,nLosses,gCases,caseI,last_in,last_out,train_x,train_y):
  for lossI in range(nLosses):
    train_x.append(gCases[int(caseI+lossI):int(caseI+lossI+(nCases-nLosses+1))])
    train_x[-1].extend(last_in[lossI:lossI+(nDeaths-nLosses+1)])
    train_y.append([last_out[lossI]])


def findUCB(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol,ucb_mult):
  #test_x=[]
  #for idx in range(nBins**(nDeaths-nLosses+1)):
    #d = itol(idx, [nBins]*(nDeaths-nLosses+1))
    #test_x.append(gCases[caseI:caseI+nCases-nLosses+1] + [bins[x] for x in d])
  cases=np.repeat([gCases[caseI:caseI+nCases-nLosses+1]],nBins**(nDeaths-nLosses+1),axis=0)
  actions=np.array(list(itertools.product(bins,repeat=nDeaths-nLosses+1)))
  test_x=np.concatenate((cases,actions),axis=1).tolist()
  mean, stdev = GP(train_x, train_y, test_x)
  best_x = test_x[np.argmin(mean - ucb_mult * stdev)][nCases-nLosses+1:]
  for i in range(1,nLosses):
    test_x=[]
    for a in bins:
      test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+best_x[i:i+nDeaths-nLosses]+[a])
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(bins[np.argmin(mean - ucb_mult * stdev)])
  return best_x


def findBest(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol):
  test_x=[]
  for idx in range(nBins**(nDeaths-nLosses+1)):
    d = itol(idx, [nBins]*(nDeaths-nLosses+1))
    test_x.append(gCases[caseI:caseI+nCases-nLosses+1] + [bins[x] for x in d])
  mean, stdev = GP(train_x, train_y, test_x)
  best_x = test_x[np.argmin(mean)][nCases-nLosses+1:]
  for i in range(1,nLosses):
    test_x=[]
    for a in bins:
      test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+best_x[i:i+nDeaths-nLosses]+[a])
    mean, stdev = GP(train_x, train_y, test_x)
    best_x.append(bins[np.argmin(mean)])
  return best_x
