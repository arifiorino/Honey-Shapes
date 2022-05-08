import numpy as np
name='A3'
def addTrainingData(nCases,nDeaths,nLosses,gCases,caseI,last_in,last_out,train_x,train_y):
  for lossI in range(nLosses):
    train_x.append(gCases[int(caseI+lossI):int(caseI+lossI+(nCases-nLosses+1))])
    train_x[-1].extend(last_in[lossI:lossI+(nDeaths-nLosses+1)])
    train_y.append([last_out[lossI]])

def findUCB(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol,ucb_mult):
  test_x=[]
  for idx in range(nBins**nDeaths):
    a=[bins[i] for i in itol(idx,[nBins]*nDeaths)]
    for i in range(nLosses):
      test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+a[i:i+(nDeaths-nLosses+1)])
  mean, stdev = GP(train_x, train_y, test_x)
  best_x=None
  best_y=1e10
  for idx in range(nBins**nDeaths):
    y=np.sum(mean[idx*nLosses:(idx+1)*nLosses]-ucb_mult*stdev[idx*nLosses:(idx+1)*nLosses])
    if y<best_y:
      best_x=itol(idx,[nBins]*nDeaths)
      best_y=y
  best_x = [bins[i] for i in best_x]
  return best_x


def findBest(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol):
  test_x=[]
  for idx in range(nBins**nDeaths):
    a=[bins[i] for i in itol(idx,[nBins]*nDeaths)]
    for i in range(nLosses):
      test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+a[i:i+(nDeaths-nLosses+1)])
  mean, stdev = GP(train_x, train_y, test_x)
  best_x=None
  best_y=1e10
  for idx in range(nBins**nDeaths):
    y=np.sum(mean[idx*nLosses:(idx+1)*nLosses])
    if y<best_y:
      best_x=itol(idx,[nBins]*nDeaths)
      best_y=y
  best_x = [bins[i] for i in best_x]
  return best_x
