name='A2'
def addTrainingData(nCases,nDeaths,nLosses,gCases,caseI,last_in,last_out,train_x,train_y):
  train_x.append(gCases[caseI:caseI+nCases])
  train_x[-1].extend(last_in)
  train_y.append([sum(last_out)])


def findUCB(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol,ucb_mult):
  test_x=[]
  for idx in range(nBins**nDeaths):
    a=[bins[i] for i in itol(idx,[nBins]*nDeaths)]
    test_x.append(gCases[caseI:caseI+nCases]+a)
  mean, stdev = GP(train_x, train_y, test_x)
  best_x=None
  best_y=1e10
  for idx in range(nBins**nDeaths):
    y=mean[idx][0]-ucb_mult*stdev[idx][0]
    if y<best_y:
      best_x=itol(idx,[nBins]*nDeaths)
      best_y=y
  best_x = [bins[i] for i in best_x]
  return best_x

def findBest(nCases,nDeaths,nLosses,gCases,caseI,train_x,train_y,bins,nBins,GP,itol):
  test_x=[]
  for idx in range(nBins**nDeaths):
    a=[bins[i] for i in itol(idx,[nBins]*nDeaths)]
    test_x.append(gCases[caseI:caseI+nCases]+a)
  mean, stdev = GP(train_x, train_y, test_x)
  best_x=None
  best_y=1e10
  for idx in range(nBins**nDeaths):
    y=mean[idx][0]-ucb_mult*stdev[idx][0]
    if y<best_y:
      best_x=itol(idx,[nBins]*nDeaths)
      best_y=y
  best_x = [bins[i] for i in best_x]
  return best_x
