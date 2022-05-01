#!/usr/bin/python3
import csv, datetime, numpy as np, matplotlib.pyplot as plt

ucb_mult = 2
GP_ERROR = 0.1
RBF_SIGMA = 2
RBF = lambda a, b: np.exp(-np.dot(a-b, a-b)/(2*RBF_SIGMA**2))
cov_matrix = lambda A, B: (np.fromfunction(np.vectorize(lambda i, j:
                           RBF(A[i], B[j])), (len(A), len(B)), dtype=int))
def GP(xs, F, x):
  M = cov_matrix(xs, xs) + GP_ERROR * np.identity(len(xs))
  v = cov_matrix(x, xs)
  temp = v @ np.linalg.inv(M)
  mean = temp @ (F)
  var = np.fromfunction(np.vectorize(lambda i,_: RBF(x[i], x[i])), mean.shape, dtype=int)
  var -= np.fromfunction(np.vectorize(lambda i,_: np.dot(temp[i], v[i])), mean.shape, dtype=int)
  return (mean, var)

with open('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv', 'r') as f:
  rows=[list(row) for row in csv.reader(f)][1:]
rows = [[datetime.datetime.strptime(row[0], '%m/%d/%Y')]+row[1:-3] for row in rows]
rows.sort(key=lambda s:s[0])
# filter before omicron
rows = [row for row in rows if row[0]<=datetime.datetime(year=2021,month=11,day=1)]
states=set([row[1] for row in rows])
cases,deaths=None,None
for state in states:
  if cases is None:
    cases=np.array([int(row[2]) for row in rows if row[1]==state])
    deaths=np.array([int(row[7]) for row in rows if row[1]==state])
  else:
    cases+=np.array([int(row[2]) for row in rows if row[1]==state])
    deaths+=np.array([int(row[7]) for row in rows if row[1]==state])
# cumulative -> daily
cases = [(cases[i+1]-cases[i]) for i in range(len(cases)-1)]
deaths = [(deaths[i+1]-deaths[i]) for i in range(len(deaths)-1)]
# weekly average
cases = [sum(cases[i:i+7])/7 for i in range(len(cases)-7)]
deaths = [sum(deaths[i:i+7])/7 for i in range(len(deaths)-7)]
avg_cases=sum(cases)/len(cases)
avg_deaths=sum(deaths)/len(deaths)
# normalize
cases = [case/avg_cases for case in cases]
deaths = [death/avg_deaths for death in deaths]
# dates
dates=[row[0] for row in rows if row[1]==list(states)[0]][:len(cases)]

d_sort = sorted(deaths)
nBins=5
binLen = len(d_sort)/nBins
bins=[sum(d_sort[int(i*binLen):int((i+1)*binLen)])/binLen for i in range(nBins)]

def plot():
  bins2=[d_sort[int(i*binLen)] for i in range(nBins)]+[d_sort[-1]]
  def b(x):
    for i in range(nBins):
      if x<=bins2[i+1]:
        return i
    raise ValueError()
  fig, ax = plt.subplots()
  ax.plot(dates, cases)
  ax2 = ax.twinx()
  ax2.plot(dates, deaths)
  ax2.plot(dates, [bins[b(death)] for death in deaths])
  plt.show()

def itol(idx, mods):
  r=[]
  for i in range(len(mods)):
    r.append(idx % mods[i])
    idx //= mods[i]
  return r

gCases, gDeaths = cases, deaths
nCases = 8
nDeaths = 6
nLosses = 4

def f(deathI, deaths):
  diff = np.abs(deaths-np.array(gDeaths[deathI:deathI+nDeaths]))
  losses = np.zeros(nLosses)
  for i in range(nDeaths-nLosses+1):
    losses+=diff[i:i+nLosses]
  return losses/(nDeaths-nLosses+1)

while 1:
  caseIs = [np.random.randint(len(gCases)-nCases+1)]
  past_in = [np.random.choice(bins, nDeaths)]
  past_out = [f(caseIs[0], past_in[0])]
  print(gCases[caseIs[0]:caseIs[0]+nCases],past_in[0],past_out[0])
  past_in[0]=past_in[0].tolist()
  train_x, train_y = [], []
  for t in range(100):
    caseI = np.random.randint(len(gCases)-nCases+1)
    print(''.join(['%7.2f'%a for a in [t]+gCases[caseIs[-1]:caseIs[-1]+nCases]+past_in[-1]+[sum(past_out[-1])]]))
    #print('cases given', gCases[caseI:caseI+nCases])
    for lossI in range(nLosses):
      train_x.append(gCases[int(caseIs[t]+lossI):int(caseIs[t]+lossI+(nCases-nLosses+1))])
      train_x[-1].extend(past_in[t][lossI:lossI+(nDeaths-nLosses+1)])
      train_y.append([past_out[t][lossI]])
    #print('train points added',train_x[-1],train_y[-1])
    test_x=[]
    for idx in range(nBins**nDeaths):
      a=[bins[i] for i in itol(idx,[nBins]*nDeaths)]
      for i in range(nLosses):
        test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+a[i:i+(nDeaths-nLosses+1)])
    mean, stdev = GP(np.array(train_x), np.array(train_y), np.array(test_x))
    #print('--ENTERING GP--')
    #for i in range(len(test_x)):
      #print(''.join(['%7.2f'%a for a in test_x[i]+[mean[i]]+[stdev[i]]]))
    #print('--EXITING GP--')
    #input('c?')
    best_x=None
    best_y=1e10
    #print('--ENTERING DECISION MAKING--')
    for idx in range(nBins**nDeaths):
      y=np.sum(mean[idx*nLosses:(idx+1)*nLosses]-ucb_mult*stdev[idx*nLosses:(idx+1)*nLosses])
      #print(''.join(['%7.2f'%a for a in [bins[i] for i in itol(idx,[nBins]*nDeaths)]+\
            #(mean[idx*nLosses:(idx+1)*nLosses]).T[0].tolist()+\
            #(ucb_mult*stdev[idx*nLosses:(idx+1)*nLosses]).T[0].tolist()+\
            #[y]]))
      if y<best_y:
        best_x=itol(idx,[nBins]*nDeaths)
        best_y=y
    #print('--EXITING DECISION MAKING--')
    best_x = [bins[i] for i in best_x]
    #print('deaths predicted', best_x)
    #input('c?')
    caseIs.append(caseI)
    past_in.append(best_x)
    past_out.append(f(caseI + (nCases-nDeaths), np.array(best_x)))

  # APPEND MODE
  with open('A3.csv', 'r') as fi:
    data=[list(row) for row in csv.reader(fi)]
  if len(data)==0:
    data=[[] for _ in past_out]
  with open('A3.csv', 'w') as fi:
    csv.writer(fi).writerows([data[i]+[sum(x)] for i,x in enumerate(past_out)])



