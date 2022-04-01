import csv, datetime, numpy as np, matplotlib.pyplot as plt

ucb_mult = 5
GP_ERROR = 1
RBF_SIGMA = 10**4
RBF = lambda a, b: np.exp(-np.dot(a-b, a-b)/(2*RBF_SIGMA**2))
cov_matrix = lambda A, B: (np.fromfunction(np.vectorize(lambda i, j:
                           RBF(A[i], B[j])), (len(A), len(B)), dtype=int))
def GP(xs, F, x):
  M = cov_matrix(xs, xs) + GP_ERROR * np.identity(len(xs))
  v = cov_matrix(x, xs)
  temp = v @ np.linalg.inv(M)
  mean = temp @ F
  var = np.fromfunction(np.vectorize(lambda i,_: RBF(x[i], x[i])), mean.shape, dtype=int)
  var -= np.fromfunction(np.vectorize(lambda i,_: np.dot(temp[i], v[i])), mean.shape, dtype=int)
  return (mean, var)

with open('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv', 'r') as f:
  rows=[list(row) for row in csv.reader(f)][1:]
rows = [[datetime.datetime.strptime(row[0], '%m/%d/%Y')]+row[1:-3] for row in rows]
rows.sort(key=lambda s:s[0])
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
cases = [cases[i+1]-cases[i] for i in range(len(cases)-1)]
deaths = [deaths[i+1]-deaths[i] for i in range(len(deaths)-1)]
# weekly average
cases = [round(sum(cases[i:i+7])/7,2) for i in range(len(cases)-7)]
deaths = [round(sum(deaths[i:i+7])/7,2) for i in range(len(deaths)-7)]
# normalize
#deaths=[d/1000 for d in deaths]
#cases=[c/1000 for c in cases]
# dates
dates=[row[0] for row in rows if row[1]=='MD'][:len(cases)]

d_sort = sorted(deaths)
nBins=20
binLen = len(d_sort)/nBins
bins=[sum(d_sort[int(i*binLen):int((i+1)*binLen)])/binLen for i in range(nBins)]
print(bins)
'''
bins=[d_sort[int(i*binLen)] for i in range(nBins)]+[d_sort[-1]]
def b(x):
  for i in range(nBins):
    if x<=bins[i+1]:
      return i
  raise ValueError()
with open('corona.csv','w') as f:
  csv.writer(f).writerows([['date','cases','deaths','deaths_binned']]+[[dates[i],cases[i],deaths[i],binAvg[b(deaths[i])]] for i in range(len(cases))])
fig, ax = plt.subplots()
ax.plot(dates, cases)
ax2 = ax.twinx()
#ax2.plot(dates, [binAvg[b(death)] for death in deaths])
ax2.plot(dates, deaths)
plt.show()
'''
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
  diff = abs(deaths-gDeaths[deathI:deathI+nDeaths])
  print('diff',diff)
  losses = np.zeros(nLosses)
  for i in range(nDeaths-nLosses+1):
    losses+=diff[i:i+nLosses]
  return losses/(nDeaths-nLosses+1)

caseIs = [np.random.randint(len(gCases)-nCases+1)]
past_in = [np.random.choice(bins, nDeaths)]
past_out = [f(caseIs[0], past_in[0])]
past_in[0]=past_in[0].tolist()
train_x, train_y = [], []
for t in range(100):
  caseI = np.random.randint(len(gCases)-nCases+1)
  print('Cases:',gCases[caseIs[t]:caseIs[t]+nCases])
  print('t',t)
  print('in',past_in[-1])
  print('out',past_out[-1])
  print('sum out',sum(past_out[-1]))
  for lossI in range(nLosses):
    train_x.append(gCases[caseIs[t]+lossI:caseIs[t]+lossI+(nCases-nLosses+1)])
    train_x[-1].extend(past_in[t][lossI:lossI+(nDeaths-nLosses+1)])
    train_y.append([past_out[t][lossI]])
  print('train_x',train_x[-1])
  print('train_y',train_y[-1])
  test_x=[]
  for idx in range(nBins**(nDeaths-nLosses+1)):
    d = itol(idx, [nBins]*(nDeaths-nLosses+1))
    test_x.append(gCases[caseI:caseI+nCases-nLosses+1] + [bins[x] for x in d])
  mean, stdev = GP(np.array(train_x), np.array(train_y), np.array(test_x))
  print('qwerty',mean[:5], stdev[:5])
  mean2, stdev2 = GP(np.array(train_x), np.array(train_y), np.array(train_x))
  print('qwerty2',train_y[:5],mean2[:5], stdev2[:5])
  best_x = test_x[np.argmin(mean - ucb_mult * stdev)][nCases-nLosses+1:]
  print('best_x',best_x)
  input('?')
  for i in range(1,nLosses):
    test_x=[]
    for a in bins:
      test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+best_x[i:i+nDeaths-nLosses]+[a])
    mean, stdev = GP(np.array(train_x), np.array(train_y), np.array(test_x))
    best_x.append(bins[np.argmin(mean - ucb_mult * stdev)])
  caseIs.append(caseI)
  past_in.append(best_x)
  past_out.append(f(caseI + (nCases-nDeaths), np.array(best_x)))





