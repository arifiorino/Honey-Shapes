import csv, datetime, numpy as np, matplotlib.pyplot as plt

ucb_mult = 50
GP_ERROR = 0.1
RBF_SIGMA = 5*10**4
alpha=None
RBF = lambda a, b: np.exp(-np.dot((a-b)*alpha, (a-b)*alpha)/(2*RBF_SIGMA**2))
cov_matrix = lambda A, B: (np.fromfunction(np.vectorize(lambda i, j:
                           RBF(A[i], B[j])), (len(A), len(B)), dtype=int))
def GP(xs, F, x, avg=0):
  M = cov_matrix(xs, xs) + GP_ERROR * np.identity(len(xs))
  v = cov_matrix(x, xs)
  temp = v @ np.linalg.inv(M)
  mean = temp @ (F-avg)
  var = np.fromfunction(np.vectorize(lambda i,_: RBF(x[i], x[i])), mean.shape, dtype=int)
  var -= np.fromfunction(np.vectorize(lambda i,_: np.dot(temp[i], v[i])), mean.shape, dtype=int)
  return (mean+avg, var)

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
cases = [(cases[i+1]-cases[i]) for i in range(len(cases)-1)]
deaths = [(deaths[i+1]-deaths[i]) for i in range(len(deaths)-1)]
# weekly average
cases = [sum(cases[i:i+7])/7 for i in range(len(cases)-7)]
deaths = [sum(deaths[i:i+7])/7 for i in range(len(deaths)-7)]
avg_deaths=sum(deaths)/len(deaths)
print('avg_deaths',avg_deaths)
# dates
dates=[row[0] for row in rows if row[1]=='MD'][:len(cases)]

d_sort = sorted(deaths)
nBins=20
binLen = len(d_sort)/nBins
bins=[int(sum(d_sort[int(i*binLen):int((i+1)*binLen)])/binLen) for i in range(nBins)]
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
nCases = 8+6
nDeaths = 6+6
nLosses = 4+6

def f(deathI, deaths):
  diff = np.abs(deaths-np.array(gDeaths[deathI:deathI+nDeaths]))
  losses = np.zeros(nLosses)
  for i in range(nDeaths-nLosses+1):
    losses+=diff[i:i+nLosses]
  return losses/(nDeaths-nLosses+1)

caseIs = [np.random.randint(len(gCases)-nCases+1)]
past_in = [np.random.choice(bins, nDeaths)]
past_out = [f(caseIs[0], past_in[0])]
past_in[0]=past_in[0].tolist()
train_x, train_y = [], []
alpha=np.array([1]*(nCases-nLosses+1)+[100]*(nDeaths-nLosses+1))
for t in range(100):
  caseI = np.random.randint(len(gCases)-nCases+1)
  #print('Cases:',gCases[caseI:caseI+nCases])
  print('t',t,sum(past_out[-1]))
  #print('in',past_in[-1])
  #print('out',past_out[-1])
  #print('sum out',sum(past_out[-1]))
  for lossI in range(nLosses):
    train_x.append(gCases[int(caseIs[t]+lossI):int(caseIs[t]+lossI+(nCases-nLosses+1))])
    train_x[-1].extend(past_in[t][lossI:lossI+(nDeaths-nLosses+1)])
    train_y.append([past_out[t][lossI]])
  #print('train_x',train_x[-1])
  #print('train_y',train_y[-1])
  test_x=[]
  for idx in range(nBins**(nDeaths-nLosses+1)):
    d = itol(idx, [nBins]*(nDeaths-nLosses+1))
    test_x.append(gCases[caseI:caseI+nCases-nLosses+1] + [bins[x] for x in d])
  mean, stdev = GP(np.array(train_x), np.array(train_y), np.array(test_x),avg_deaths)
  #print('qwerty',test_x[:5],mean[:5].tolist(), stdev[:5].tolist())
  #mean2, stdev2 = GP(np.array(train_x), np.array(train_y), np.array(train_x),avg_deaths)
  #print('qwerty2',train_y[:5],mean2[:5].tolist(), stdev2[:5].tolist())
  best_x = test_x[np.argmin(mean - ucb_mult * stdev)][nCases-nLosses+1:]
  for i in range(1,nLosses):
    test_x=[]
    for a in bins:
      test_x.append(gCases[caseI+i:caseI+i+nCases-nLosses+1]+best_x[i:i+nDeaths-nLosses]+[a])
    mean, stdev = GP(np.array(train_x), np.array(train_y), np.array(test_x),avg_deaths)
    best_x.append(bins[np.argmin(mean - ucb_mult * stdev)])
  #print('best_x',best_x)
  #input('?')
  caseIs.append(caseI)
  past_in.append(best_x)
  past_out.append(f(caseI + (nCases-nDeaths), np.array(best_x)))

with open('data.csv', 'w') as f:
  csv.writer(f).writerows([[sum(x)] for x in past_out])

