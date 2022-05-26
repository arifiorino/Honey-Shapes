#!/usr/bin/python3
import csv, datetime, numpy as np, torch, os
import A1,A1p,A2,A2p,A3,A4,A4p,A4pp
from utils import *

nBins=5
nDeaths = 5
nLosses = 3
algos=[A1,A1p,A2,A2p,A3,A4,A4p,A4pp]
device='cuda'
nExp=50

ucb_mult = 1
GP_ERROR = 0.05
RBF_SIGMA = 1
def GP(xs, F, x):
  xs,F,x=torch.Tensor(xs).to(device),torch.Tensor(F).to(device),torch.Tensor(x).to(device)
  xs1 = torch.pow(xs,2)@torch.ones((xs.size()[1],1)).to(device)
  xs2 = torch.cat([torch.t(xs1)]*xs.size()[0])
  M = xs2 - 2*(xs @ torch.t(xs)) + torch.t(xs2)
  M = torch.exp(-M/(2*RBF_SIGMA**2))
  M += GP_ERROR * torch.eye(xs.size()[0]).to(device)
  x1 = torch.pow(x,2)@torch.ones((x.size()[1],1)).to(device)
  x2 = torch.cat([torch.t(x1)]*xs.size()[0])
  xs3 = torch.cat([torch.t(xs1)]*x.size()[0])
  v = torch.t(x2) - 2*(x @ torch.t(xs)) + xs3
  v = torch.exp(-v/(2*RBF_SIGMA**2))
  temp = v @ torch.linalg.inv(M)
  mean = temp @ F
  var = torch.ones(mean.size()).to(device)
  var -= (temp*v)@torch.ones((temp.size()[1],1)).to(device)
  stdev = torch.sqrt(var)
  return (mean.to('cpu').numpy(), stdev.to('cpu').numpy())

with open('United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv', 'r') as f:
  rows=[list(row) for row in csv.reader(f)][1:]
rows = [[datetime.datetime.strptime(row[0], '%m/%d/%Y')]+row[1:-3] for row in rows]
rows.sort(key=lambda s:s[0])
# filter before 11/2021, after 7/2020
rows = [row for row in rows if row[0]<=datetime.datetime(year=2021,month=11,day=1)]
rows = [row for row in rows if row[0]>=datetime.datetime(year=2020,month=10,day=1)]
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
# normalize
avg_cases=sum(cases)/len(cases)
avg_deaths=sum(deaths)/len(deaths)
cases = [case/avg_cases for case in cases]
deaths = [death/avg_deaths for death in deaths]
# dates
dates=[row[0] for row in rows if row[1]==list(states)[0]][:len(cases)]

d_sort = sorted(deaths)
binLen = len(d_sort)/nBins
bins=[sum(d_sort[int(i*binLen):int((i+1)*binLen)])/binLen for i in range(nBins)]

gCases, gDeaths = cases, deaths

def f(deathI, deaths):
  diff = np.abs(deaths-np.array(gDeaths[deathI:deathI+nDeaths]))
  losses = np.zeros(nLosses)
  for i in range(nDeaths-nLosses+1):
    losses+=diff[i:i+nLosses]
  return losses.tolist()

#points=[]
for expI in range(nExp):
  print(expI,end=' ',flush=True)
  caseIs = np.random.randint(len(gCases)-nDeaths-8,size=101).tolist()
  start=np.random.choice(bins, nDeaths).tolist()
  for algo in algos:
    print(algo.name,end=' ',flush=True)
    past_in = [start]
    past_out = [f(caseIs[0]+8, start)]
    train_x, train_y = [], []
    for t in range(100):
      caseI = caseIs[t+1]
      #print(''.join(['%7.2f'%a for a in [t]+past_in[-1]+[sum(past_out[-1])]]))
      algo.addTrainingData(gCases[caseIs[t]:caseIs[t]+nDeaths],past_in[t],past_out[t],train_x,train_y)
      best_x=algo.findUCB(gCases[caseI:caseI+nDeaths],bins,train_x,train_y,GP,ucb_mult)
      past_in.append(best_x)
      past_out.append(f(caseI+8, np.array(best_x)))
      #for i in range(nDeaths):
        #points.append([dates[caseI+i+8],best_x[i]])

    # APPEND MODE
    filename='res/'+algo.name+'.csv'
    append(filename, past_out)
  print()

'''
with open('res/actual.csv','w') as fi:
  csv.writer(fi).writerows([[dates[i],d] for i,d in enumerate(gDeaths)])
with open('res/points.csv','w') as fi:
  csv.writer(fi).writerows(points)

print("PREDICTING DEATHS")
points=[]
for caseI in range(0,len(gCases)-nDeaths-8,nDeaths):
  best_x=algo.findUCB(gCases[caseI:caseI+nDeaths],bins,train_x,train_y,GP,0)
  for i in range(nDeaths):
    points.append([dates[caseI+i+8],best_x[i]])
with open('res/points2.csv','w') as fi:
  csv.writer(fi).writerows(points)
'''
