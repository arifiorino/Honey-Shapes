#!/usr/bin/python3
import numpy as np, csv, torch
import A1,A1p,A2,A2p,A3,A4,A4p,A4pp
from utils import *

N=5
N_actions = 5
N_losses = 3
algos=[A1,A1p,A2,A2p,A3,A4,A4p,A4pp]
ucb_mult= 1
device='cuda'
nExp=100
GP_ERROR = 0.01
RBF_SIGMA = 2
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

K=N_actions-N_losses+1

def f(c,x):
  a,b=np.array(c),np.array(x)
  diff = np.abs(a-b)
  losses = np.zeros(N_losses)
  for i in range(K):
    losses+=diff[i:i+N_losses]
  return losses.tolist()

for expI in range(nExp):
  print(expI,end=' ',flush=True)
  start=np.random.choice(N, N_actions).tolist()
  contexts=[[np.random.randint(N)]*N_actions for _ in range(101)]
  #contexts=(np.random.randint(N, size=(101,N_actions))).tolist()
  for algo in algos:
    print(algo.name,end=' ',flush=True)
    past_in = [start]
    past_out = [f(contexts[0],start)]
    train_x, train_y = [], []
    for t in range(100):
      #print(''.join(['%7.2f'%a for a in [t]+past_in[-1]+[sum(past_out[-1])]]))
      algo.addTrainingData(contexts[t],past_in[t],past_out[t],train_x,train_y)
      best_x=algo.findUCB(contexts[t+1],list(range(N)),train_x,train_y,GP,ucb_mult)
      past_in.append(best_x)
      past_out.append(f(contexts[t+1],np.array(best_x)))

    # APPEND MODE
    filename='res/'+algo.name+'.csv'
    append(filename, past_out)
  print()


