#!/usr/bin/python3
import numpy as np, csv, torch
import A1,A1p,A2,A2p,A3,A4,A4p,A4pp
from utils import *

N=5
N_actions = 5
N_losses = 3
algos=[A1,A1p,A2,A2p,A3,A4,A4p,A4pp]
#algos=[A4p]
ucb_mult= 1
device='cuda'
context = [5, 10, 2, 15, 7]
GP_ERROR = 0.01
RBF_SIGMA = 2
nExp=100
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
A=[context[i:i+K] for i in range(N_losses)]
B=allSequences(K,N)
A=[a+b for b in B for a in A]
#calculate covariance of A
A=torch.Tensor(A).to(device)
A1 = torch.pow(A,2)@torch.ones((A.size()[1],1)).to(device)
A1 = torch.cat([torch.t(A1)]*A.size()[0])
A = A1 - 2*(A @ torch.t(A)) + torch.t(A1)
A = torch.exp(-A/(2*RBF_SIGMA**2)).to('cpu').numpy()
actual_gp = np.random.multivariate_normal(np.zeros(N**K*N_losses),A)

def f(x):
  assert(len(x)==len(context))
  r = []
  for i in range(N_losses):
    idx=i
    idx+=sum([a*N**i for i,a in enumerate(x[i:i+K])])*N_losses
    r.append(actual_gp[idx])
  return r

for expI in range(nExp):
  print(expI,end=' ',flush=True)
  start=np.random.choice(N, N_actions).tolist()
  for algo in algos:
    print(algo.name,end=' ',flush=True)
    past_in = [start]
    past_out = [f(start)]
    train_x, train_y = [], []
    for t in range(100):
      #print(''.join(['%7.2f'%a for a in [t]+past_in[-1]+[sum(past_out[-1])]]))
      algo.addTrainingData(context,past_in[t],past_out[t],train_x,train_y)
      best_x=algo.findUCB(context,list(range(N)),train_x,train_y,GP,ucb_mult)
      past_in.append(best_x)
      past_out.append(f(np.array(best_x)))

    # APPEND MODE
    filename='res/'+algo.name+'.csv'
    append(filename, past_out)
  print()


