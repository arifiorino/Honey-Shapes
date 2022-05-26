#!/usr/bin/python3
import matplotlib.pyplot as plt, numpy as np, csv, datetime

def A3_A4():
  n=[1]
  files = ['A%d.csv'%i for i in n]
  labels = ['Algo %d'%i for i in n]
  datas = []
  for filename in files:
    with open(filename, 'r') as f:
      datas.append(np.array([[float(x) for x in row] for row in csv.reader(f)]))
  for i,data in enumerate(datas):
    datas[i]=((np.mean(data, axis=1),np.std(data, axis=1)))
  T = len(datas[0][0])
  plt.title("Regret versus iteration")
  plt.xlabel("Iteration")
  plt.ylabel("Regret")
  for i,(mean,std) in enumerate(datas):
    plt.fill_between(range(T), mean - std, mean + std, alpha=0.5)
    plt.plot(mean, label=labels[i])
  plt.legend()
  plt.show()
  #plt.savefig('A3_4.png')
A3_A4()

def param_search():
  for ucb_mult in [1,2,5]:
    for GP_ERROR in [0.05,0.1,0.2]:
      for RBF_SIGMA in [1,2,3]:
        filename=f'tune_hp/A4_{ucb_mult}_{GP_ERROR}_{RBF_SIGMA}.csv'
        with open(filename, 'r') as f:
          print(filename,np.mean(np.array([[float(x) for x in row] for row in csv.reader(f)])))
#param_search()

def points():
  with open('actual.csv') as f:
    actual=[[x,float(y)] for x,y in csv.reader(f)]
  with open('points2.csv') as f:
    points=[[x,float(y)] for x,y in csv.reader(f)]
  x=[datetime.datetime.fromisoformat(p[0]) for p in points]
  y=[p[1] for p in points]
  x2=[datetime.datetime.fromisoformat(p[0]) for p in actual]
  y2=[p[1] for p in actual]
  plt.plot(x2,y2,zorder=1)
  plt.scatter(x,y,s=8,c='black',zorder=2)
  plt.show()
points()

