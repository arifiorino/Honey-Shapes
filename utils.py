import csv, os

def append(filename, x):
  data=[]
  if os.path.isfile(filename):
    with open(filename, 'r') as fi:
      data=[list(row) for row in csv.reader(fi)]
  if len(data)==0:
    data=[[] for _ in x]
  with open(filename, 'w') as fi:
    csv.writer(fi).writerows([data[i]+[sum(y)] for i,y in enumerate(x)])

def allSequences(N,K):
  seq=[]
  a=[0 for _ in range(N)]
  while a[-1]<K:
    seq.append(a.copy())
    a[0]+=1
    for i in range(N-1):
      if a[i]==K:
        a[i]=0
        a[i+1]+=1
      else:
        break
  return seq

