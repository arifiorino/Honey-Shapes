import numpy as np
from scipy.stats import norm
import csv

N = 5
M = 5
K = 3
T = 50
ucb_multiplier = 5

contexts = [0, 1, 0, 1, 0]

GP_ERROR = 0.01
RBF_SIGMA = 2
RBF = lambda a, b: np.exp(-np.dot(a-b, a-b)/(2*RBF_SIGMA))
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

def acq(mu, sig, f_st):
  if sig==0:
    return 0
  x = (mu - f_st) * norm.cdf((mu - f_st)/sig) + sig/(np.sqrt(2*np.pi))*np.exp(-(f_st-mu)**2/(2*sig**2))
  return x

def A2_itol(idx):
  r=[]
  for i in range(M):
    r.append(idx % N)
    idx //= N
  return r
def A2_ltoi(features):
  r = 0
  x = 1
  for feature in features:
    r += feature * x
    x *= N
  return r
def A2p_itol(idx):
  r=[]
  for i in range(K):
    r.append(idx % N)
    idx //= N
  return r
A2p_ltoi = A2_ltoi
def A3_itol(idx):
  r=[]
  for i in range(K):
    r.append(idx % N)
    idx //= N
  return r + [idx]
A3_ltoi = A2_ltoi

A1_in = np.arange(N)
A1p_in = np.arange(N)
A2_in = []
for i in range(N**M):
  A2_in.append(A2_itol(i))
A2_in = np.array(A2_in)
A2p_in = []
for i in range(N**K):
  A2p_in.append(A2p_itol(i))
A2p_in = np.array(A2p_in)
A3_in = []
for i in range(N**K * 2):
  A3_in.append(A3_itol(i))
A3_in = np.array(A3_in)

actual_gp = np.random.multivariate_normal(np.zeros(N**K * 2),
                                          cov_matrix(A3_in, A3_in))

def f(x):
  r = []
  for end in range(M):
    l = []
    for curr in range(end-K+1, end+1):
      l.append(x[curr])
    l.append(contexts[end])
    r.append(actual_gp[A3_ltoi(l)])
  return r

best_score, best_x = 0, []
for idx in range(N**M):
  x = A2_itol(idx)
  score = sum(f(x))
  if score > best_score:
    best_score = score
    best_x = x
print('best',best_score,best_x)


final_data_A1 = [[] for _ in range(T)]
final_data_A1p = [[] for _ in range(T)]
final_data_A2 = [[] for _ in range(T)]
final_data_A2p = [[] for _ in range(T)]
final_data_A3 = [[] for _ in range(T)]

for trial in range(100):
  first = np.random.randint(N, size=M).tolist()

  print('A1')
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [], []
  for t in range(T-1):
    print(t, sum(past_out[-1]))
    for i in range(M):
      train_x.append([past_in[t][i]])
      train_y.append([sum(past_out[t])])
    mean, stdev = GP(np.array(train_x), np.array(train_y), A1_in)
    best_ucb = -1e10
    best_x = []
    for x in range(N):
      ucb = mean[x] + ucb_multiplier * stdev[x]
      if ucb > best_ucb:
        best_ucb = ucb
        best_x = x
    past_in.append([best_x for _ in range(N)])
    past_out.append(f(past_in[-1]))
  final = [sum(out) for out in past_out]
  for i in range(T):
    final_data_A1[i].append(final[i])
  with open('A1.csv', 'w') as f1:
    cf1 = csv.writer(f1)
    cf1.writerows(final_data_A1)

  print('A1p')
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [[] for _ in range(M)], [[] for _ in range(M)]
  for t in range(T-1):
    print(t, sum(past_out[-1]))
    mean, stdev = [], []
    for i in range(M):
      train_x[i].append([past_in[t][i]])
      train_y[i].append([past_out[t][i]])
      m, s= GP(np.array(train_x[i]), np.array(train_y[i]), A1p_in)
      mean.append(m)
      stdev.append(s)
    best_x = []
    for i in range(M):
      best_ucb = -1e10
      best_xi = None
      for xi in range(N):
        ucb = mean[i][xi] + ucb_multiplier * stdev[i][xi]
        if ucb > best_ucb:
          best_ucb = ucb
          best_xi = xi
      best_x.append(best_xi)
    past_in.append(best_x)
    past_out.append(f(past_in[-1]))
  final = [sum(out) for out in past_out]
  for i in range(T):
    final_data_A1p[i].append(final[i])
  with open('A1p.csv', 'w') as f1:
    cf1 = csv.writer(f1)
    cf1.writerows(final_data_A1p)

  print('A2')
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [], []
  for t in range(T-1):
    print(t, sum(past_out[-1]))
    train_x.append(past_in[-1])
    train_y.append([sum(past_out[-1])])
    mean, stdev = GP(np.array(train_x), np.array(train_y), A2_in)
    best_ucb = -1e10
    best_x = []
    for idx in range(N**M):
      x = A2_itol(idx)
      ucb = mean[idx] + ucb_multiplier * stdev[idx]
      if ucb > best_ucb:
        best_ucb = ucb
        best_x = x
    past_in.append(best_x)
    past_out.append(f(best_x))
  final = [sum(out) for out in past_out]
  for i in range(T):
    final_data_A2[i].append(final[i])
  with open('A2.csv', 'w') as f1:
    cf1 = csv.writer(f1)
    cf1.writerows(final_data_A2)

  print('A2p')
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [], []
  for t in range(T-1):
    print(t, sum(past_out[-1]))
    for end in range(M):
      train_x.append([past_in[t][curr] for curr in range(end-K+1, end+1)])
      train_y.append([past_out[t][end]])
    mean, stdev = GP(np.array(train_x), np.array(train_y), A2p_in)
    best_ucb = -1e10
    best_x = []
    for idx in range(N**M):
      x = A2_itol(idx)
      ucb = 0
      for end in range(M):
        l = [x[curr] for curr in range(end-K+1, end+1)]
        ucb += mean[A2p_ltoi(l)]
        ucb += ucb_multiplier * stdev[A2p_ltoi(l)]
      if ucb > best_ucb:
        best_ucb = ucb
        best_x = x
    past_in.append(best_x)
    past_out.append(f(best_x))
  final = [sum(out) for out in past_out]
  for i in range(T):
    final_data_A2p[i].append(final[i])
  with open('A2p.csv', 'w') as f1:
    cf1 = csv.writer(f1)
    cf1.writerows(final_data_A2p)

  print('A3')
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [], []
  for t in range(T-1):
    print(t, sum(past_out[-1]))
    for end in range(M):
      train_x.append([past_in[t][curr] for curr in range(end-K+1, end+1)])
      train_x[-1].append(contexts[end])
      train_y.append([past_out[t][end]])
    mean, stdev = GP(np.array(train_x), np.array(train_y), A3_in)
    best_ucb = -1e10
    best_x = []
    for idx in range(N**M):
      x = A2_itol(idx)
      ucb = 0
      for end in range(M):
        l = [x[curr] for curr in range(end-K+1, end+1)]
        l.append(contexts[end])
        ucb += mean[A3_ltoi(l)]
        ucb += ucb_multiplier * stdev[A3_ltoi(l)]
      if ucb > best_ucb:
        best_ucb = ucb
        best_x = x
    past_in.append(best_x)
    past_out.append(f(best_x))
  final = [sum(out) for out in past_out]
  for i in range(T):
    final_data_A3[i].append(final[i])
  with open('A3.csv', 'w') as f1:
    cf1 = csv.writer(f1)
    cf1.writerows(final_data_A3)


