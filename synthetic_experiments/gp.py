import numpy as np
from scipy.stats import norm
import csv, time

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

def itol(idx, mods):
  r=[]
  for i in range(len(mods)):
    r.append(idx % mods[i])
    idx //= mods[i]
  return r
def ltoi(features, mods):
  r, x = 0, 1
  for i, feature in enumerate(features):
    r += feature * x
    x *= mods[i]
  return r
GP_inputs = np.array([itol(i, [N]*K+[2]) for i in range(N**K * 2)])
f_inputs = np.array([itol(i, [N]*(M+K-1)) for i in range(N**(M+K-1))])
f_outputs = np.array([itol(i, [N]*M) for i in range(N**M)])

actual_gp = np.random.multivariate_normal(np.zeros(N**K * 2),
                                          cov_matrix(GP_inputs, GP_inputs))

def f(x):
  r = []
  for end in range(K-1, M+K-1):
    l = []
    for i in range(end-K+1, end+1):
      l.append(x[i])
    l.append(contexts[end-(K-1)])
    r.append(actual_gp[ltoi(l, [N]*K+[2])])
  return r

best_score, best_x = 0, []
for idx in range(N**(M+K-1)):
  x = itol(idx, [N]*(M+K-1))
  score = sum(f(x))
  if score > best_score:
    best_score = score
    best_x = x
scores = [sum(f(itol(idx, [N]*(M+K-1)))) for idx in range(N**(M+K-1))]
idx = np.argmax(scores)
print('best', itol(idx,[N]*(M+K-1)), scores[idx])

final_data_1 = [[] for _ in range(T)]
final_data_2 = [[] for _ in range(T)]

for trial in range(100):
  first = np.random.randint(N, size=M+K-1).tolist()

  start = time.time()
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [], []
  for t in range(T-1):
    print(t, past_in[-1], sum(past_out[-1]))
    for end in range(K-1, M+K-1):
      train_x.append([past_in[t][i] for i in range(end-K+1, end+1)])
      train_x[-1].append(contexts[end-(K-1)])
      train_y.append([past_out[t][end-(K-1)]])
    mean, stdev = GP(np.array(train_x), np.array(train_y), GP_inputs)
    best_ucb = -1e10
    best_x = []
    for idx in range(N**(M+K-1)):
      x = itol(idx, [N]*(M+K-1))
      ucb = 0
      for end in range(K-1, M+K-1):
        l = [x[i] for i in range(end-K+1, end+1)]
        l.append(contexts[end-(K-1)])
        ucb += mean[ltoi(l, [N]*K+[2])] + ucb_multiplier * stdev[ltoi(l, [N]*K+[2])]
      if ucb > best_ucb:
        best_ucb = ucb
        best_x = x
    past_in.append(best_x)
    past_out.append(f(best_x))
  print('A3',time.time()-start)
  final_data_1 = [final_data_1[i]+[sum(past_out[i])] for i in range(T)]
  with open('A3.csv', 'w') as f1:
    csv.writer(f1).writerows(final_data_1)

  start = time.time()
  past_in, past_out  = [first], [f(first)]
  train_x, train_y = [], []
  for t in range(T-1):
    print(t, past_in[-1], sum(past_out[-1]))
    for end in range(K-1, M+K-1):
      train_x.append([past_in[t][i] for i in range(end-K+1, end+1)])
      train_x[-1].append(contexts[end-(K-1)])
      train_y.append([past_out[t][end-(K-1)]])
    mean, stdev = GP(np.array(train_x), np.array(train_y), GP_inputs)
    best_ucb = -1e10
    best_x = []
    for idx in range(N**K):
      l = itol(idx, [N]*K) + [contexts[0]]
      ucb = mean[ltoi(l, [N]*K+[2])] + ucb_multiplier * stdev[ltoi(l, [N]*K+[2])]
      if ucb > best_ucb:
        best_ucb = ucb
        best_x = l[:-1]
    for end in range(K, M+K-1):
      best_ucb = -1e10
      best_a = None
      for a in range(N):
        l = [best_x[i] for i in range(end-K+1, end)]+[a]+[contexts[end-(K-1)]]
        ucb = mean[ltoi(l, [N]*K+[2])] + ucb_multiplier * stdev[ltoi(l, [N]*K+[2])]
        if ucb > best_ucb:
          best_ucb = ucb
          best_a = a
      best_x.append(best_a)
    past_in.append(best_x)
    past_out.append(f(best_x))
  print('A4',time.time()-start)
  final_data_2 = [final_data_2[i]+[sum(past_out[i])] for i in range(T)]
  with open('A4.csv', 'w') as f1:
    csv.writer(f1).writerows(final_data_2)


