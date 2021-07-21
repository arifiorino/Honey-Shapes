import numpy as np

RBF_SIGMA = 1
EPSILON = 0.01
RBF = lambda a, b: np.exp(-np.dot(a-b, a-b)/(2*RBF_SIGMA**2))
cov_matrix = lambda A, B: (np.fromfunction(np.vectorize(lambda i, j:
                           RBF(A[i], B[j])), (len(A), len(B)), dtype=int))
def GP(train_x, train_y, test_x):
  train_y2 = train_y - np.mean(train_y)
  K = cov_matrix(train_x, train_x)
  K_st = cov_matrix(train_x, test_x)
  K_stst = cov_matrix(test_x, test_x)
  temp = np.matmul(np.transpose(K_st), np.linalg.inv(K +
                   EPSILON * np.identity(len(train_x))))
  mean = np.dot(temp, train_y2) + np.mean(train_y)
  variance = np.diag(np.transpose(K_stst) - np.matmul(temp, K_st))
  stdev = np.sqrt(variance)
  return (mean, stdev)

train_x = np.array([[1,0,0,0], [1,0,1,0]])
train_y = np.array([2,5.0])
test_x = np.array([[1,0,0.1,0]])
mean, stdev = GP(train_x, train_y, test_x)

print('train_x', train_x,
      'train_y', train_y,
      'test_x', test_x,
      'mean', mean,
      'stdev', stdev, sep='\n')
