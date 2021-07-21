import sys, scipy.optimize
import numpy as np
from draw_square import draw_config, score_image, deps

target_image = draw_config([0.006,0.01,0,0,0.01,0,0,0.01,0,0,0.01,0,0])
target_O = score_image(target_image)

N_test = 20
test_E = np.random.rand(N_test, 13) * 0.015
test_O = np.zeros((N_test, 8))
for i in range(N_test):
  test_O[i] = score_image(draw_config(test_E[i]))

K = np.zeros((8, 13))
for i in range(8):
  Os = test_O[:,i]
  deletes=[]
  for j in range(13):
    if deps[i][j] == 0:
      deletes.append(j)
  my_test_E = np.delete(test_E, deletes, 1)
  res = scipy.optimize.lsq_linear(my_test_E, Os, bounds=(0,np.inf))
  Ks = res['x']
  idx=0
  for j in range(13):
    if deps[i][j]:
      K[i,j]=Ks[idx]
      idx+=1
print(K)
res = scipy.optimize.lsq_linear(K, target_O, bounds=(0,np.inf))
E = res['x']

im = draw_config(E)
im.show()
O = score_image(im)
print(E)
