import numpy as np
from draw_square import draw_config, score_image
import sys, scipy.optimize

target_image = draw_config([0.006,0.01,0,0,0.01,0,0,0.01,0,0,0.01,0,0])
target_O = score_image(target_image)

N_test = 10
test_E = np.random.rand(N_test, 13) * 0.01
test_O = np.zeros((N_test, 8))
for i in range(N_test):
  E = test_E[i]
  im = draw_config(E)
  O = score_image(im)
  test_O[i] = O

K = np.zeros((8, 13))
for i in range(8):
  Es = np.zeros((N_test, 13))
  Os = test_O[:,i]
  res = scipy.optimize.lsq_linear(Es, Os)
  Ks = res['x']
  for j in range(13):
    K[i,j]=Ks[j]

res = scipy.optimize.lsq_linear(K, target_O, bounds=(0,np.inf))
E = res['x']
im = draw_config(E)
im.show()
O = score_image(im)
print(O - target_O)
print(O)
