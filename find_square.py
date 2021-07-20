import numpy as np
from draw_square import draw_config, score_image
import sys, scipy.optimize

target_image = draw_config([0.006,0.01,0,0,0.01,0,0,0.01,0,0,0.01,0,0])
target_O = score_image(target_image)

N_test = 20
test_E = np.random.rand(N_test, 13) * 0.01
test_O = np.zeros((N_test, 8))
for i in range(N_test):
  E = test_E[i]
  im = draw_config(E)
  O = score_image(im)
  test_O[i] = O

deps = \
[[1,0,0,0,0,0,0,0,0,0,1,1,1],
 [1,1,1,0,0,0,0,0,0,0,1,1,1],
 [0,1,1,1,1,0,0,0,0,0,0,0,0],
 [0,0,0,1,1,1,0,0,0,0,0,0,0],
 [0,0,0,0,1,1,1,1,0,0,0,0,0],
 [0,0,0,0,0,0,1,1,1,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,1,1,1,0]]

K = np.zeros((8, 13))
for i in range(8):
  Os = test_O[:,i]
  bounds = (np.array(deps[i])+0.000001)*1000000
  #print(bounds)
  res = scipy.optimize.lsq_linear(test_E, Os, bounds=(-bounds,bounds))
  #res = scipy.optimize.lsq_linear(test_E, Os)
  Ks = res['x']
  for j in range(13):
    K[i,j]=Ks[j]
print(K)

res = scipy.optimize.lsq_linear(K, target_O, bounds=(0,np.inf))
E = res['x']
print('expected',res['fun'])
im = draw_config(E)
im.show()
O = score_image(im)
print(O - target_O)
