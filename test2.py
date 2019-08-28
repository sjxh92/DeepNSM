from scipy.stats import poisson
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# Poisson分布
x = np.random.poisson(lam=5, size=1000)  # lam为λ size为k
pillar = 25
a = plt.hist(x, bins=pillar, density=True, range=[0, pillar], color='g', alpha=0.2)
plt.plot(a[1][0:pillar], a[0], 'r')
plt.grid()
# plt.show()

rateParameter = 1 / 100


def nextTime(rate_parameter):
    return -math.log(1.0 - random.random()) / rate_parameter


print(nextTime(1 / 100))

np.random.seed(0)
print(np.random.rand(4))
print(np.zeros([1000, 4], dtype=int))

x = np.empty(shape=[0, 4], dtype=int)
x = np.append(x, [[1, 2, 3, 4]], axis=0)
x = np.append(x, [[1, 2, 2, 4]], axis=0)
x = np.append(x, [[1, 2, 3, 4]], axis=0)
x = np.append(x, [[1, 2, 1, 4]], axis=0)
x = np.append(x, [[1, 2, 4, 4]], axis=0)
x = np.append(x, [[1, 2, 6, 4]], axis=0)
x = np.append(x, [[1, 2, 5, 4]], axis=0)
print(x)
y = x[..., 2]
print(y)
print(np.where(y == 3))

