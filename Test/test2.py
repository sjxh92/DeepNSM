import numpy as np

a = [[1, 2],
     [2, 4]]
b = [[1, 2],
     [2, 4]]

print(a + b)

a = np.ones(shape=(3, 5), dtype=int)
b = np.ones(shape=(3, 5), dtype=int)
c = np.ones(shape=(3, 5), dtype=int)

print(a + b + c)