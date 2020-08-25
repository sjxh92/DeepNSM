import numpy as np
a = np.empty(5)
print(a)
a[0] = 12
a[1] = 23
a[2] = 15
a[3] = 3
a[4] = 15

b = np.argsort(a)

print(b)
print(b[0])

a = np.zeros(shape=(2, 3))
print('---------------------')
print(a.shape[0])
print(a.shape[1])

