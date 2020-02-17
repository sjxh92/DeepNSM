import numpy as np

a = [[1, 2],
     [2, 4]]
b = [[1, 2],
     [2, 4]]

print(a + b)

a = np.arange(16).reshape(4, 4)
b = np.arange(16).reshape(4, 4)

c = a[0:1]
d = np.empty(shape=(4, 4), dtype=bool)
print(d)
d[:] = True
print(d)

print(a + b)

print(np.arange(4))
print(np.argmax(a, axis=0))

x = np.array([True, False, True, False])
y = np.array([False, True, False, True])
print(x & y)

c = np.empty(shape=())
print(a)
print(b)
print(np.concatenate((a, b), axis=0))

print(np.sum(a[0]))
