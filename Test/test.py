import numpy as np

a = np.array([1, 2, 3])
b = np.array([11, 22, 33])
c = np.array([44, 55, 66])

d = np.concatenate((a, b, c), axis=0)
print(d)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[11, 21, 31], [7, 8, 9]])
c = np.concatenate((a, b), axis=0)
print(c)

a = np.zeros(shape=(10, 10), dtype=int)
print(a)
b = np.zeros(shape=(9, 10), dtype=int)
print(b)
c = np.concatenate((a, b), axis=0)
print(c.shape)
d = [[0 for _ in range(10)]]
print(d)
e = np.concatenate((c, d), axis=0)
print(e)
print(e.shape)
print('a:', a)

