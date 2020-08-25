import numpy as np

a = {}
a[1] = [2, 3, 4]
a[2] = [5, 6]
a[3] = []
a[6] = []

a[1].append(7)
a[3].append(8)
a[6].append(11)
a[6].append(12)
print(a)
print(len(a[3]))
a[3].append(9)
print(len(a[3]))


