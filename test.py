import tensorflow as tf
import numpy as np

a = [[1, 2, 3],
     [4, 5, 6]]

a_float = tf.cast(a, tf.float32)

b = [[7, 8, 9],
     [10, 11, 12]]

b_float = tf.cast(b, tf.float32)

c = tf.squared_difference(a_float, b_float)

d = tf.reduce_mean(c)


with tf.Session() as sess:
    x, y = sess.run([c, d])
print(x)

print(y)
print('--------------------------------')

a = np.array([1, 2, 3])
print(a)
a = a[np.newaxis]
print(a)
print('--------------------------------')

a = 10
b = 3

c = np.random.choice(a, b, replace=True, p=None)

print(c)
print('--------------------------------')

a = np.array([[1, 2, 3, 4, 5, 6],
     [11, 22, 33, 44, 55, 66],
     [111, 222, 333, 444, 555, 666],
     [7, 8, 9, 10, 11, 12],
     [13, 14, 15, 16, 17, 18]])
print(a.shape)
b = [0, 2, 4]

c = a[b, :]
print(c)

with tf.variable_scope('eval_net'):
    diyi = 1
    dier = 2
    print(diyi, dier)

print(diyi, dier)

batch_memory = a.copy()
print(batch_memory)
print('--------------------------------')
print(batch_memory[:, -6:])
print('--------------------------------')
print(np.argmax(batch_memory, axis=1))

print(np.arange(6))

a = np.array([1, 1, 1, 1, 1, 1])
b = np.array([2])
# a = a[np.newaxis, :]
# b = b[np.newaxis, :]
print('11111111111111111111111111111')
print(a)
print(b)
print([a, b])
print(np.hstack(([[1, 2, 3, 4]], [a, b], [[1, 3, 5, 7]])))
