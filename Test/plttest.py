import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 50)
y1 = x**2 + 1
y2 = 2 * x + 1

##plt.figure()
##plt.plot(x, y1)

plt.figure(num=5, figsize=(8,5))
plt.plot(x, y2, label='up')
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('i am x')
plt.ylabel('i am y')
plt.legend()

new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
plt.show()
