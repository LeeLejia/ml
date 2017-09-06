import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 1, 20)
y = 2 * x ** 2
plt.figure()
plt.plot(x, y)
plt.show()


plt.xlabel('level')
plt.ylabel('count')
x = np.array([0, 1, 1.5, 2, 2.2])
y = np.array([5, 1, 6, 4, 8])
plt.xticks(x, ['0', 'loss', 'middle', 'ok', 'good'])
plt.yticks(y)
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--', label='square line')
plt.legend(loc='upper right')
plt.show()
