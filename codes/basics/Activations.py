"""
    激励函数
    cjwddz@qq.com
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)

with tf.Session() as sess:
    plt.figure()
    # 负数置0，保留正数
    plt.plot(x, sess.run(tf.nn.relu(x)), c='blue', label='relu')
    # 将x映射到0-x之间
    plt.plot(x, sess.run(tf.nn.softplus(x)), c='green', label='softplus')
    # 将值映射到-1和1之间
    plt.plot(x, sess.run(tf.nn.tanh(x)), c='red', label='tanh')
    # sigmoid将值映射到0-1之间
    plt.plot(x, sess.run(tf.nn.sigmoid(x)), c='black', label='sigmoid')

    plt.legend(loc='best')
    plt.show()
