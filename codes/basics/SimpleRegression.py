"""
    简单线性回归
    cjwddz@qq.com
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 准备数据和标签
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = x ** 2 + noise

# 定义输入输出变量
in_x = tf.placeholder(dtype=tf.float32, shape=(None, 1))
in_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
# 定义神经网络
l1 = tf.layers.dense(in_x, 10, tf.nn.relu)
output = tf.layers.dense(l1, 1)
# 计算损失表达式
loss = tf.losses.mean_squared_error(in_y, output)
# 训练优化
opt = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 画出训练的点集
plt.figure()
plt.scatter(x, y)
# 用相同的数据多次训练
for step in range(100):
    _, pred = sess.run([opt, output], feed_dict={in_x: x, in_y: y})
    plt.cla()                       # 清除窗口
    plt.scatter(x, y)               # 画出训练点集
    plt.plot(x, pred, 'r-', lw=3)   # 画出预测
    plt.pause(0.1)                  # 暂停
# 输入一个值,并输出预测值
pred = sess.run(output, feed_dict={in_x: [[0.5]]})
print(pred)
plt.show()                          # 保持窗口
