"""
    分类学习
    cjwddz@qq.com
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义输入
in_x = tf.placeholder(tf.float32, (None, 2))
in_y = tf.placeholder(tf.int32, (None, ))

# 定义网络
l1 = tf.layers.dense(in_x, 10, tf.nn.relu)
out = tf.layers.dense(l1, 2)

# 定义损失
# loss = tf.reduce_mean(tf.argmax(out, axis=1) - in_y, axis=1)
# loss = tf.losses.mean_squared_error(in_y, out)
# # loss = tf.losses.sparse_softmax_cross_entropy(labels=in_y, logits=out)
# train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)


loss = tf.losses.sparse_softmax_cross_entropy(labels=in_y, logits=out)           # compute cost
accuracy = tf.metrics.accuracy(labels=tf.squeeze(in_y), predictions=tf.argmax(out, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = optimizer.minimize(loss)

# 定义数据
d = np.ones(shape=(200, 2))
d_x1 = np.random.normal(0.7*d, 1)
d_x2 = np.random.normal(-0.7*d, 1)

d_y1 = np.zeros((200,))
d_y2 = np.ones((200,))

d_x = np.vstack((d_x1, d_x2))
d_y = np.hstack((d_y1, d_y2))

# 可视化
plt.figure()
plt.scatter(d_x1[:, 0], d_x1[:, 1], c='red')
plt.scatter(d_x2[:, 0], d_x2[:, 1], c='blue')
plt.show()
# 开始训练
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for step in range(1000):
    _, pred = sess.run([train_step, out], feed_dict={in_x: d_x, in_y: d_y})
    plt.cla()
    a = np.argmax(pred, 1)
    plt.scatter(d_x[:, 0], d_x[:, 1], c=a)
    plt.pause(0.1)
    print("step:%d " % step)
print('OK')
