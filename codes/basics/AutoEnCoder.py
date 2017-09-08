"""
    自编码(非监督学习)
    cjwddz@qq.com
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 定义输入
in_x = tf.placeholder(tf.float32, shape=(None, 784))

# 编码层
l1 = tf.layers.dense(in_x, 256)
l2 = tf.layers.dense(l1, 128)

# 解码层
l25 = tf.layers.dense(l2, 128, tf.nn.sigmoid)
l3 = tf.layers.dense(l25, 256, tf.nn.sigmoid)
out = tf.layers.dense(l3, 784, tf.nn.sigmoid)

loss = tf.reduce_mean((in_x - out)**2)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    total = int(mnist.train.num_examples/50)
    for time in range(20):
        for i in range(total):
            x, y = mnist.train.next_batch(50)
            _, c = sess.run([optimizer, loss], feed_dict={in_x: x})
            if time % 10 == 0:
                print("time: %d" % (time+1), "loss=", "{:.4f}".format(c))
    print("finished train!")
    plt.figure()
    pred = sess.run(out, feed_dict={in_x: mnist.test.images[:5]})
    f, display = plt.subplots(2, 5, figsize=(5, 2))
    for i in range(5):
        display[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        display[1][i].imshow(np.reshape(pred[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
    print("ok")
