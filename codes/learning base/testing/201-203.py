#  实现一个矩阵常量和一个矩阵变量相乘

import tensorflow as tf

a = tf.constant([[2.], [2.]])
b = tf.placeholder(tf.float32, shape=(1, 2))
op = tf.matmul(a, b)
c = tf.Variable([[2., 2.], [2., 2.]])
op2 = tf.assign(c, op)

tf.initialize_all_variables()
with tf.Session() as sess:
    print("operation:", sess.run(op2, feed_dict={b: [[2., 3.]]}))
    print("variable:", sess.run(c))

