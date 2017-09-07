"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3], [3]])
dot_operation = tf.matmul(m1, m2)

# 错误！该处并未进行运算
print(dot_operation)

# method1
sess = tf.Session()
result = sess.run(dot_operation)
print(result)
sess.close()

# method2
with tf.Session() as sess:
    result_ = sess.run(dot_operation)
    print(result_)
