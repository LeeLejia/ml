"""
    自编码(非监督学习)
    cjwddz@qq.com
"""

import numpy as np
import tensorflow as tf

a = tf.constant([-1.0, 2.0])
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))
