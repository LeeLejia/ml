"""
    自编码(非监督学习)
    cjwddz@qq.com
"""

import numpy as np
import tensorflow as tf

in_x = tf.placeholder(tf.float32, shape=(50, 1))
in_y = tf.placeholder(tf.float32, shape=(50, 1))
