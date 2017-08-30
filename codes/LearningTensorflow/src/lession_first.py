import tensorflow as tf


def main(_):
    # 定义两个矩阵
    l = tf.constant([[3, 3]])
    r = tf.constant([[2], [2]])
    # 定义一个矩阵的乘法
    k = tf.matmul(r, l)
    # 创建Session对象，并运算
    se = tf.Session()
    result = se.run(k)
    print(result)
    return

if __name__ == '__main__':
    tf.app.run(main=main)



