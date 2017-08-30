import tensorflow as tf

x1 = tf.placeholder(dtype=tf.float32, shape=None)
y1 = tf.placeholder(dtype=tf.float32, shape=None)
z1 = x1 + y1

x2 = tf.placeholder(dtype=tf.float32, shape=[2, 1])
y2 = tf.placeholder(dtype=tf.float32, shape=[1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    # when only one operation to run
    z1_value = sess.run(z1, feed_dict={x1: 1, y1: 2})

    # when run multiple operations
    z1_value, z2_value = sess.run(
        [z1, z2],       # run them together
        feed_dict={
            x1: 1, y1: 2,
            x2: [[2], [2]], y2: [[3, 3]]
        })
    print(z1_value)
    print(z2_value)



'''
 1.
    dtype=tf.float32, shape=[1,2]对应矩阵：[[3.,3.]],及两行一列
 2.  
    x=tf.constant([[1.],[2.]]) //两列
    y=tf.constant([[3.,4.]])   //两行
    op=tf.matmul(y,x)
    run得到值：
        [[ 11.]]
 3. 
    sess.run(运算，初始值键值对集合)
    sess.run(变量或对象)

'''