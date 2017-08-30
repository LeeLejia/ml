import tensorflow as tf

var = tf.Variable(0)  # global_variable集合中的第一个成员

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)  # 赋值

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 先初始化变量
    for _ in range(3):
        sess.run(update_operation)  # 运行赋值操作
        print(sess.run(var))
