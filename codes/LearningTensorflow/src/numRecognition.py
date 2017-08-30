import cv2 as cv
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np


class NumRec(object):
    # 初始化
    def init(self, isVisual=False):
        self.isVisual = isVisual
        self.mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        self.sess = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.sess.run(tf.global_variables_initializer())
        self.W_conv1 = self.weight_variable([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        self.W_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv2 = self.bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # if isVisual:
        #     self._visual()
        #     self.summary_str=self.sess.run(tf.global_variables_initializer())
        # else:
        self.sess.run(tf.global_variables_initializer())

    # 训练
    def train(self):
        for i in range(1000):
            batch = self.mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("-->step %d, training accuracy %.4f" % (i, train_accuracy))
                # if self.isVisual:
                #     self.summary_writer.add_summary(self.summary_str,i)
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

    # 测试
    def test(self):
        print("准确率测试中..")
        batch_size = 50
        batch_num = int(self.mnist.test.num_examples / batch_size)
        test_accuracy = 0
        for i in range(batch_num):
            batch = self.mnist.test.next_batch(batch_size)
            test_accuracy += self.accuracy.eval(feed_dict={
                self.x: batch[0],
                self.y_: batch[1],
                self.keep_prob: 1.0})
        test_accuracy /= batch_num
        print("测试集准确率：%g" % test_accuracy)

    # 一些初始化操作
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 存储，恢复数据
    def save(self, fileName='data'):
        saver = tf.train.Saver()
        saver.save(self.sess, "E:/project/python_project/LearningTensorflow/src/data/" + fileName)
        print("训练数据已缓存")

    def restore(self, fileName='data'):
        saver = tf.train.Saver()
        saver.restore(self.sess, "E:/project/python_project/LearningTensorflow/src/data/" + fileName)
        print("训练数据加载完毕")
        # 训练过程可视化
        # def _visual(self):
        #     tf.summary.scalar('x',self.x)
        #     self.merged_summary_op =tf.summary.merge_all()
        #     self.summary_writer = tf.train.summary.FileWriter(self.sess.graph,'E:\project\python_project\LearningTensorflow\data\dir','XXXX')


class PicHelper(object):
    def __init__(self, path):
        self.path = path

    def enhance(self):
        img = cv.imread(self.path)
        cv.imshow("img", img)
        cv.waitKey(0)


nr = NumRec()
nr.init(True)
# nr.train()
# nr.save()
# nr.restore("data")
nr.test()

'''
InteractiveSession和Session区别：
    InteractiveSession运行在没有指定会话对象的情况下运行变量。这是与Session（）最大的不同。
    Session（）使用with..as..后可以不使用close关闭对话，而调用InteractiveSession需要在最后调用close

建立完整模型过程：
    # 建立抽象模型
    x = tf.placeholder(tf.float32, [None, 784]) # 输入占位符
    y = tf.placeholder(tf.float32, [None, 10])  # 输出占位符（预期输出）
    W = tf.Variable(tf.zeros([784, 10]))        
    b = tf.Variable(tf.zeros([10]))
    a = tf.nn.softmax(tf.matmul(x, W) + b)      # a表示模型的实际输出
    
    # 定义损失函数和训练方法
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1])) # 损失函数为交叉熵
    optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降法，学习速率为0.5
    train = optimizer.minimize(cross_entropy)  # 训练目标：最小化损失函数
    
    # 测试模型
    correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1)) 
    #tf.argmax表示找到最大值的位置(也就是预测的分类和实际的分类)，然后equal判断是否一致（True,False）
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #tf.cast将boolean数组转成int数组，最后求平均值，得到分类的准确率(怎么样，是不是很巧妙)
    
    # 训练
    sess = tf.InteractiveSession()      # 建立交互式会话
    tf.initialize_all_variables().run() # 所有变量初始化
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)    # 获得一批100个数据
        train.run({x: batch_xs, y: batch_ys})   # 给训练模型提供输入和输出
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
#训练过程可视化
    # 1. 由之前的各种运算得到此批数据的loss
        loss = ..... 
    # 2.使用tf.scalar_summary来收集想要显示的变量,命名为loss
        tf.scalar_summary('loss',loss)  
    # 3.定义一个summury op, 用来汇总由scalar_summary记录的所有变量
        merged_summary_op = tf.merge_all_summaries()
    # 4.生成一个summary writer对象，需要指定写入路径,例如我这边就是/tmp/logdir
        summary_writer = tf.train.SummaryWriter('/tmp/logdir', sess.graph)
    # 开始训练，分批喂数据
        for(i in range(batch_num)):
            # 5.使用sess.run来得到merged_summary_op的返回值
            summary_str = sess.run(merged_summary_op)
            # 6.使用summary writer将运行中的loss值写入
            summary_writer.add_summary(summary_str,i)
'''
