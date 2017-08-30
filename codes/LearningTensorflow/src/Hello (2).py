# -*- coding: utf-8 -*-
'''卷积神经网络测试MNIST数据'''
#导入MNIST数据
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
from PIL import Image

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

#权重初始化函数,用一个较小的正数来初始化偏置项
def weight_variable(shape):
    #tf.truncated_normal(shape,mean=0.0,stddev)：
    # shape：张量的维度；mean：正态分布的均值；stddev：正态分布的标准差
    # 从截断的正态分布中输出随机值。
    #生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积和池化函数
def conv2d(x, W):
    # strides=[1, x 方向的步长, y 方向的步长, 1]
    #padding='SAME' ：SAME 使卷积后的输出图像与原来一样， 0 填充
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
# 5*5 是卷积核大小， 1 是一个通道， 32 输出个数
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#把x变成一个4d向量
#tf.reshape(x, [-1:先不管数据维度,28,28,颜色通道])
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
#输出 28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#池化，输出：14*14*32
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#输出 14*14*64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#输出7*7*64
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#将第二次pooling 后的结果展成 一维 向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#添加一个softmax层，就像softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#训练设置
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #ADAM优化器来做梯度最速下降
#tf.argmax(y_conv,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

#训练
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("-->step %d, training accuracy %.4f"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#最终评估
print('测试集准确度：\n')

#切片测试，将分成 batch_num 次放入， 每次放入 batch_size
batch_size = 50
batch_num = int(mnist.test.num_examples / batch_size)
test_accuracy = 0

for i in range(batch_num):
    batch = mnist.test.next_batch(batch_size)
    test_accuracy += accuracy.eval(feed_dict={x: batch[0],
                                              y_: batch[1],
                                              keep_prob: 1.0})

test_accuracy /= batch_num
print("test accuracy %g" % test_accuracy)

batch_x, batch_y = mnist.test.next_batch(1)
#batch_x 为（1，784）数组（保存图像信息） batch_y 为（1,10）（保存图像标签，第几位数是1，就表示几）
print('验证训练数据的准确性：',sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y,keep_prob: 1.0}))  #验证训练数据的准确性
im = np.reshape(batch_x,(28,28))   #将一维数组转化为28*28的图像数组  float32 （0-1）
#此时通过观察数组中数字部分，能大致的看出图像表示的数字
#为了直观的看到，可以将数组转化为图像
from PIL import Image
imag=Image.fromarray(np.uint8(im*255))  #这里读入的数组是 float32 型的，范围是 0-1，而 PIL.Image 数据是 uinit8 型的，范围是0-255，要进行转换
imag.show()
imag.save('C:/Users/lejia/Pictures/7.png')

imm =np.array(Image.open("C:/Users/lejia/Pictures/7.png").convert('L')) #打开图片，转化为灰度并转化为数组size（n,m） 值0-255
imm = imm/255           #将值转化为0-1
imm_3 = Image.fromarray(imm)    #转化为图像
imm_4 = imm_3.resize([28,28])   #压缩
im_array = np.array(imm_4)     #转化为数组
fs = im_array.reshape((1,784))  #转化为符合验证一维的数组
print('输出模型的识别值',sess.run(tf.argmax(y_,1), feed_dict={x: fs})) #输出模型的识别值


#或者
imm =np.array(Image.open("C:/Users/lejia/Pictures/7.png").convert('L').resize([28,28]))
#imm = 255-imm  #imm、255  反向处理
imm = imm/255
#imm = -imm+1   #自己测试图片效果太差，示例的数组无字处为0（黑底白字）。可以通过自定义函数转化自己的数组，这里利用的是最简单的 函数
imm = imm.reshape((1,784))

print('tf.argmax 算出模型值:',sess.run(tf.argmax(y_,1), feed_dict={x: imm}))  #tf.argmax 算出模型值


