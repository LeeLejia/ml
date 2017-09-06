import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image


def main(_):

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    # 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None, 784]。（这里的None表示此张量的第一个维度可以是任何长度的。）
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 我们赋予tf.Variable不同的初值来创建不同的Variable：在这里，我们都用全为零的张量来初始化W和b。
    # 因为我们要学习W和b的值，它们的初值可以随意设置。
    # 注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
    # b的形状是[10]，所以我们可以直接把它加到输出上面。

    y = tf.nn.softmax(tf.matmul(x, w) + b)
    # 我们用tf.matmul(​​X，W)表示x乘以W，这里x是一个2维张量拥有多个输入。
    # 然后再加上b，把和输入到tf.nn.softmax函数里面

    y_ = tf.placeholder("float", [None, 10])
    # 计算交叉熵
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # 训练模型
    # 为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。
    # 其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss）
    # 然后尽量最小化这个指标。但是，这两种方式是相同的。
    # 一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。
    # 交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。它的定义如下：
    # y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)。
    # 比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性。
    # 更详细的关于交叉熵的解释超出本教程的范畴，但是你很有必要好好理解它。
    # 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
    # 梯度下降算法（gradient descent algorithm）是一个简单的学习过程
    # TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。
    # 当然TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法。

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 我们已经设置好了我们的模型。在运行计算之前，我们添加一个操作来初始化我们创建的变量

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # 开始训练模型，这里我们让模型循环训练1000次！
    # 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点
    # 然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
    # 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。
    # 在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。
    # 所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 评估我们的模型
    # 首先让我们找出那些预测正确的标签。
    # tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
    # 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
    # 比如 tf.argmax(y,1) 返回的是模型对于任一输入x预测到的标签值
    # 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
    # 这行代码会给我们一组布尔值。
    # 为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
    # 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # 计算学习到的模型在测试数据集上面的正确率
    # #该模型准确率是：0.9156

    # region 自定义的预览中间数据
    print("自定义预览...")
    batch_x, batch_y = mnist.test.next_batch(1)   # 取一组训练数据
    # batch_x 为（1，784）数组（保存图像信息） batch_y 为（1,10）（保存图像标签，第几位数是1，就表示几）
    print(sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))  # 验证训练数据的准确性
    im = np.reshape(batch_x, (28, 28))
    # 将一维数组转化为28*28的图像数组  float32 （0-1）
    # 此时通过观察数组中数字部分，能大致的看出图像表示的数字
    # 为了直观的看到，可以将数组转化为图像

    # endregion

    imag=Image.fromarray(np.uint8(im*255))
    # 这里读入的数组是 float32 型的，范围是 0-1，而 PIL.Image 数据是 uinit8 型的，范围是0-255，要进行转换
    imag.show()
    imag.save('C:/Users/lejia/Pictures/7.png')
    print("自定义的测试...")

    imm =np.array(Image.open('C:/Users/lejia/Pictures/7.png').convert('L'))  # 打开图片，转化为灰度并转化为数组size（n,m） 值0-255
    imm = imm/255                   # 将值转化为0-1
    imm_3 = Image.fromarray(imm)    # 转化为图像
    imm_4 = imm_3.resize([28,28])   # 压缩
    im_array = np.array(imm_4)      # 转化为数组
    fs = im_array.reshape((1,784))  # 转化为符合验证一维的数组
    print('before')
    print(sess.run(tf.argmax(y,1), feed_dict={x: fs}))  # 输出模型的识别值
    print('after')
    # 或者 自己数据的话需要反色

    imm =np.array(Image.open('C:/Users/lejia/Pictures/2.png').convert('L').resize([28,28]))

    imagg=Image.fromarray(np.uint8(imm*255))
    imagg.show()

    imm = 255-imm  # imm、255  反向处理
    imm = imm/255
    imm = -imm+1   # 自己测试图片效果太差，示例的数组无字处为0（黑底白字）。可以通过自定义函数转化自己的数组，这里利用的是最简单的 函数

    imaggg=Image.fromarray(np.uint8(imm*255))
    imaggg.show()

    imm = imm.reshape((1,784))
    print("x:",type(x)," imm:",type(imm))
    print(sess.run(tf.argmax(y,1), feed_dict={x: imm}))  # tf.argmax 算出模型值


if __name__ == '__main__':
    tf.app.run(main=main)


def check_single_pic(sess, path, x, y, format=1):
    imm = np.array(Image.open(path).convert('L'))  # 打开图片，转化为灰度并转化为数组size（n,m） 值0-255
    # 将值转化为0-1
    if format == 0:
        imm = 255-imm  # imm、255  反向处理
        imm = imm/255
        imm = -imm+1   # 自己测试图片效果太差，示例的数组无字处为0（黑底白字）。可以通过自定义函数转化自己的数组，这里利用的是最简单的 函数
    imm = imm/255
    imm_3 = Image.fromarray(imm)    # 转化为图像
    imm_4 = imm_3.resize([28, 28])   # 压缩
    im_array = np.array(imm_4)      # 转化为数组
    fs = im_array.reshape((1, 784))  # 转化为符合验证一维的数组
    return sess.run(tf.argmax(y, 1), feed_dict={x: fs})  # 输出模型的识别值
