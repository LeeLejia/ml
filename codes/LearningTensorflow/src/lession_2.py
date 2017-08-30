from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 加载MNIST数据
    # 为了方便起见，我们已经准备了一个脚本来自动下载和导入MNIST数据集。它会自动创建一个'MNIST_data'的目录来存储数据。
    # 这里，mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。
    # 同时提供了一个函数，用于在迭代中获得minibatch，后面我们将会用到。

    sess = tf.InteractiveSession()
    # 使用InteractiveSession可以更加灵活地构建你的代码。
    # 它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。
    # 这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。
    # 如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图。


# 权重初始化
# 为了创建这个模型，我们需要创建大量的权重和偏置项。
# 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项
# 以避免神经元节点输出恒为0的问题（dead neurons）。为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积和池化
# TensorFlow在卷积和池化上有很强的灵活性。
# 我们怎么处理边界？步长应该设多大？
# 在这个实例里，我们会一直使用vanilla版本。
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板
# 保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling。
# 为了代码更简洁，我们把这部分抽象成一个函数。


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def test():
    # 第一层卷积
    # 现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
    # 卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]
    # 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
    # 而对于每一个输出通道都有一个对应的偏置量。
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
    x_image = tf.reshape(x, [-1,28,28,1])
    # 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 第二层卷积
    # 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # 密集连接层
    # 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    # 为了减少过拟合，我们在输出层之前加入dropout。
    # 我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    # 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
    # TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
    # 所以用dropout的时候可以不用考虑scale。
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # 输出层
    # 最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)



