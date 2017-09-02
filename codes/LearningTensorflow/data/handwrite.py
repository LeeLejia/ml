import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv

# 定义变量
p_x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
p_y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义数据源
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 初始化第一层卷积 卷积核5*5,1通道,32输出
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))  # 得到正态分布的预设值
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# 将输入转换为4d张量,-1:数据维度,28,28,颜色通道
x_image = tf.reshape(p_x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二层卷积
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 密集连接层
w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

# 第二次pooling后结果展开成一维向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 输出层之前加入dropout,减少过拟合
p_keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, p_keep_prob)

# 添加softmax层,类似softmax regression
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 训练设置
cross_entropy = -tf.reduce_mean(p_y * tf.log(y_conv))  # 交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 预测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(p_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

Train = False
saver = tf.train.Saver()
if Train:
    # 训练
    train_accuracy = 0
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_accuracy, _ = sess.run([accuracy, train_step], feed_dict={p_x: batch[0], p_y: batch[1], p_keep_prob: 1.0})
        if i % 100 == 0:
            print("step %d, accuracy %.4f" % (i, train_accuracy))
    print("训练完毕！当前模型准确率%.4f" % train_accuracy)
    saver.save(sess, "C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\src\\data\\handwrite")
else:
    # 读取训练的模型
    saver.restore(sess, "C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\src\\data\\handwrite")

# 是否评估模型
assess = False
if assess:
    # 抽一个数据评估模型
    mnist.train.next_batch(105)
    batch, label = mnist.train.next_batch(1)
    im = np.reshape(batch, (28, 28))
    img = Image.fromarray(np.uint8(im * 255))
    img.save('C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\src\\data\\4.png')
    # img.show()
    print("正确值", np.argmax(label, 1), " 预测：", np.argmax(sess.run(y_conv, feed_dict={p_x: batch, p_keep_prob: 1.0}), 1))
    # 使用自定义数据评估模型
    url = 'C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\data\\number'
    accuracy_sum = 0
    count = 0
    for parent, _, filenames in os.walk(url):
        arr = None
        for index, filename in enumerate(filenames):
            img = Image.open(os.path.join(parent, filename)).convert('L')
            im = np.array(img.resize([28, 28])) / 255
            im = im.reshape((1, 784))
            if index % 100 == 0 and index != 0:
                pred = np.argmax(sess.run(y_conv, feed_dict={p_x: arr, p_keep_prob: 1.0}), 1)
                check = (pred == int(parent[-1:]))
                # print(check)
                accuracy = np.mean(check)
                accuracy_sum += accuracy
                count = count + 1
                print("全部准确率：", accuracy_sum / count, "   当前批准确率：", accuracy)
            else:
                if arr is None:
                    arr = im
                else:
                    arr = np.vstack([arr, im])
    print("ok!")

# region 手写板实现
# 当鼠标按下时设置 要进行绘画
drawing = False
# 如果mode为True时就画矩形，按下‘m'变为绘制曲线
mode = True
last_x = -1
last_y = -1


# 创建回调函数，用于设置滚动条的位置
def draw_pic(event, x, y, flags, _):
    global drawing, mode, last_x, last_y, img
    # 当按下左键时，返回起始的位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        last_x = -1
        last_y = -1
    elif event == cv.EVENT_RBUTTONDOWN:
        # im = np.array(img.convert('L').resize([28, 28])) / 255
        im = cv.resize(img, (28, 28))/255
        # img = Image.fromarray(np.uint8(im*255))
        # img.show()
        im = im.reshape((1, 784))
        pred = np.argmax(sess.run(y_conv, feed_dict={p_x: im, p_keep_prob: 1.0}), 1)
        print("你手写的数字为：", pred)
        img = np.zeros((512, 512), np.uint8)
    # 当鼠标左键按下并移动则是绘画圆形，event可以查看移动，flag查看是否按下
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_LBUTTONDOWN:
        if drawing:
            if last_x != -1:
                cv.line(img, (last_x, last_y), (x, y), (255, 255, 255), 23)
            last_x = x
            last_y = y

img = np.zeros((512, 512), np.uint8)
cv.namedWindow('drawing', cv.WINDOW_NORMAL)
cv.setMouseCallback('drawing', draw_pic)

while True:
    cv.imshow('drawing', img)
    key = cv.waitKey(10) & 0xFFF
    if key == 27:
        break
    elif key == 'c':
        print("按下c")

cv.destroyAllWindows()
cv.img.release()
# endregion
