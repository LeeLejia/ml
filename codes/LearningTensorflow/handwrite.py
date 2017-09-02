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
train_step_2 = tf.train.AdamOptimizer(1e-4).minimize((tf.cast(tf.reduce_sum(y_conv), tf.float32)/tf.cast(tf.reduce_max(y_conv, 1), tf.float32)))

# 预测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(p_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# 调整内容区域位置
def adjust_content(img):
    arr = None
    if img.shape[1] != 784:
        raise Exception("不正确的数据类型！")
    for t in img:
        t.resize([28, 28])
        h = np.max(t, axis=1)
        v = np.max(t, axis=0)
        ta = np.argwhere(h)
        tb = np.argwhere(v)
        left = np.reshape(tb[0], 1)
        right = np.reshape(tb[-1:], 1)
        top = np.reshape(ta[0], 1)
        bottom = np.reshape(ta[-1:], 1)
        h_center = right-left
        cut = 28-bottom+top   # 垂直切除
        excursion_l = int(left if (cut/2) >= left else (cut/2))              # 左边偏移
        excursion_r = int((27-right) if (cut/2) >= (27-right) else (cut/2))  # 右边偏移
        # 截取关键区域图片平移
        tt = t[top:bottom, excursion_l:27-excursion_r]
        if np.sum(tt.shape) < 6 or np.min(tt.shape) == 0:
            arr = t.reshape((1, 784)) if arr is None else np.vstack([arr, t.reshape((1, 784))])
        tt = cv.resize(tt, (28, 28), interpolation=cv.INTER_CUBIC)
        m = np.float32([[1, 0, 15-h_center], [0, 1, 0]])
        tt = cv.warpAffine(tt, m, (28, 28))
        tt = np.float32((tt > 0.2))
        tabel = (tt > 0)
        hs_max_value = np.max(np.sum(tabel, axis=0))
        vs_max_index = np.argmax(np.sum(tabel, axis=1))
        arr = tt.reshape((1, 784)) if arr is None else np.vstack([arr, tt.reshape((1, 784))])
    return arr, [hs_max_value, vs_max_index]

# 是否进入训练
Train = False
saver = tf.train.Saver()
if Train:
    # 训练
    train_accuracy = 0
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        feed_data, _ = adjust_content(batch[0])
        train_accuracy, _, _ = sess.run([accuracy, train_step, train_step_2], feed_dict={p_x: feed_data, p_y: batch[1], p_keep_prob: 1.0})
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
    batch, _ = adjust_content(batch)
    im = np.reshape(batch, (28, 28))
    img = Image.fromarray(np.uint8(im * 255))
    # img.save('C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\src\\data\\4.png')
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
            im, _ = adjust_content(im.reshape((1, 784)))
            if index % 100 == 0 and index != 0:
                label = np.zeros([arr.shape[0], 10])
                label[:, int(parent[-1:])] = 1
                pred, _ = sess.run([y_conv, train_step], feed_dict={p_x: arr, p_y: label, p_keep_prob: 1.0})
                pred = np.argmax(pred, 1)
                check = (pred == int(parent[-1:]))
                # print(check)
                accuracy = np.mean(check)
                accuracy_sum += accuracy
                count = count + 1
                if accuracy <= 0.3:
                    print("可能错误的样本:", parent[-1:], "last:", filename, "accuracy:", accuracy)
                print("全部准确率:", accuracy_sum / count, "   当前批准确率:", accuracy, "完成:", count*100)
                if count % 10 == 0 and count != 0:
                    break
            else:
                if arr is None:
                    arr = im
                else:
                    arr = np.vstack([arr, im])
    saver.save(sess, "C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\src\\data\\handwrite")
    print("ok!")

# region 手写板实现
drawing = False
last_x = -1
last_y = -1


# 创建回调函数
def draw_pic(event, x, y, flags, _):
    global drawing, mode, last_x, last_y, img
    # 当按下左键时，返回起始的位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        last_x = -1
        last_y = -1
    elif event == cv.EVENT_RBUTTONDOWN:
        im = cv.resize(img, (28, 28))/255
        # img = Image.fromarray(np.uint8(im*255))
        # img.show()
        im,_ = adjust_content(im.reshape((1, 784)))
        pred = sess.run(y_conv, feed_dict={p_x: im, p_keep_prob: 1.0})
        print("你手写的数字为:", np.argmax(pred, 1), "可信度:%0.4f " % (np.float32(np.max(pred, 1))/np.float32(np.sum(pred))), np.int32(pred > 1e-02)*[[10, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
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
