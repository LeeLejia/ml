import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# 调整内容区域位置
def adjust_content(img):
    return img, None
    arr = None
    features = None
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
        t = t[top:bottom, excursion_l:27-excursion_r]
        if np.array_equal(0, t.shape):
            raise Exception("这里可能有个bug！")
        t = cv.resize(t, (28, 28), interpolation=cv.INTER_CUBIC)
        m = np.float32([[1, 0, 15-h_center], [0, 1, 0]])
        t = cv.warpAffine(t, m, (28, 28))
        t = np.float32((t > 0.2))
        tabel = (t > 0)
        hs_max_value = np.max(np.sum(tabel, axis=0))
        vs_max_index = np.argmax(np.sum(tabel, axis=1))
        arr = t.reshape((1, 784)) if arr is None else np.vstack([arr, t.reshape((1, 784))])
        features = [[hs_max_value, vs_max_index]] if features is None else np.vstack([features, [[hs_max_value, vs_max_index]]])
    return arr, features

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
# 梯度下降算法（gradient descent algorithm）是一个简单的学习过程
# TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。
# 当然TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法。

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    in_x, _ = adjust_content(batch_xs)
    sess.run([train_step, y], feed_dict={x: in_x, y_: batch_ys})
    print("进度：", i/1000)
print('OK!')
# 开始训练模型，这里我们让模型循环训练1000次！
# 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点
# 然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
# 使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。
# 在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。
# 所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# 计算学习到的模型在测试数据集上面的正确率
# 该模型准确率是：0.9156

# region 手写板实现
drawing = False
last_x = -1
last_y = -1


# 创建回调函数
def draw_pic(event, mouse_x, mouse_y, flags, _):
    global drawing, last_x, last_y, img
    # 当按下左键时，返回起始的位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        last_x = -1
        last_y = -1
    elif event == cv.EVENT_RBUTTONDOWN:
        im = cv.resize(img, (28, 28))/255
        im, _ = adjust_content(im.reshape((1, 784)))
        pred = sess.run(y, feed_dict={x: im})
        print("你手写的数字为:", np.argmax(pred, 1), "可信度:%0.4f " % (np.float32(np.max(pred, 1))/np.float32(np.sum(pred))), np.int32(pred > 1e-02)*[[10, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        img = np.zeros((512, 512), np.uint8)
    # 当鼠标左键按下并移动则是绘画圆形，event可以查看移动，flag查看是否按下
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_LBUTTONDOWN:
        if drawing:
            if last_x != -1:
                cv.line(img, (last_x, last_y), (mouse_x, mouse_y), (255, 255, 255), 23)
            last_x = mouse_x
            last_y = mouse_y

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





