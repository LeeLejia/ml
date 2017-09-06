import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 定义数据源
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 调整内容区域位置
def adjust_content(img):
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

# 定义输入
p_x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
p_y = tf.placeholder(dtype=tf.int32, shape=(None, 1))

l1 = tf.layers.dense(p_x, 10, tf.nn.relu)
out = tf.layers.dense(l1, 10)

# 损失计算
loss = tf.losses.sparse_softmax_cross_entropy(labels=p_y, logits=out)
accuracy = tf.metrics.accuracy(labels=tf.squeeze(p_y), predictions=tf.argmax(out, axis=1),)[1]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# 是否进入训练
Train = True
saver = tf.train.Saver()
if Train:
    # 训练
    plt.figure()
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        _, feed_data = adjust_content(batch[0])
        a = feed_data
        # plt.cla()
        # plt.scatter(a[:, 0], a[:, 1], c=np.argmax(batch[1], axis=1))
        # plt.pause(0.1)
        train_accuracy, _ = sess.run([accuracy, train_op], feed_dict={p_x: feed_data, p_y: np.argmax(batch[1], axis=1).reshape((feed_data.shape[0], 1))})
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
    _, batch = adjust_content(batch)
    print("正确值", np.argmax(label, 1), " 预测：", np.argmax(sess.run(out, feed_dict={p_x: batch}), 1))
    # 使用自定义数据评估模型
    url = 'C:\\Users\\lejia\\Desktop\\git-project\\ml\\codes\\LearningTensorflow\\data\\number'
    accuracy_sum = 0
    count = 0
    for parent, _, filenames in os.walk(url):
        arr = None
        for index, filename in enumerate(filenames):
            img = Image.open(os.path.join(parent, filename)).convert('L')
            im = np.array(img.resize([28, 28])) / 255
            _, im = adjust_content(im.reshape((1, 784)))
            if index % 100 == 0 and index != 0:
                label = np.zeros([arr.shape[0], 10])
                label[:, int(parent[-1:])] = 1
                pred, _ = sess.run([out, train_op], feed_dict={p_x: arr, p_y: np.argmax(batch[1], axis=1).reshape((feed_data.shape[0], 1))})
                pred = np.argmax(pred, 1)
                check = (pred == int(parent[-1:]))
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
    global drawing, last_x, last_y, img
    # 当按下左键时，返回起始的位置坐标
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        last_x = -1
        last_y = -1
    elif event == cv.EVENT_RBUTTONDOWN:
        im = cv.resize(img, (28, 28))/255
        _, im = adjust_content(im.reshape((1, 784)))
        pred = sess.run(out, feed_dict={p_x: im})
        print("你手写的数字最可能为：", np.argmax(pred, 1), "机会：", pred)
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
