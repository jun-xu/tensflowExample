from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt  # 绘图
import numpy as np
from sklearn.metrics import confusion_matrix  # 混淆矩阵，分析模型误差
from tensorflow.examples.tutorials.mnist import input_data

### 载入数据

mnist = input_data.read_data_sets("../data/", one_hot=True)

print("数据集大小")
print(' - 训练集 {}'.format(len(mnist.train.labels)))
print(' - 测试集 {}'.format(len(mnist.test.labels)))
print(' - 验证集 {}'.format(len(mnist.validation.labels)))

print(mnist.test.labels[:5])  # 前5张图片
print(np.argmax(mnist.test.labels[:5], axis=1))  # 前5张图片的值

### 数据维度

print(" 样本维度: ", mnist.train.images.shape)
print(" 标签维度: ", mnist.train.labels.shape)

### 参数定义

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

num_classes = 10


### 打印部分图片

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每张图之间的间隔

    for i, ax in enumerate(axes.flat):
        # 绘图，将一维向量变为二维矩阵，黑白二值图像使用 binary
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = " True:{0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        # 删除坐标信息
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


### 测试打印图片

# indices = np.arange(len(mnist.test.labels))
# np.random.shuffle(indices)
# indices = indices[:9]
#
# images = mnist.test.images[indices]
# cls_true = np.argmax(mnist.test.labels[indices], axis=1)
#
# plot_image(images, cls_true)


### 设置参数

x = tf.placeholder(tf.float32, shape=[None, img_size_flat])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes])
y_true_cls = tf.placeholder(tf.int64, shape=[None])

weights = tf.Variable(tf.random_normal([img_size_flat, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))

### 模型

# logits = tf.nn.relu(tf.matmul(x, weights) + biases)

logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算准确率

### 运行
batch_size = 1000


def optimize(num_iterations, sess):
    for i in range(num_iterations):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: x_batch, y_true: y_batch})


test_feed_dict = {x: mnist.test.images, y_true: mnist.test.labels,
                  y_true_cls: np.argmax(mnist.test.labels, axis=1)}


def print_accuracy(sess):
    acc = sess.run(accuracy, feed_dict=test_feed_dict)
    print("测试集准确率: {0:.1%}".format(acc))  # 保留1位小数


def plot_example_errors(sess):
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict=test_feed_dict)
    incorrect = (correct == False)
    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.labels[incorrect]

    # 随机挑选9个
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    indices = indices[:9]

    plot_images(images[indices], np.argmax(cls_true[indices], axis=1), cls_pred[indices])


def print_confusion_matrix(sess):
    cls_true = np.argmax(mnist.test.labels, axis=1)  # 真实类别
    # 运行y_pred_cls计算出的真实类别
    cls_pred = sess.run(y_pred_cls, feed_dict=test_feed_dict)

    # 使用scikit-learn的confusion_matrix来计算混淆矩阵
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # 打印混淆矩阵
    print(cm)

    # 将混淆矩阵输出为图像
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # 调整图像
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_weights(sess):
    # Get the values for the weights from the TensorFlow variable.
    w = sess.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < 10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # optimize(num_iterations, sess)
    print_accuracy(sess)
    plot_example_errors(sess)

    # optimize(num_iterations=1, sess=sess)
    # print_accuracy(sess)
    # plot_example_errors(sess)
    # print_confusion_matrix(sess)
    # plot_weights(sess)

    optimize(num_iterations=1000, sess=sess)
    print_accuracy(sess)
    plot_example_errors(sess)
    print_confusion_matrix(sess)
    plot_weights(sess)
