from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import confusion_matrix  # 混淆矩阵，分析模型误差
from tensorflow.examples.tutorials.mnist import input_data

# 读取测试数据
mnist = input_data.read_data_sets('../data/', one_hot=True)
# print("data.train.images:", mnist.train.images[0], type(mnist.train.images[0]))
image_size = 28
image_2d_size = image_size * image_size
num_classes = 10


def weights2d(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.5, name=name))


def biases2d(shape, name=None):
    return tf.Variable(tf.constant(shape=shape, value=0.05), name=name)


def conv2d(input, in_channels, filter_size, out_channels, name, user_pooling=True):
    shape = [filter_size, filter_size, in_channels, out_channels]

    weights = weights2d(shape=shape, name=name + '_w')
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    biases = biases2d([out_channels], name=name + '_b')

    layer += biases
    middle_layer = layer
    if user_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights, biases, middle_layer


x = tf.placeholder(tf.float32, shape=[None, image_2d_size], name='x')
input_layer = tf.reshape(x, shape=[-1, image_size, image_size, 1])

filter_size = 5
input_channel = 1
layer_1_out_channel = 10
layer_2_out_channel = 20

# conversions
# 定义知识层
layer_1, layer_w1, biases_1, middle_layer1 = conv2d(input_layer,
                                                    in_channels=input_channel,
                                                    filter_size=filter_size,
                                                    out_channels=layer_1_out_channel,
                                                    name='layer_1')

print(" ------------  ")
print("layer_w1:", layer_w1)
print("middle_layer1:", middle_layer1)
print("layer_1: ", layer_1)

layer_2, layer_w2, biases_2, middle_layer2 = conv2d(layer_1,
                                                    in_channels=layer_1_out_channel,
                                                    filter_size=filter_size,
                                                    out_channels=layer_2_out_channel,
                                                    name='layer_2')

print(" ------------  ")
print("layer_w2:", layer_w2)
print("middle_layer2:", middle_layer2)
print("layer_2: ", layer_2)


# def show_middle_filters_of_images(sess, middle_layer):
# wShape = weights.get_shape()
# images = []
#
# for i in range(9):
#     xx = np.zeros(shape=[1, image_size * image_size], dtype=np.float32)
#     # 取第一个filter返回的所有的out_channel之和
#     arg_max = tf.reduce_sum(tf.reduce_sum(middle_layer, axis=1), axis=0)[0, 0]
#     max_cost = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(-arg_max)
#     # 训练x图片
#     for j in range(10):
#         sess.run(max_cost, feed_dict={x: xx})
#     img = sess.run(tf.reshape(xx, [image_size * image_size]))
#     images.append(img)
#
# print(images[0])
# plot_images(images, [0, 0, 0, 0, 0, 0, 0, 0, 0])


# 定义 全连接层

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:len(layer_shape)].num_elements()

    layer_flatten = tf.reshape(layer, shape=[-1, num_features])
    return layer_flatten, num_features


def dence2d(input, in_channels, out_channels, use_relu=True):
    weights = weights2d(shape=[in_channels, out_channels], name='d_w')
    biases = biases2d(shape=[out_channels], name='d_b')

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y, axis=1)

layer_flat, num_feature = flatten_layer(layer_2)
print("layer_flat: ", layer_flat)

fc_size = 1024
layer_fc1 = dence2d(input=layer_flat,
                    in_channels=num_feature,
                    out_channels=fc_size,
                    use_relu=True)
print("layer_fc1:", layer_fc1)
layer_fc2 = dence2d(input=layer_fc1,
                    in_channels=fc_size,
                    out_channels=num_classes,
                    use_relu=False)

print("layer_fc2:", layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y,
                                                        name='cross_entropy')
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_pred = tf.equal(y_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_batch_size = 64
total_iteration = 0


def optimize(sess, num_iter):
    global total_iteration
    start_time = time.time()
    for i in range(total_iteration, total_iteration + num_iter):

        x_batch, y_batch = mnist.train.next_batch(train_batch_size)
        feed_dict = {
            x: x_batch,
            y: y_batch
        }
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print("迭代轮次: {0:>6}, 训练准确率: {1:>6.1%}".format(i + 1, acc))

    total_iteration += num_iter
    end_time = time.time()
    print("用时: " + str(timedelta(seconds=int(round(end_time - start_time)))))


def plot_images(images, cls_true, cls_pred=None):
    """
    绘制图像，输出真实标签与预测标签
    images:  图像（9张）
    cls_true: 真实类别
    cls_pred: 预测类别
    """
    assert len(images) == len(cls_true) == 9  # 保证存在9张图片

    fig, axes = plt.subplots(3, 3)  # 创建3x3个子图的画布
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整每张图之间的间隔

    for i, ax in enumerate(axes.flat):
        # 绘图，将一维向量变为二维矩阵，黑白二值图像使用 binary
        ax.imshow(images[i].reshape((image_size, image_size)), cmap='binary')

        if cls_pred is None:  # 如果未传入预测类别
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)

        # 删除坐标信息
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_confusion_matrix(cls_pred):
    cls_true = mnist.test.cls  # 真实类别

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


test_batch_size = 256

mnist.test.cls = np.argmax(mnist.test.labels, axis=1)


def print_test_accuracy(sess, show_example_errors=False,
                        show_confusion_matrix=False):
    # 测试集图像数量.
    num_test = len(mnist.test.images)

    # 为预测结果申请一个数组.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # 数据集的起始id为0
    i = 0
    while i < num_test:
        # j为下一批次的截止id
        j = min(i + test_batch_size, num_test)

        # 获取i，j之间的图像
        images = mnist.test.images[i:j, :]

        # 获取相应标签.
        labels = mnist.test.labels[i:j, :]

        # 创建feed_dict
        feed_dict = {x: images,
                     y: labels}

        # 计算预测结果
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        # 设定为下一批次起始值.
        i = j

    cls_true = mnist.test.cls
    # 正确的分类
    correct = (cls_true == cls_pred)
    # 正确分类的数量
    correct_sum = correct.sum()
    # 分类准确率
    acc = float(correct_sum) / num_test

    # 打印准确率.
    msg = "测试集准确率: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # 打印部分错误样例.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # 打印混淆矩阵.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = mnist.test.images[incorrect]

    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    indices = indices[:9]

    plot_images(images[indices], cls_true[indices], cls_pred[indices])


def plot_conv_weights(sess, weights, input_channel=0):
    # 运行weights以获得权重
    w = sess.run(weights)
    # 获取权重最小值最大值，这将用户纠正整个图像的颜色密集度，来进行对比
    w_min = np.min(w)
    w_max = np.max(w)
    # 卷积核数目
    num_filters = w.shape[3]
    # 需要输出的卷积核
    num_grids = 3  # math.ceil(num_feature)
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i < 9:
            # 获得第i个卷积核在特定输入通道上的权重
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_image(image):
    plt.imshow(image.reshape(image_2d_size),
               interpolation='nearest',
               cmap='binary')

    plt.show()


def plot_conv_layer(sess, layer, image):
    # layer_conv1 or layer_conv2.

    # feed_dict只需要x，标签信息在此不需要.
    feed_dict = {x: [image]}

    # 获取该层的输出结果
    values = sess.run(layer, feed_dict=feed_dict)

    # 卷积核树木
    num_filters = values.shape[3]

    # 每行需要输出的卷积核网格数
    num_grids = 3  # num_filters

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i < 9:
            # 获取第i个卷积核的输出
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print_test_accuracy(sess)

    optimize(sess, num_iter=10)
    print_test_accuracy(sess)
    optimize(sess, num_iter=900)
    print_test_accuracy(sess, show_example_errors=True)
    # 输出中间结果
    image1 = mnist.test.images[0]
    # 以上就是16个卷积核在第一个通道的权重情况。其中红色为正的权重，蓝色为负的权重。在这里我们很难判别这些权重是如何起作用的。
    # 将image1喂入卷积层1，得到使用不同卷积后得到的图像，这些图像的棱角更加分明，而且在不同的边的突出情况也不同：
    plot_conv_weights(sess, weights=layer_w1)
    # 将image2喂入卷积层1，得到如下图像，在不同部位的突出情况不同：
    plot_conv_layer(sess, layer=layer_1, image=image1)

    plot_conv_weights(sess, weights=layer_w2, input_channel=1)
    # 将image2喂入卷积层1，得到如下图像，在不同部位的突出情况不同：
    # 所输出的图像达到了一个更高的层次，卷积核试图提取一些边缘化的特征，这些特征对于同类图像的变化并不敏感。
    # 在运行完整个计算图后，需要将它关闭，否则将一直占用资源:
    plot_conv_layer(sess, layer=layer_2, image=image1)
