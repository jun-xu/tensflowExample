import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix

import time
from datetime import timedelta
import math

filter_size1 = 5  # 5 * 5 卷积
num_filters1 = 16

filter_size2 = 5
num_filters2 = 32

# fully-connected layer
fc_size = 1024

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("../data/", one_hot=True)

print("数据集大小：")
print('- 训练集：{}'.format(len(data.train.labels)))
print('- 测试集：{}'.format(len(data.test.labels)))
print('- 验证集：{}'.format(len(data.validation.labels)))

print(data.test.labels[:5])
data.test.cls = np.argmax(data.test.labels, axis=1)
print(data.test.cls[:5])
print("样本维度：", data.train.images.shape)
print("标签维度：", data.train.labels.shape)

img_size = 28  # 图片的高度和宽度
img_size_flat = img_size * img_size  # 展平为向量的尺寸
img_shape = (img_size, img_size)  # 图片的二维尺寸

num_channels = 1  # 输入为单通道灰度图像
num_classes = 10  # 类别数目


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
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:  # 如果未传入预测类别
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)

        # 删除坐标信息
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


indices = np.arange(len(data.test.cls))
np.random.shuffle(indices)
indices = indices[:9]

images = data.test.images[indices]
cls_true = data.test.cls[indices]

plot_images(images, cls_true)


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.5))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


### 创建卷积层
def new_conv_layer(input,  # 前一层.
                   num_input_channels,  # 前一层通道数
                   filter_size,  # 卷积核尺寸
                   num_filters,  # 卷积核数目
                   use_pooling=True):  # 使用 2x2 max-pooling.
    # 卷积核权重的形状，由TensorFlow API决定
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    # 根据跟定形状创建权重
    weights = new_weights(shape)
    # 创建新的偏置，每个卷积核一个偏置
    biases = new_biases(num_filters)

    # 创建卷积层。注意stride全设置为1。
    # 第1个和第4个必须是1，因为第1个是图像的数目，第4个是图像的通道。
    # 第2和第3指定和左右、上下的步长。
    # padding设置为'SAME' 意味着给图像补零，以保证前后像素相同。

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        # 这是 2x2 max-pooling, 表明使用 2x2 的窗口，选择每一窗口的最大值作为该窗口的像素，
        # 然后移动2格到下一窗口。
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # 对每个输入像素x，计算 max(x, 0)，把负数的像素值变为0.
    # 这一步为原输出添加了一定的非线性特性，允许我们学习更加复杂的函数。
    layer = tf.nn.relu(layer)
    # 注意 relu 通常在pooling前执行，但是由于 relu(max_pool(x)) == max_pool(relu(x))，
    # 我们可以通过先max_pooling再relu省去75%的计算。

    # 返回结果层和权重，结果层用于下一层输入，权重用于显式输出
    return layer, weights


def flatten_layer(layer):
    # 获取输入层的形状，
    # layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()

    # 特征数量: img_height * img_width * num_channels
    # 可以使用TensorFlow内建操作计算.
    num_features = layer_shape[1:4].num_elements()

    # 将形状重塑为 [num_images, num_features].
    # 注意只设定了第二个维度的尺寸为num_filters，第一个维度为-1，保证第一个维度num_images不变
    # 展平后的层的形状为:
    # [num_images, img_height * img_width * num_channels]
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


### 创建全连接层

def new_dense_layer(input,  # 前一层.
                    num_inputs,  # 前一层输入维度
                    num_outputs,  # 输出维度
                    use_relu=True):  # 是否使用RELU

    # 新的权重和偏置，与第一章一样.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # 计算 y = wx + b，同第一章
    layer = tf.matmul(input, weights) + biases

    # 是否使用RELU
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


### 参数
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, 1])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weight_conv1 = new_conv_layer(input=x_image,
                                           num_input_channels=1,
                                           filter_size=filter_size1,
                                           num_filters=num_filters1,
                                           use_pooling=True)

print('layer_conv1:', layer_conv1)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

print('layer_conv2:', layer_conv2)

layer_flat, num_features = flatten_layer(layer_conv2)
print('layer_flat:', layer_flat)

# 全连接层 1

layer_dense1 = new_dense_layer(input=layer_flat,  # 展平层输出
                               num_inputs=num_features,  # 输入特征维度
                               num_outputs=fc_size,  # 输出特征维度
                               use_relu=True)

print(layer_dense1)

# 注意该层未使用relu，因为将要输入到后续的softmax中

layer_dense2 = new_dense_layer(input=layer_dense1,
                               num_inputs=fc_size,
                               num_outputs=num_classes,
                               use_relu=False)
print(layer_dense2)

# 预测类别
#
# 第二个全连接层估计输入的图像属于某一类别的程度，这个估计有些粗糙，需要添加一个softmax层归一化为概率表示。
#

y_pred = tf.nn.softmax(layer_dense2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_dense2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# 优化方法
#
# 这一部分与上一章类似，但是优化器使用改进版的梯度下降，Adam。

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# 性能度量

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 运行TensorFlow计算图
#
# 创建Session以及变量初始化
#
# TensorFlow计算图运行在一个session中，计算之前需要先创建这个session，并初始化其中的一些变量（w 和 b），TensorFlow使用session.run()来运行计算图。

train_batch_size = 64
total_iteration = 0


def optimize(num_iter):
    global total_iteration
    start_time = time.time()
    for i in range(total_iteration, total_iteration + num_iter):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            msg = "迭代轮次: {0:>6}, 训练准确率: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    total_iteration += num_iter
    end_time = time.time()
    time_dif = end_time - start_time
    print("用时: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # 计算错误情况
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    # 随机挑选9个
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    indices = indices[:9]

    plot_images(images[indices], cls_true[indices], cls_pred[indices])


def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls  # 真实类别

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


# 将测试集分成更小的批次
test_batch_size = 256


def print_test_accuracy(sess, show_example_errors=False,
                        show_confusion_matrix=False):
    # 测试集图像数量.
    num_test = len(data.test.images)

    # 为预测结果申请一个数组.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # 数据集的起始id为0
    i = 0
    while i < num_test:
        # j为下一批次的截止id
        j = min(i + test_batch_size, num_test)

        # 获取i，j之间的图像
        images = data.test.images[i:j, :]

        # 获取相应标签.
        labels = data.test.labels[i:j, :]

        # 创建feed_dict
        feed_dict = {x: images, y_true: labels}

        # 计算预测结果
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        # 设定为下一批次起始值.
        i = j

    cls_true = data.test.cls
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


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print_test_accuracy(sess)
    optimize(num_iter=1)
    print_test_accuracy(sess)

    optimize(num_iter=99)
    print_test_accuracy(sess)

    optimize(num_iter=900)
    print_test_accuracy(sess, show_example_errors=True)

    optimize(num_iter=9000)
    print_test_accuracy(sess, show_example_errors=True,
                        show_confusion_matrix=True)
