from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# 权重初始化
#
# 为了创建这个模型，我们需要创建大量的权重和偏置项。
# 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。
# 为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 我们的池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
# 卷积在每个5x5的patch中算出32个特征。
# 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
# 而对于每一个输出通道都有一个对应的偏置量。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，
# 如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 密集连接层
#
# 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout
#
# 为了减少过拟合，我们在输出层之前加入dropout。
# 我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
# 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
# 所以用dropout的时候可以不用考虑scale。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

mnist = input_data.read_data_sets("../data/", one_hot=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step %d,training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0, keep_prob: 0.5})

    print("test accuracy %g" % sess.run(accuracy,
                                        feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    sess.close()
