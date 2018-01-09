from __future__ import print_function

import tensorflow as tf
import utils.tensor_board_util  as boardUtil
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/", one_hot=True)

# tensor-board
# def variable_summaries(var, name):
#     with tf.name_scope('summaries'):
#         # 计算参数的均值
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean/' + name, mean)
#         # 计算参数的标准差
#         with tf.name_scope("stddev"):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#
#         # 使用tf.summary.scaler记录标准差,最大值,最小值
#         tf.summary.scalar('stddev/' + name, stddev)
#         tf.summary.scalar('max/' + name, tf.reduce_max(var))
#         tf.summary.scalar('min/' + name, tf.reduce_min(var))
#         # 用直方图记录参数分布
#         tf.summary.histogram(name, var)


# 图片点阵
x = tf.placeholder('float', [None, 784], name="x")
W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")
# 实际概率分布(q)
q = tf.nn.softmax(tf.matmul(x, W) + b, name="q")

# 真实的概率分布(p), 从t10k-labels中读取
p = tf.placeholder("float", [None, 10], name="p")

# 交叉熵 求和 (p&log(q))
cross_entropy = -tf.reduce_sum(p * tf.log(q), name="cross_entropy")
learning_rate = 0.01
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
boardUtil.variable_summaries(cross_entropy, "loss")

# 评估模型
correct_prediction = tf.equal(tf.argmax(q, 1), tf.argmax(p, 1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
boardUtil.variable_summaries(accuracy, "accuracy")

# 运行
log_dir = "../tensor-boards/mnist-base"

epoch = 20
max_step = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

    for j in range(epoch):
        for i in range(max_step):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            acc = sess.run([train_step], feed_dict={x: batch_xs, p: batch_ys})
            # train_writer.add_summary(summary, i)

        summary, accuracy_tensor, rets = sess.run([merged, accuracy, tf.argmax(q, 1)],
                                                  feed_dict={x: mnist.test.images, p: mnist.test.labels})
        train_writer.add_summary(summary, j)
    print(accuracy_tensor)
