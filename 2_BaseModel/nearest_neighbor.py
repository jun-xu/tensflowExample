from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/', one_hot=True)
#使用训练集数目为5000条
#使用验证集（测试集）数目为200
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.train.next_batch(200)

xtr = tf.placeholder(tf.float32, [None, 784])
xte = tf.placeholder(tf.float32, [784])
#计算各个对应位置的距离（减法使用广播形式）
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
#寻找距离最近（即最相似的行所在位置）
pred = tf.argmin(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        # 计算最相近的所在行位置
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("测试", i, "prediction:", np.argmax(Ytr[nn_index]), " true class:", np.argmax(Yte[i]))
        # 取出测试集上最相近行对应的label与真是label对比
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)

    print("donw!, accuracy:", accuracy)
