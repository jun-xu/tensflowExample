from __future__ import print_function

import tensorflow as tf


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        # 计算参数的均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算参数的标准差
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        # 使用tf.summary.scaler记录标准差,最大值,最小值
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        # 用直方图记录参数分布
        tf.summary.histogram(name, var)
