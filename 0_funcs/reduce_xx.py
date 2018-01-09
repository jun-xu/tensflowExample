from __future__ import print_function

import tensorflow as tf

x = tf.constant([[1., 1.], [2., 4.]])
with tf.Session() as sess:
    print(tf.reduce_mean(x), sess.run(tf.reduce_mean(x)))  # 1.5
    print(tf.reduce_mean(x, 0, keep_dims=True), sess.run(tf.reduce_mean(x, 0, keep_dims=True)),
          sess.run(tf.reduce_mean(x, 0)))  # [1.5, 1.5]
    print(tf.reduce_mean(x, 1, keep_dims=True), sess.run(tf.reduce_mean(x, 1, keep_dims=True)),
          sess.run(tf.reduce_mean(x, 1)))  # [1.,  2.]
