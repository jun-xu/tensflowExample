import numpy as np

#
# a = np.zeros([1, 3, 3, 1], dtype=np.int64)
#
# print(a)

# a = np.zeros([1, 16], dtype=np.int64)
# print(a)
#
# a = a.reshape([-1, 4, 4, 1])
# print(a)

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("../data/", one_hot=True)
img_size = 28  # 图片的高度和宽度
img_size_flat = img_size * img_size  # 展平为向量的尺寸
x = tf.placeholder(tf.float32, shape=[None, img_size_flat])
y = tf.reshape(x, [-1, img_size, img_size, 10])
print(x.get_shape())
print(y.get_shape())
# a = np.array(data.train.images[0])
# print(np.shape(a),np.shape(np.zeros([1])))

a = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
print(np.shape(a), a)
a_ = np.reshape(a, [-1, 2, 2, 4])
print(np.shape(a_))
print(a_)

a = np.array(data.train.images)
print(np.shape(a))
a_ = np.reshape(a, [-1, img_size, img_size, 10])

print(np.shape(a_))

# with tf.Session() as sess:
#     y_ = sess.run(y, feed_dict={x: data.train.images})
