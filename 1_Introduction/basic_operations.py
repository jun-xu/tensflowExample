from __future__ import print_function

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2, b=2")
    print("增加a+b=%i" % sess.run(a + b))
    print("乘法a*b=%i" % sess.run(a * b))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("加法: 3+4=%i" % sess.run(add, feed_dict={a: 3, b: 4}))
    print("3*4=%i" % sess.run(mul, feed_dict={a: 3, b: 4}))

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print(sess.run(product))


