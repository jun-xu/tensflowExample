""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

"""

from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='w1'),  # 784 x 256
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='w2'),  # 256 x 256
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='w_out')  # 256 x 10
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
}


def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
log_dir = "../tensor-boards/multilayer_perceptron"

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("epoch:", '%04d' % (epoch), " cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
