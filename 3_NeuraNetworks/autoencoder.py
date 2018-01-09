from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils.tensor_board_util as boardUtil
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/", one_hot=True)

learning_rate = 0.01
num_steps = 1000
batch_size = 256

display_step = 1000
examples_to_show = 10

num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784

X = tf.placeholder(tf.float32, [None, num_input],name="input_x")

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1]),name="encoder_h1"),  # 784 x 256
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]),name="encoder_h2"),  # 256 x 128
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1]),name="decoder_h1"),  # 128 x 256
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input]),name="decoder_h2"),  # 256 x 784
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1]),name="encoder_b1"),  # 256
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2]),name="encoder_b2"),  # 128
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1]),name="decoder_b1"),  # 256
    'decoder_b2': tf.Variable(tf.random_normal([num_input]),name="decoder_b2"),  # 784
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   # l1 = sigmod(x*W1+b)   [None,784] x [784 x 256] + [256] = [None,256]+ [256]
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  # l2 = sigmod(l1*W2+b)
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  # l1 = e * W2(T) + b
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  # l2 = l1 * W1(T) + b
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
log_dir = "../tensor-boards/autoEncoder"
# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    train_writer.close()
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()