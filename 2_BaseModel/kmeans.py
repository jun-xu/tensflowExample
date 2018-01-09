from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.factorization import KMeans

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/", one_hot=True)

full_data_x = mnist.train.images

num_step = 50
batch_size = 1024
k = 25
num_classes = 10
num_features = 28 * 28

# 输入图片
X = tf.placeholder(tf.float32, shape=[None, num_features], name="X")
# 对比值
Y = tf.placeholder(tf.float32, shape=[None, num_classes], name="Y")
# 采用"余弦相似度"
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

# build KMeans graph
# all_scores:   distance of each input to each cluster center.
# score:        distance of each input to closest cluster center.
# cluster_idx:  index of cluster center closest to the corresponding input.
(all_scores, cluster_idx, scores, cluster_centers_initialized,
 cluster_centers_var, init_op, training_op) = kmeans.training_graph()

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

log_dir = "../tensor-boards/kmean"

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

sess.run(init_vars)
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_step + 1):
    _, d, idx = sess.run([training_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print('step %i, avg distance: %f' % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]

# Assign the most frequent label to the centroid
labels_list = [np.argmax(c) for c in counts]
labels_list = tf.convert_to_tensor(labels_list)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_list, cluster_idx)

# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
accuracy_tensor = sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y})
print("test accuracy:", accuracy_tensor)

train_writer.close()
sess.close()
