# Visual sanity tests to ensure the Spatial transformer pooling work
# as expected. Code is absed on tensorflow ST contrlib by chen.wu@icrar.org
# original copyright below
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer, batch_transformer
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %% Create a batch of three images (1600 x 1200)
# %% Image retrieved from:
# %% https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('cat.jpg')
im = im / 255.
im = im.reshape(1, 1200, 1600, 3)
im = im.astype('float32')

# %% Let the output size of the transformer be half the image size.
out_size = (90, 90) # (w, h)

# %% Simulate batch
#batch = np.append(im, im, axis=0)
#batch = np.append(batch, im, axis=0)
batch = im
num_batch = 1

conv5_3 = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
conv5_3 = tf.cast(batch, 'float32')
print("x shape = ", conv5_3.get_shape())

# %% Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):
    # python native implementation
    x1 = 300.0
    y1 = 300.0
    x2 = 600.0
    y2 = 600.0
    # xc = (x1 + x2) / 2
    # yc = (y1 + y2) / 2
    # w = x2 - x1
    # h = y2 - y1
    W = 1600.0
    H = 1200.0
    #
    # initial = np.array([[h / H, 0, (2 * yc - H - 1) / (H - 1)],
    #                     [0, w / W, (2 * xc - W - 1) / (W - 1)]])
    # initial = initial.astype('float32')
    # initial = initial.flatten()

    angle = tf.random_uniform([3, 1], maxval=np.pi)
    #angle = tf.cast(np.zeros([3, 1]), 'float32')
    s_ang = tf.sin(angle)
    c_ang = tf.cos(angle)

    proposal_init = np.zeros([3, 5])
    proposal_init[0, :] = [0, x1, y1, x2, y2]
    proposal_init[1, :] = [0, x1 + 100, y1 + 100, x2 + 100, y2 + 100]
    proposal_init[2, :] = [0, x1 - 100, y1 - 100, x2 - 100, y2 - 100]
    proposals = tf.convert_to_tensor(proposal_init, dtype=tf.float32)
    num_prop = 3

    x1v = tf.slice(proposals, [0, 1], [num_prop, 1])
    x2v = tf.slice(proposals, [0, 3], [num_prop, 1])
    y1v = tf.slice(proposals, [0, 2], [num_prop, 1])
    y2v = tf.slice(proposals, [0, 4], [num_prop, 1])

    xc = tf.divide(tf.add(x1v, x2v), 2.0)
    yc = tf.divide(tf.add(y1v, y2v), 2.0)
    w = tf.subtract(x2v, x1v)
    h = tf.subtract(y2v, y1v)
    print("w.shape = {0}".format(w.get_shape().as_list()))

    h_translate_p = tf.subtract(tf.subtract(tf.multiply(2.0, yc), H), 1.0)
    h_translate = tf.divide(h_translate_p, tf.subtract(H, 1.0))
    #row1 = tf.concat([tf.divide(h, H), np.zeros([num_prop, 1]), h_translate], axis=1)
    #row2 = tf.concat([np.zeros([num_prop, 1]), tf.divide(h, H), h_translate], axis=1)
    #row2 = tf.concat([s_ang * tf.divide(w, W), tf.divide(h, H) * c_ang, h_translate], axis=1)
    row2 = tf.concat([tf.multiply(s_ang, tf.divide(w, W)), tf.divide(h, H) * c_ang, h_translate], axis=1)

    w_translate_p = tf.subtract(tf.subtract(tf.multiply(2.0, xc), W), 1.0)
    w_translate = tf.divide(w_translate_p, tf.subtract(W, 1.0))
    #row2 = tf.concat([np.zeros([num_prop, 1]), tf.divide(w, W), w_translate], axis=1)
    #row1 = tf.concat([tf.divide(w, W), np.zeros([num_prop, 1]), w_translate], axis=1)
    #row1 = tf.concat([tf.divide(w, W) * c_ang, -1 * s_ang * tf.divide(w, W), w_translate], axis=1)
    row1 = tf.concat([tf.multiply(tf.divide(w, W), c_ang), -1 * s_ang * tf.divide(w, W), w_translate], axis=1)

    thetas = tf.stack([row1, row2], axis=1)
    thetas = tf.reshape(thetas, [1, num_prop, 6])

    final_output = batch_transformer(conv5_3, thetas, (out_size[1], out_size[0]))

# %% Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y, theta_val, ag = sess.run([final_output, thetas, angle], feed_dict={conv5_3: batch})
print("random angle = {0}".format(ag))
#print(y.shape, theta_val)

origin_im = cv2.imread('cat.jpg')

for j in range(num_prop):
    plt.imshow(y[j])
    plt.savefig('test%d_pred.png' % j)
    plt.close()

    x1, y1, x2, y2 = [int(x) for x in proposal_init[j, :][1:]]
    cutout = origin_im[x1:x2, y1:y2, :]
    to_dump = cv2.resize(cutout, out_size)
    cv2.imwrite('test%d_truth.png' % j, to_dump)
