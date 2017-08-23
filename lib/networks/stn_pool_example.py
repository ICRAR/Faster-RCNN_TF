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
from spatial_transformer import transformer
import numpy as np
import matplotlib.pyplot as plt

# %% Create a batch of three images (1600 x 1200)
# %% Image retrieved from:
# %% https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('cat.jpg')
im = im / 255.
im = im.reshape(1, 1200, 1600, 3)
im = im.astype('float32')

# %% Let the output size of the transformer be half the image size.
out_size = (95, 125)

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

    proposal_init = np.zeros([3,5])
    proposal_init[0, :] = [0, x1, y1, x2, y2]
    proposal_init[1, :] = [0, x1, y1, x2, y2]
    proposal_init[2, :] = [0, x1, y1, x2, y2]
    proposals = tf.convert_to_tensor(proposal_init, dtype=tf.float32)

    num_prop = 3
    output_list = []
    tensor_two = tf.convert_to_tensor(2.0, dtype=tf.float32)
    tensor_one = tf.convert_to_tensor(1.0, dtype=tf.float32)
    tensor_zero = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
    scale_tensor = tf.convert_to_tensor([[1.0]], dtype=tf.float32)
    for i in range(num_prop):
        proposal = tf.reshape(tf.slice(proposals, [i, 1], [1, 4]), [4])
        #print("A proposal's shape = {0}".format(proposal.get_shape().as_list()))
        x1 = tf.slice(proposal, [0], [1])
        y1 = tf.slice(proposal, [1], [1])
        x2 = tf.slice(proposal, [2], [1])
        y2 = tf.slice(proposal, [3], [1])
        #print(x1, y1, x2, y2)
        xc = tf.divide(tf.add(x1, x2), tensor_two)
        yc = tf.divide(tf.add(y1, y2), tensor_two)
        w = tf.subtract(x2, x1)
        h = tf.subtract(y2, y1)
        h_translate_p = tf.subtract(tf.subtract(tf.multiply(tensor_two, yc), H), tensor_one)
        h_translate = tf.divide(h_translate_p, tf.subtract(H, tensor_one))
        row1_p = tf.concat([tf.multiply(scale_tensor, tf.divide(h, H)),
                            tensor_zero,
                            tf.multiply(scale_tensor, h_translate)], axis=1)
        row1 = tf.reshape(row1_p, [3])

        w_translate_p = tf.subtract(tf.subtract(tf.multiply(tensor_two, xc), W), tensor_one)
        w_translate = tf.divide(w_translate_p, tf.subtract(W, tensor_one))
        row2_p = tf.concat([tensor_zero,
                            tf.multiply(scale_tensor, tf.divide(w, W)),
                            tf.multiply(scale_tensor, w_translate)], axis=1)
        row2 = tf.reshape(row2_p, [3])

        #print("row2 shape = {0}".format(row2.get_shape().as_list()))
        theta = tf.stack([row1, row2], axis=0)
        theta_shape = theta.get_shape().as_list()
        #print("theta shape = {0}".format(theta_shape))
        #assert(theta_shape[0] == 2 and theta_shape[1] == 3)

        h_trans = transformer(conv5_3, tf.reshape(theta, [1, 6]), out_size)
        #h_trans = transformer(conv5_3, theta, out_size)
        output_list.append(h_trans)

    final_output = tf.concat(output_list, axis=0)

# %% Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y, theta_val = sess.run([final_output, theta], feed_dict={conv5_3: batch})
print(y.shape, theta_val)

plt.imshow(y[0])
plt.savefig('test.png')
