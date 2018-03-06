# Test how scatter_multiply works
import tensorflow as tf
import numpy as np

def test_zerofy():
    M = 3 # batch size
    C = 2 # number of classes
    N = 4 # four coordinates per subject

    input_score = np.random.random([M, C * N])
    print(input_score)
    one_hot = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

    one_hot_tensor = tf.placeholder(tf.float32, [M, C])
    input_score_tensor = tf.placeholder(tf.float32, [M, C * N])
    A2 = tf.reshape(one_hot_tensor, [C, M, 1])
    A2_tile = tf.tile(A2, [1, 1, N])
    A2_final = tf.reshape(A2_tile, [M, C * N])

    output_score = tf.multiply(input_score_tensor, A2_final)

    with tf.Session() as sess:
        print(sess.run([output_score], feed_dict={one_hot_tensor: one_hot, input_score_tensor: input_score}))

def test_one_hot():
    A1 = tf.constant([3, 4, 6, 1, 0, 3])
    A2 = tf.one_hot(A1, 7)
    with tf.Session() as sess:
        print(sess.run([A2]))
