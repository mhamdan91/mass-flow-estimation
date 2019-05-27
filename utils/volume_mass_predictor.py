from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()


def model_pos(vol):
    beta = 2.759320676099754
    vol_b = tf.nn.relu(vol - beta)
    w1 = tf.constant([1.51248678, 0.05577656, 1.13650198])
    b1 = tf.constant([0.69135934, 0.22673389, 0.05709011])
    w2 = [[-0.14444944], [-1.25086896], [-0.19815545]]
    b2 = tf.constant([-1.24339893])
    L1 = tf.add(tf.math.scalar_mul(vol_b, w1), b1)
    L1_1 = tf.nn.tanh(L1)
    L1_1 = tf.reshape(L1_1, [1, 3])
    L2 = tf.add(tf.matmul(L1_1, w2), b2)
    L21 = tf.exp(L2)
    L2_1 = L21 * vol_b
    return L2_1

