import os
import numpy as np
import tensorflow as tf
from collections import deque

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
    
def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])

def lstm(xs, hidden_s, scope, nh, cell_count, my_initializer, init_scale=1.0):
    nbatch, nin = [v.value for v in xs.get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("lstm_wx", [nin, nh*4], initializer=my_initializer)
        wh = tf.get_variable("lstm_wh", [nh, nh*4], initializer=my_initializer)
        b = tf.get_variable("lstm_b", [nh*4], initializer=tf.constant_initializer(0.0))
        lstm_output_w = tf.get_variable("lstm_output_w", [nh, nin], initializer=my_initializer)
        lstm_output_b = tf.get_variable("lstm_output_b", [nin], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=hidden_s)    
    for idy in range(cell_count):
        z = tf.matmul(xs, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)        
    lstm_output = h
    s = tf.concat(axis=1, values=[c, h])
    return lstm_output, s