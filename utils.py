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

def lstm(xs, is_inference, hidden_s, mask_s, scope, nh, cell_count, my_initializer, init_scale=1.0):
    nbatch, nin = [v.value for v in xs.get_shape()]
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        wx = tf.get_variable("lstm_wx", [nin, nh*4], initializer=my_initializer)
        wh = tf.get_variable("lstm_wh", [nh, nh*4], initializer=my_initializer)
        b = tf.get_variable("lstm_b", [nh*4], initializer=tf.constant_initializer(0.0))    
    

    c_old, h_old = tf.split(axis=1, num_or_size_splits=2, value=hidden_s)
    
    def f1(): 
        c, h = None, None
        output_seq = []
        time_steps = 1
        steps_cell_count = 1        
        for idy in range(steps_cell_count):        
            c, h = tf.expand_dims(c_old[idy * time_steps], axis = 0), tf.expand_dims(h_old[idy * time_steps], 0)        
            for idx in range(time_steps):
                single_input = tf.expand_dims(xs[idy * time_steps + idx], 0)
                single_mask = tf.expand_dims(mask_s[idy * time_steps + idx], 0)
                c = c * tf.cast((1 - single_mask), tf.float32)
                h = h * tf.cast((1 - single_mask), tf.float32)
                z = tf.matmul(single_input, wx) + tf.matmul(h, wh) + b
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i)
                f = tf.nn.sigmoid(f)
                o = tf.nn.sigmoid(o)
                u = tf.tanh(u)
                c = f*c + i*u
                h = o*tf.tanh(c)   
                output_seq.append(h)     
        lstm_output = tf.concat(axis=0, values=output_seq)        
        s = tf.concat(axis=1, values=[c, h])
        return lstm_output, s

    def f2(): 
        c, h = None, None
        output_seq = []
        batch_count = 128
        time_steps = 8
        steps_cell_count = int(batch_count / time_steps)        
        for idy in range(steps_cell_count):        
            c, h = tf.expand_dims(c_old[idy * time_steps], axis = 0), tf.expand_dims(h_old[idy * time_steps], 0)        
            for idx in range(time_steps):
                single_input = tf.expand_dims(xs[idy * time_steps + idx], 0)
                single_mask = tf.expand_dims(mask_s[idy * time_steps + idx], 0)
                c = c * tf.cast((1 - single_mask), tf.float32)
                h = h * tf.cast((1 - single_mask), tf.float32)
                z = tf.matmul(single_input, wx) + tf.matmul(h, wh) + b
                i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
                i = tf.nn.sigmoid(i)
                f = tf.nn.sigmoid(f)
                o = tf.nn.sigmoid(o)
                u = tf.tanh(u)
                c = f*c + i*u
                h = o*tf.tanh(c)   
                output_seq.append(h)     
        lstm_output = tf.concat(axis=0, values=output_seq)        
        s = tf.concat(axis=1, values=[c, h])
        return lstm_output, s

    lstm_output, s = tf.cond(tf.equal(is_inference, True), f1, f2)
    return lstm_output, s