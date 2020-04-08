import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def build_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn

g_embed_hero_id = False
def _init_single_actor_net(scope, input_pl, num_layers, layer_size, trainable=True):        
    my_initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        flat_output_size = input_pl.get_shape()[1].value
        flat_output = tf.reshape(input_pl, [-1, flat_output_size], name='flat_output')

        # Add hero one-hot vector embedding
        if g_embed_hero_id:
            input_state = flat_output[:,:STATE_SIZE]
            input_hero_id = flat_output[:,STATE_SIZE:STATE_SIZE + 1]
            input_hero_one_hot = tf.one_hot(input_hero_id, ONE_HOT_SIZE)
            fc_W_embed = tf.get_variable(shape=[ONE_HOT_SIZE, EMBED_SIZE], name='fc_W_embed',
                trainable=trainable, initializer=my_initializer)

            output_embedding = tf.matmul(input_hero_one_hot, fc_W_embed)

            input_after_embed = tf.concat([input_state, output_embedding], -1)
            fc_W_1 = tf.get_variable(shape=[EMBED_SIZE + STATE_SIZE, layer_size], name='fc_W_1',
                trainable=trainable, initializer=my_initializer)

            fc_b_1 = tf.get_variable(shape=[layer_size], name='fc_b_1',
                trainable=trainable, initializer=my_initializer)                    

            tf.summary.histogram("fc_W_1", fc_W_1)
            tf.summary.histogram("fc_b_1", fc_b_1)

            output1 = tf.nn.relu(tf.matmul(input_after_embed, fc_W_1) + fc_b_1)
        else:
            fc_W_1 = tf.get_variable(shape=[flat_output_size, layer_size], name='fc_W_1',
                trainable=trainable, initializer=my_initializer)

            fc_b_1 = tf.get_variable(shape=[layer_size], name='fc_b_1',
                trainable=trainable, initializer=my_initializer)

            tf.summary.histogram("fc_W_1", fc_W_1)
            tf.summary.histogram("fc_b_1", fc_b_1)

            output1 = tf.nn.relu(tf.matmul(flat_output, fc_W_1) + fc_b_1)                

        fc_W_2 = tf.get_variable(shape=[layer_size, layer_size], name='fc_W_2',
            trainable=trainable, initializer=my_initializer)

        fc_b_2 = tf.get_variable(shape=[layer_size], name='fc_b_2',
            trainable=trainable, initializer=my_initializer)

        tf.summary.histogram("fc_W_2", fc_W_2)
        tf.summary.histogram("fc_b_2", fc_b_2)

        output2 = tf.nn.relu(tf.matmul(output1, fc_W_2) + fc_b_2)


        fc_W_3 = tf.get_variable(shape=[layer_size, layer_size], name='fc_W_3',
            trainable=trainable, initializer=my_initializer)

        fc_b_3 = tf.get_variable(shape=[layer_size], name='fc_b_3',
            trainable=trainable, initializer=my_initializer)

        tf.summary.histogram("fc_W_3", fc_W_3)
        tf.summary.histogram("fc_b_3", fc_b_3)

        output3 = tf.nn.relu(tf.matmul(output2, fc_W_3) + fc_b_3)
        # return hidden state
        return output3


@register("multi_unit_mlp")
def multi_unit_mlp(num_layers=2, num_hidden=64):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        unit_count = X.get_shape()[1].value

        hidden_state_arr = []

        for actor_idx in range(unit_count):
            single_unit_input = X[:,actor_idx, ...]
            # Share weights
            hidden_state = _init_single_actor_net('single_unit_backbone', single_unit_input, num_layers, num_hidden, trainable = True)

            hidden_state_arr.append(hidden_state)

        concated_tensor = tf.concat(hidden_state_arr, -1)
        return concated_tensor

    return network_fn    

@register("multi_unit_mlp_lstm")
def multi_unit_mlp_lstm(num_layers=2, num_hidden=64):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        unit_count = X.get_shape()[1].value
        h = tf.layers.flatten(X)
        for i in range(unit_count):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn 

@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs)
    return network_fn

@register("impala_cnn")
def impala_cnn(**conv_kwargs):
    def network_fn(X):
        return build_impala_cnn(X)
    return network_fn

@register("cnn_small")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h
    return network_fn

@register("lstm")
def lstm(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


@register("cnn_lstm")
def cnn_lstm(nlstm=128, layer_norm=False, conv_fn=nature_cnn, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = conv_fn(X, **conv_kwargs)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

@register("impala_cnn_lstm")
def impala_cnn_lstm():
    return cnn_lstm(nlstm=256, conv_fn=build_impala_cnn)

@register("cnn_lnlstm")
def cnn_lnlstm(nlstm=128, **conv_kwargs):
    return cnn_lstm(nlstm, layer_norm=True, **conv_kwargs)


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer

    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = tf.contrib.layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           **conv_kwargs)

        return out
    return network_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
