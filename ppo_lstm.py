# coding=utf-8
"""
Some of code copy from
https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
"""

import sys
import copy
import numpy as np

from collections import deque
import gym, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random, cv2
import time
import math
import utils

import tensorflow.contrib.eager as tfe 
conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads = 1)
#tfe.enable_eager_execution(conf)


EPSILON = 0.2


C = 1

NUM_FRAME_PER_ACTION = 4
BATCH_SIZE = 128
TIME_STEP = 8

EPOCH_NUM = 4
LEARNING_RATE = 1e-4
TIMESTEPS_PER_ACTOR_BATCH = 2048
GAMMA = 0.99
LAMBDA = 0.95
NUM_STEPS = 5000
ENV_NAME = 'Breakout-v0'
RANDOM_START_STEPS = 10
ACTION_COUNT = 4
F = (84, 84)
INPUT_DIMENS = (*F, C)
INPUT_DIMENS_FLAT = INPUT_DIMENS[0]*INPUT_DIMENS[1]*INPUT_DIMENS[2]
USE_CNN = True
HIDDEN_STATE_LEN = 64
LSTM_CELL_COUNT = 1

global g_step

# Save model in pb format
g_save_pb_model = False

# Control if output to tensorboard
g_out_tb = False

# Control if display game scene
g_display_scene = False

# Control if train or play
g_is_train = True
g_manual_ctrl_enemy = False
# True means start a new train task without loading previous model.
g_start_anew = True

# Control if use priority sampling
g_enable_per = False
g_per_alpha = 0.6
g_is_beta_start = 0.4
g_is_beta_end = 1

def inflate_random_indice(indices, inflate_ratio):
    output_indices = []
    for idx in indices:
        for idy in range(inflate_ratio):
            output_indices.append(idx * inflate_ratio + idy)
        
    return output_indices

def stable_softmax(logits, name): 
    a = logits - tf.reduce_max(logits, axis=-1, keepdims=True) 
    ea = tf.exp(a) 
    z = tf.reduce_sum(ea, axis=-1, keepdims=True) 
    return tf.div(ea, z, name = name) 

class Environment(object):
  def __init__(self):
    global ACTION_COUNT
    self.env = gym.make(ENV_NAME)
    self._screen = None
    self.reward = 0
    self.terminal = True
    self.random_start = RANDOM_START_STEPS
    self.obs_w = self.env.observation_space.shape[0]
    self.obs_h = self.env.observation_space.shape[1]
    ACTION_COUNT = self.env.action_space.n
    self.obs = np.zeros(shape=(*F, C), dtype=np.float)
#  @staticmethod
  def get_action_number():
    return 4

  def step(self, action):
    self._screen, self.reward, self.terminal, info = self.env.step(action)
    self._screen = cv2.resize(self._screen, F)
    self._screen = np.sum(self._screen, axis = 2) / 3.0 / 255.0
    self.obs[..., :-1] = self.obs[..., 1:]
    self.obs[..., -1] = self._screen
    return self.obs, self.reward, self.terminal, info

  def reset(self):
    self._screen = self.env.reset()
    for _ in range(random.randint(3, self.random_start - 1)):
      step_idx = np.random.choice(range(Environment.get_action_number()))
      ob, _1, _2, _3 = self.step(step_idx)
    return self.obs

  def render(self):
    self.env.render()

class Data_Generator():
    def __init__(self, agent):
        self.env = Environment()
 
        self.agent = agent
        self.timesteps_per_actor_batch = TIMESTEPS_PER_ACTOR_BATCH
        self.seg_gen = self.traj_segment_generator(horizon=self.timesteps_per_actor_batch)
    
    def traj_segment_generator(self, horizon=256):
        '''
        horizon: int timesteps_per_actor_batch
        '''
        t = 0
        batch_actions = [0]

        new = True # marks if we're on first timestep of an episode
        ob = self.env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_unclipped_ret = 0 # unclipped return in current episode
        cur_ep_len = 0 # len of current episode
        ep_rets = [] # returns of completed episodes in this segment
        ep_unclipped_rets = [] # unclipped returns of completed episodes in this segment
        ep_lens = [] # lengths of ...

        # Initialize history arrays
        obs = np.array([np.zeros(shape=(*F, C), dtype=np.float32) for _ in range(horizon)])
        hidden_states = np.array([np.zeros(shape=(HIDDEN_STATE_LEN * 2), dtype=np.float32) for _ in range(horizon)])

        rews = np.zeros(horizon, 'float32')
        unclipped_rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([batch_actions for _ in range(horizon)])
        prevacs = acs.copy()

        hidden_state_before = np.zeros(shape=(HIDDEN_STATE_LEN * 2), dtype=np.float32)
        while True:            
            hidden_state = hidden_state_before
            prevac = batch_actions
            batch_actions, vpred, hidden_state = self.agent.predict(ob[np.newaxis, ...], hidden_state_before[np.newaxis, ...], [0])
            ac = batch_actions[0]
            #print('Action:', ac, 'Value:', vpred)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                        "ep_unclipped_rets": ep_unclipped_rets, "hidden_states": hidden_states}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_unclipped_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = batch_actions
            prevacs[i] = prevac
            hidden_states[i] = hidden_state_before

            ob, unclipped_rew, new, step_info = self.env.step(ac)
            rew = unclipped_rew
            # rew = float(np.sign(unclipped_rew))
            rews[i] = rew
            unclipped_rews[i] = unclipped_rew

            cur_ep_ret += rew
            cur_ep_unclipped_ret += unclipped_rew
            cur_ep_len += 1
            if new:
                hidden_state_before = np.zeros(shape=(HIDDEN_STATE_LEN * 2), dtype=np.float32)

                if False:#cur_ep_unclipped_ret == 0:
                    pass
                else:
                    ep_rets.append(cur_ep_ret)
                    ep_unclipped_rets.append(cur_ep_unclipped_ret)
                    ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_unclipped_ret = 0
                cur_ep_len = 0
                ob = self.env.reset()
            t += 1
            if g_display_scene:
                self.env.render()
    
    def add_vtarg_and_adv(self, seg, gamma=0.99, lam=0.95):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def get_one_step_data(self):
        seg = self.seg_gen.__next__()
        self.add_vtarg_and_adv(seg, gamma=GAMMA, lam=LAMBDA)
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        if atarg.std() < 1e-5:
            print('atarg std too small')
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        seg['std_atvtg'] = atarg
        return seg

class Agent():

    def __init__(self, session, a_space, a_space_keys, **options):       
        self.importance_sample_arr = np.ones([TIMESTEPS_PER_ACTOR_BATCH], dtype=np.float)
        self.session = session
        # Here we get 4 action output heads
        # 1 --- Meta action type
        # 0 : stay where they are
        # 1 : move
        # 2 : normal attack
        # 3 : skill 1
        # 4 : skill 2
        # 5 : skill 3
        # 6 : extra skill

        # Why we need seperate move direction and skill direction to two output heads??
        # Answer: Help resolving sample independence.
        # 2 --- Move direction, 8 directions, 0-7
        # 3 --- Skill direction, 8 directions, 0-7
        # 4 --- Skill target area, 0-8
        self.a_space = a_space
        self.a_space_keys = a_space_keys
        self.policy_head_num = len(a_space)

        self.input_dims = INPUT_DIMENS
        self.learning_rate = LEARNING_RATE
        self.num_total_steps = NUM_STEPS
        self.lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=100) # rolling buffer for clipped episode rewards
        self.unclipped_rewbuffer = deque(maxlen=100) # rolling buffer for unclipped episode rewards
        
        self.restrict_x_min = -0.5
        self.restrict_x_max = 0.5
        self.restrict_y_min = -0.5
        self.restrict_y_max = 0.5

        # Full connection layer node size
        self.layer_size = 32

        self._init_input()
        self._init_nn()
        self._init_op()

        self.session.run(tf.global_variables_initializer())        
        

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            TEST = [109, *self.input_dims]
            self.s = tf.placeholder(tf.float32, [None, *self.input_dims], name='s')

            # Action shall have four elements, 
            self.a = tf.placeholder(tf.int32, [None, self.policy_head_num], name='a')

            self.cumulative_r = tf.placeholder(tf.float32, [None, ], name='cumulative_r')
            self.adv = tf.placeholder(tf.float32, [None, ], name='adv')
            self.importance_sample_arr_pl = tf.placeholder(tf.float32, [None, ], name='importance_sample')
            self.lstm_hidden = tf.placeholder(tf.float32, [None, HIDDEN_STATE_LEN * 2], name='lstm_hidden')
            self.lstm_mask = tf.placeholder(tf.int32, [None, ], name='lstm_mask')
            self.is_inference = tf.placeholder(tf.bool, name='is_inference')

            tf.summary.histogram("input_state", self.s)
            tf.summary.histogram("input_action", self.a)
            tf.summary.histogram("input_cumulative_r", self.cumulative_r)
            tf.summary.histogram("input_adv", self.adv)
            tf.summary.histogram("lstm_hidden", self.lstm_hidden)


    def _init_nn(self, *args):
        self.a_policy_new, self.a_policy_logits_new, self.value, self.hidden_state_new, self.summary_train_new = self._init_actor_net('policy_net_new', trainable=True, is_inference=False)
        self.a_policy_new_infer, _1, self.value_infer, self.hidden_state_new_infer, self.summary_infer_new = self._init_actor_net('policy_net_new', trainable=True, is_inference=True)
        self.a_policy_old, self.a_policy_logits_old, _, self.hidden_state_old, self.summary_train_old = self._init_actor_net('policy_net_old', trainable=False, is_inference=False)

    def _init_op(self):
        self.lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

        with tf.variable_scope('update_target_actor_net'):
            # Get eval w, b.
            params_new = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_net_new')
            params_old = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_net_old')
            self.update_policy_net_op = [tf.assign(o, n) for o, n in zip(params_old, params_new)]

        
        with tf.variable_scope('critic_loss_func'):
            # loss func.
            #self.c_loss_func = tf.losses.mean_squared_error(labels=self.cumulative_r, predictions=self.value)
            self.c_loss_func_arr = tf.square(self.value - self.cumulative_r)
            self.c_loss_func = tf.reduce_mean(self.c_loss_func_arr)

        with tf.variable_scope('actor_loss_func'):
            batch_size = tf.shape(self.a)[0]

            ratio = tf.ones([batch_size, ], dtype = tf.float32) 
            for idx in range(self.policy_head_num):
                a = tf.slice(self.a, [0, idx], [batch_size, 1]) # a = a[:,idx]
                a = tf.squeeze(a)
                a_mask_cond = tf.equal(a, -1)
                a = tf.where(a_mask_cond, tf.zeros([batch_size, ], dtype=tf.int32), a)
                # We correct -1 value in a, in order to make the condition operation below work
                a_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), a], axis=1)
                
                new_policy_prob = tf.where(a_mask_cond, tf.ones([batch_size, ], dtype=tf.float32), tf.gather_nd(params=self.a_policy_new[idx], indices=a_indices))
                old_policy_prob = tf.where(a_mask_cond, tf.ones([batch_size, ], dtype=tf.float32), tf.gather_nd(params=self.a_policy_old[idx], indices=a_indices))

                ratio = tf.multiply(ratio, tf.exp(tf.log(new_policy_prob) - tf.log(old_policy_prob)))

            surr = ratio * self.adv                                 # surrogate loss
            self.a_loss_func_arr = tf.minimum(        # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - EPSILON*self.lrmult, 1. + EPSILON*self.lrmult) * self.adv)            
            self.a_loss_func = -tf.reduce_mean(self.a_loss_func_arr)

        with tf.variable_scope('kl_distance'):
            self.kl_distance = tf.zeros([1, ], dtype = tf.float32)
            for idx in range(self.policy_head_num):
                a0 = self.a_policy_logits_new[idx] - tf.reduce_max(self.a_policy_logits_new[idx], axis=-1, keepdims=True)
                a1 = self.a_policy_logits_old[idx] - tf.reduce_max(self.a_policy_logits_old[idx], axis=-1, keepdims=True)
                ea0 = tf.exp(a0)
                ea1 = tf.exp(a1)
                z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
                p0 = ea0 / z0
                self.kl_distance += tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

        with tf.variable_scope('policy_entropy'):            
            self.policy_entropy_arr = tf.zeros([BATCH_SIZE, ], dtype = tf.float32)
            for idx in range(self.policy_head_num):
                a0 = self.a_policy_logits_new[idx] - tf.reduce_max(self.a_policy_logits_new[idx] , axis=-1, keepdims=True)
                ea0 = tf.exp(a0)
                z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                p0 = ea0 / z0
                self.policy_entropy_arr += tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
                
            self.policy_entropy = tf.reduce_mean(self.policy_entropy_arr)

        with tf.variable_scope('optimizer'):
            
            
            if g_enable_per:          
                self.total_loss_arr = -self.a_loss_func_arr + self.c_loss_func_arr - 0.01*self.policy_entropy_arr
                self.total_loss_arr = tf.multiply(self.total_loss_arr, self.importance_sample_arr_pl)
                self.total_loss = tf.reduce_mean(self.total_loss_arr)
            else:
                self.total_loss = self.a_loss_func + self.c_loss_func - 0.01*self.policy_entropy 
            '''
            self.total_loss_arr = self.a_loss_func_arr + self.c_loss_func_arr - 0.01*self.policy_entropy_arr
            self.total_loss = self.a_loss_func + self.c_loss_func - 0.01*self.policy_entropy 
            '''
            tf.summary.scalar("total_loss", self.total_loss)

            learning_rate = self.learning_rate * self.lrmult
            # Passing global_step to minimize() will increment it at each step.
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
        #     self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.total_loss)
            #self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)



    def _init_actor_net(self, scope, trainable=True, is_inference = False):        
        my_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            last_output_dims = 0
            last_output = None
            last_hidden_state = None
            if USE_CNN:
                cnn_w_1 = tf.get_variable("cnn_w_1", [8, 8, C, 32], initializer=my_initializer)
                cnn_b_1 = tf.get_variable("cnn_b_1", [32], initializer=tf.constant_initializer(0.0))

                output1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.s, cnn_w_1, strides=[1, 4, 4, 1], padding='SAME'), cnn_b_1))
                cnn_w_2 = tf.get_variable("cnn_w_2", [4, 4, 32, 64], initializer=my_initializer)
                cnn_b_2 = tf.get_variable("cnn_b_2", [64], initializer=tf.constant_initializer(0.0))

                output2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(output1, cnn_w_2, strides=[1, 2, 2, 1], padding='SAME'), cnn_b_2))
                cnn_w_3 = tf.get_variable("cnn_w_3", [3, 3, 64, 64], initializer=my_initializer)
                cnn_b_3 = tf.get_variable("cnn_b_3", [64], initializer=tf.constant_initializer(0.0))

                output3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(output2, cnn_w_3, strides=[1, 1, 1, 1], padding='SAME'), cnn_b_3))                
                last_output_dims = np.prod([v.value for v in output3.get_shape()[1:]])
                last_output = tf.reshape(output3, [-1, last_output_dims])
            else:
                flat_output_size = INPUT_DIMENS_FLAT
                flat_output = tf.reshape(self.s, [-1, flat_output_size], name='flat_output')

                fc_W_1 = tf.get_variable(shape=[flat_output_size, self.layer_size], name='fc_W_1',
                    trainable=trainable, initializer=my_initializer)
                fc_b_1 = tf.get_variable(shape=[1], name='fc_b_1',
                    trainable=trainable)

                output1 = tf.nn.relu(tf.matmul(flat_output, fc_W_1) + fc_b_1)

                fc_W_2 = tf.get_variable(shape=[self.layer_size, self.layer_size], name='fc_W_2',
                    trainable=trainable, initializer=my_initializer)
                fc_b_2 = tf.get_variable(shape=[1], name='fc_b_2',
                    trainable=trainable)

                output2 = tf.nn.relu(tf.matmul(output1, fc_W_2) + fc_b_2)


                fc_W_3 = tf.get_variable(shape=[self.layer_size, self.layer_size], name='fc_W_3',
                    trainable=trainable, initializer=my_initializer)
                fc_b_3 = tf.get_variable(shape=[1], name='fc_b_3',
                    trainable=trainable)

                output3 = tf.nn.relu(tf.matmul(output2, fc_W_3) + fc_b_3)

                last_output_dims = self.layer_size
                last_output = output3 

            # Add lstm here, to convert last_output to lstm_output
            tf.summary.histogram("lstm_input", last_output)
            tf.summary.histogram("lstm_input_hidden_state", self.lstm_hidden)

            use_tensorflow_lstm = True
            if use_tensorflow_lstm:
                lstm_input = last_output
                use_keras = True
                if use_keras:                    
                    lstm_layer = tf.keras.layers.LSTM(HIDDEN_STATE_LEN, 
                    return_state=True, 
                    return_sequences = True,
                    kernel_initializer=my_initializer,
                    recurrent_initializer=my_initializer
                    )

                    fold_time_step = TIME_STEP
                    if is_inference:
                        fold_time_step = 1
                    reshaped_lstm_input = tf.reshape(lstm_input, [-1, fold_time_step, last_output_dims])

                    reshaped_lstm_mask = tf.reshape(self.lstm_mask, [-1, fold_time_step])
                    reshaped_lstm_mask = 1 - reshaped_lstm_mask
                    reshaped_lstm_mask = tf.cast(reshaped_lstm_mask, dtype=np.bool)
                    c_old, h_old = tf.split(axis=1, num_or_size_splits=2, value=self.lstm_hidden)
                    c_old = c_old[::fold_time_step, ...]
                    h_old = h_old[::fold_time_step, ...]
                    reshaped_output, last_h, last_c = lstm_layer(reshaped_lstm_input, #mask=reshaped_lstm_mask,
                                                                    initial_state=[h_old, c_old])

                    '''                                                
                    layer_weights = lstm_layer.get_weights()
                    
                    for idx in range(len(layer_weights)):
                        new_tensor = tf.convert_to_tensor(layer_weights[idx])
                        tf.summary.histogram("lstm_layer_{}".format(idx), new_tensor)
                    '''

                    last_c = tf.reshape(last_c, [-1, HIDDEN_STATE_LEN])
                    last_h = tf.reshape(last_h, [-1, HIDDEN_STATE_LEN])
                    last_hidden_state = tf.concat(axis=1, values=[last_c, last_h])
                    last_output = tf.reshape(reshaped_output, [-1, HIDDEN_STATE_LEN])

                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_STATE_LEN, name='lstm_cell', dynamic = True)
                    c_old, h_old = tf.split(axis=1, num_or_size_splits=2, value=self.lstm_hidden)
                    c_old, h_old = c_old[:,::TIME_STEP], h_old[:,::TIME_STEP]
                    combined_hidden_states = tf.stack([c_old, h_old], axis = 2)
                    lstm_input = tf.reshape(lstm_input, (-1, TIME_STEP, *(lstm_input.get_shape()[1:])))
                    lstm_input = tf.unstack(lstm_input, axis = 1)
                    last_output, last_hidden_state = tf.nn.static_rnn(cell=lstm_cell, inputs=lstm_input, initial_state = combined_hidden_states)
                    # last_output, last_hidden_state = lstm_cell(inputs=lstm_input, state=(c_old, h_old))
            #        last_output, last_hidden_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_input, initial_state = combined_hidden_states)
            #        last_output = last_output[0]

            else:
                lstm_input = last_output
                last_output, last_hidden_state = utils.lstm(lstm_input, self.is_inference, self.lstm_hidden, self.lstm_mask, 'lstm', HIDDEN_STATE_LEN, LSTM_CELL_COUNT, my_initializer)

            tf.summary.histogram("last_output", last_output)
            tf.summary.histogram("last_hidden_state", last_hidden_state)

            last_output_dims = HIDDEN_STATE_LEN
            a_logits_arr = []
            a_prob_arr = []
            #self.a_space_keys
            for k in self.a_space_keys:
                output_num = self.a_space[k]
                # actor network
                weight_layer_name = 'fc_W_{}'.format(k)
                bias_layer_name = 'fc_b_{}'.format(k)
                logit_layer_name = '{}_logits'.format(k)
                head_layer_name = '{}_head'.format(k)

                fc_W_a = tf.get_variable(shape=[last_output_dims, output_num], name=weight_layer_name,
                    trainable=trainable, initializer=my_initializer)
                fc_b_a = tf.get_variable(shape=[1], name=bias_layer_name,
                    trainable=trainable, initializer=tf.constant_initializer(0.0))

                a_logits = tf.matmul(last_output, fc_W_a) + fc_b_a
                a_logits_arr.append(a_logits)

                a_prob = stable_softmax(a_logits, head_layer_name) #tf.nn.softmax(a_logits)
                a_prob_arr.append(a_prob)
                tf.summary.histogram("a_prob_{}".format(k), a_prob)

            # value network
            fc1_W_v = tf.get_variable(shape=[last_output_dims, 1], name='fc1_W_v',
                trainable=trainable, initializer=my_initializer)
            fc1_b_v = tf.get_variable(shape=[1], name='fc1_b_v',
                trainable=trainable, initializer=tf.constant_initializer(0.0))

            value = tf.matmul(last_output, fc1_W_v) + fc1_b_v
            value = tf.reshape(value, [-1, ], name = "value_output")
            tf.summary.histogram("value", value)

            summary_merged = tf.summary.merge_all()

            return a_prob_arr, a_logits_arr, value, last_hidden_state, summary_merged

    def mask_invalid_action(self, s, distrib):
        new_distrib = distrib
        delta = 0.02
        if s[0] < self.restrict_x_min + delta:
            new_distrib[1] = 0
            new_distrib[4] = 0
            new_distrib[6] = 0
        if s[1] < self.restrict_y_min + delta:
            new_distrib[1] = 0
            new_distrib[2] = 0
            new_distrib[3] = 0
        if s[0] > self.restrict_x_max - delta:
            new_distrib[3] = 0
            new_distrib[5] = 0
            new_distrib[8] = 0
        if s[1] > self.restrict_y_max - delta:
            new_distrib[6] = 0
            new_distrib[7] = 0
            new_distrib[8] = 0

        return new_distrib

    def predict(self, s, hidden_state_val, mask_state_val):
        # Calculate a eval prob.
        tuple_val = self.session.run([self.value_infer, self.a_policy_new_infer[0], self.hidden_state_new_infer], 
        feed_dict={self.s: s, self.lstm_hidden: hidden_state_val, self.lstm_mask: mask_state_val, self.is_inference:True})
        value = tuple_val[0]
        hidden_state_val  = tuple_val[-1]
        chosen_policy = tuple_val[1:] 
        #chosen_policy = self.session.run(self.a_policy_new, feed_dict={self.s: s})
        actions = []
        for idx in range(self.policy_head_num):            
            ac = np.random.choice(range(chosen_policy[idx].shape[1]), p=chosen_policy[idx][0])
            actions.append(ac)

        return actions, value, hidden_state_val[0]

    def greedy_predict(self, s):
        # Calculate a eval prob.
        tuple_val = self.session.run([self.value, self.a_policy_new[0], self.a_policy_new[1], self.a_policy_new[2]], feed_dict={self.s: s})
        value = tuple_val[0]
        chosen_policy = tuple_val[1:] 
        #chosen_policy = self.session.run(self.a_policy_new, feed_dict={self.s: s})
        actions = []
        for idx in range(self.policy_head_num):
            ac = np.argmax(chosen_policy[idx][0])
            actions.append(ac)
        if actions[0] == 0:
            # Stay still
            actions[1] = -1
            actions[2] = -1
        elif actions[0] == 1:
            # Move
            actions[2] = -1
        elif actions[0] == 2:
            # Normal attack
            actions[1] = -1
            actions[2] = -1
        elif actions[0] == 3:
            # skill 1 attack
            actions[1] = -1
            actions[2] = -1
        elif actions[0] == 4:
            # skill 2 attack
            actions[1] = -1
            actions[2] = -1        
        elif actions[0] == 5:
            # skill 3 attack
            actions[1] = -1
            actions[2] = -1
        elif actions[0] == 6:
            # skill 4 attack
            actions[1] = -1
        #    actions[2] = -1 
        else:
            print('Action predict wrong:{}'.format(actions[0]))

        return actions, value

    def do_per_sample(self, atarg, draw_count, alpha):        
        total_sample_num = len(atarg)

        distribution = np.power(np.absolute(atarg), alpha)
        dist_sum = np.sum(distribution)
        normalized_dist = distribution / dist_sum
        indices = np.random.choice(range(total_sample_num), draw_count, p=normalized_dist, replace=False)

        return indices, normalized_dist
    #    pass

    def learn_one_traj_per(self, timestep, ob, ac, atarg, tdlamret, seg, train_writer, beta):
        global g_step
        self.session.run(self.update_policy_net_op)

        loss_sum_arr = np.ones([len(ob)], dtype=np.float) * 10.0      

        lrmult = max(1.0 - float(timestep) / self.num_total_steps, .0)

        Entropy_list = []
        KL_distance_list = []
        bigN = len(ob)
        draw_count = bigN//BATCH_SIZE
        loop_count = EPOCH_NUM * draw_count
        for _idx in range(loop_count):            
            indices, distribution = self.do_per_sample(loss_sum_arr, BATCH_SIZE, g_per_alpha)
            #indices, distribution = self.do_per_sample(atarg, BATCH_SIZE, 1)
            # Update importance sample coefficiency
            _is_beta = beta
            self.importance_sample_arr = np.power(distribution * bigN, -_is_beta)

            temp_indices = indices
            batch_is_val = self.importance_sample_arr[temp_indices]
            batch_is_val /= np.max(batch_is_val)
            # Minimize loss.
            loss_val, _, entropy, kl_distance, summary_new_val, summary_old_val = self.session.run([self.total_loss_arr, self.optimizer, self.policy_entropy, self.kl_distance, self.summary_new, self.summary_old], {
                self.lrmult : lrmult,
                self.adv: atarg[temp_indices],
                self.s: ob[temp_indices],
                self.a: ac[temp_indices],
                self.cumulative_r: tdlamret[temp_indices],
                self.importance_sample_arr_pl: batch_is_val,
            })

            loss_sum_arr[temp_indices] = np.absolute(loss_val) + 1e-5
            
            if g_out_tb and _idx == loop_count - 1:
                g_step += 1
                train_writer.add_summary(summary_new_val, g_step)
                train_writer.add_summary(summary_old_val, g_step)

            Entropy_list.append(entropy)
            KL_distance_list.append(kl_distance)

        self.lenbuffer.extend(seg["ep_lens"])
        self.rewbuffer.extend(seg["ep_rets"])
        self.unclipped_rewbuffer.extend(seg["ep_unclipped_rets"])

        return np.mean(Entropy_list), np.mean(KL_distance_list)

    def learn_one_traj(self, timestep, seg, train_writer):
        global g_step
        ob, ac, atarg, tdlamret, game_ends = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg['new']
        lens, rets, unclipped_rets = seg["ep_lens"], seg["ep_rets"], seg["ep_unclipped_rets"]
        hidden_states = seg["hidden_states"]

        self.session.run(self.update_policy_net_op)

        lrmult = max(1.0 - float(timestep) / self.num_total_steps, .0)

        Entropy_list = []
        KL_distance_list = []
        loss_list = []

        for _idx in range(EPOCH_NUM):
            indices = np.random.permutation(int(len(ob) / TIME_STEP)) # list(range(len(ob))) # 
            indices = inflate_random_indice(indices, TIME_STEP)
            inner_loop_count = (len(ob)//BATCH_SIZE)
            for i in range(inner_loop_count):
                g_step += 1
                temp_indices = indices[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # Minimize loss.
                loss_val, _, entropy, kl_distance, summary_train_new_val, summary_infer_new_val, summary_train_old_val = self.session.run([self.total_loss, self.optimizer, self.policy_entropy, 
                    self.kl_distance, self.summary_train_new, self.summary_infer_new, self.summary_train_old], {
                    self.lrmult : lrmult,
                    self.adv: atarg[temp_indices],
                    self.s: ob[temp_indices],
                    self.a: ac[temp_indices],
                    self.cumulative_r: tdlamret[temp_indices],
                    self.lstm_hidden: hidden_states[temp_indices],
                    self.lstm_mask: game_ends[temp_indices],
                    self.is_inference:False
                })

                Entropy_list.append(entropy)
                KL_distance_list.append(kl_distance)
                loss_list.append(loss_val)

        #        train_writer.add_summary(summary_train_new_val, g_step)
        #        train_writer.add_summary(summary_infer_new_val, g_step)   
        #        train_writer.add_summary(summary_train_old_val, g_step)             

        self.lenbuffer.extend(lens)
        self.rewbuffer.extend(rets)
        self.unclipped_rewbuffer.extend(unclipped_rets)
        return np.mean(Entropy_list), np.mean(KL_distance_list), np.mean(loss_list)

def GetDataGeneratorAndTrainer(scene_id):   
    session = tf.Session()
    action_space_map = {'action':ACTION_COUNT}
    a_space_keys = ['action']
    agent = Agent(session, action_space_map, a_space_keys)
    data_generator = Data_Generator(agent)
    return agent, data_generator, session



def learn(num_steps=NUM_STEPS):
    global g_step
    g_step = 0    
    scene_id = 0
    agent, data_generator, session = GetDataGeneratorAndTrainer(scene_id)

    root_folder = os.path.split(os.path.abspath(__file__))[0]
    train_writer = tf.summary.FileWriter('{}/../summary_log_gerry'.format(root_folder), graph=tf.get_default_graph()) 

    saver = tf.train.Saver(max_to_keep=1)
    if False == g_start_anew:
        model_file=tf.train.latest_checkpoint('ckpt/')
        if model_file != None:
            saver.restore(session,model_file)

    _save_frequency = 50
    max_rew = -1000000
    for timestep in range(num_steps):
        g_step += 1
        seg = data_generator.get_one_step_data()
        
        if g_enable_per:
            is_beta = g_is_beta_start + (timestep / num_steps) * (g_is_beta_end - g_is_beta_start)
            entropy, kl_distance = agent.learn_one_traj_per(timestep, ob, ac, atarg, tdlamret, seg, train_writer, is_beta)
        else:
            entropy, kl_distance, loss_val = agent.learn_one_traj(timestep, seg, train_writer)

        max_rew = max(max_rew, np.max(agent.unclipped_rewbuffer))
        if (timestep+1) % _save_frequency == 0:
            saver.save(session,'ckpt/mnist.ckpt', global_step=g_step)
            if g_save_pb_model:
                tf.saved_model.simple_save(session,
                        "./model/model_{}".format(g_step),
                        inputs={"input_state":agent.s},
                        outputs={"output_policy_0": agent.a_policy_new[0], "output_policy_1": agent.a_policy_new[1], "output_policy_2": agent.a_policy_new[2], "output_value":agent.value})            
        
        summary0 = tf.Summary()
        summary0.value.add(tag='EpLenMean', simple_value=np.mean(agent.lenbuffer))
        train_writer.add_summary(summary0, g_step)

        summary1 = tf.Summary()
        summary1.value.add(tag='UnClippedEpRewMean', simple_value=np.mean(agent.unclipped_rewbuffer))
        train_writer.add_summary(summary1, g_step)

        print('Timestep:', timestep,
            "\tEpLenMean:", '%.3f'%np.mean(agent.lenbuffer),
            "\tEpRewMean:", '%.3f'%np.mean(agent.rewbuffer),
            "\tUnClippedEpRewMean:", '%.3f'%np.mean(agent.unclipped_rewbuffer),
            "\tMaxUnClippedRew:", max_rew,
            "\tEntropy:", '%.3f'%entropy,
            "\tKL_distance:", '%.8f'%kl_distance,
            "\tloss:", '{}'.format(loss_val))

def play_game():
    session = tf.Session()
    action_space_map = {'action':7, 'move':8, 'skill':8}
    a_space_keys = ['action', 'move', 'skill']
    agent = Agent(session, action_space_map, a_space_keys)

    saver = tf.train.Saver(max_to_keep=1)
    model_file=tf.train.latest_checkpoint('ckpt/')
    if model_file != None:
        saver.restore(session, model_file)

    env = Environment()
    
    ob = env.reset()

    steps = 0

    while True:
        steps += 1
        time.sleep(0.2)
        ac, _ = agent.greedy_predict(ob[np.newaxis, ...])
        print('Predict :{}'.format(ac))

        ob, unclipped_rew, new, _ = env.step(ac)
        if new:
            steps = 0
            print('Game is finishd, reward is:{}'.format(unclipped_rew))
            ob = env.reset()

    pass

def play_game_with_saved_model():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./model/model_{}".format(328704))
        input_tensor = sess.graph.get_tensor_by_name('input/s:0')
        policy_tensor = sess.graph.get_tensor_by_name('policy_net_new/soft_logits:0')
        value_tensor = sess.graph.get_tensor_by_name('policy_net_new/value_output:0')

        env = Environment()
        ob = env.reset()

        while True:
            time.sleep(0.05)

            chosen_policy, _ = sess.run([policy_tensor, value_tensor], feed_dict={input_tensor: ob[np.newaxis, ...]})
            tac = np.argmax(chosen_policy[0])

            #print('Predict :{}, input:{}, output:{}'.format(tac, ob, chosen_policy))

            ob, reward, new, _ = env.step(tac)
            if new:
                print('Game is finishd, reward is:{}'.format(reward))
                ob = env.reset()


    pass

def static_rnn():
    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_STATE_LEN, name='lstm_cell', dynamic = False)
    lstm_input = tf.placeholder(tf.float32, (None, 128))
    combined_hidden_states = tf.placeholder(tf.float32, (None, 2, HIDDEN_STATE_LEN))

    lstm_input = tf.reshape(lstm_input, (-1, TIME_STEP, *(lstm_input.get_shape()[1:])))
    lstm_input = tf.unstack(lstm_input, axis = 1)
    last_output, last_hidden_state = tf.nn.static_rnn(cell=lstm_cell, inputs=lstm_input, initial_state = combined_hidden_states)    
    pass

def dyn_rnn():
    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_STATE_LEN, name='lstm_cell', dynamic = True)
    lstm_input = tf.placeholder(tf.float32, (None, 128))
    combined_hidden_states = tf.placeholder(tf.float32, (None, HIDDEN_STATE_LEN))

    lstm_input = tf.reshape(lstm_input, (-1, TIME_STEP, *(lstm_input.get_shape()[1:])))

    last_output, last_hidden_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_input, initial_state = combined_hidden_states)  
    pass

def dyn_rnn_2():
    batch_size=10
    hidden_size=100
    max_time=40
    depth=400
    input_data=tf.Variable(tf.random_normal([batch_size, max_time, depth]))
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    #initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    initial_state_c = tf.get_variable('initial_state_c', [batch_size, hidden_size], dtype = tf.float32)
    initial_state_h = tf.get_variable('initial_state_h', [batch_size, hidden_size], dtype = tf.float32)
    initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state_c, initial_state_h)
    #tf.Variable(tf.random_normal([batch_size, hidden_size], dtype=tf.float32), dtype=tf.float32)
    # #lstm_cell.zero_state(batch_size, dtype=tf.float32)#
    #tf.get_variable('initial_state', [batch_size, hidden_size], dtype = tf.float32)
    # #tf.placeholder(tf.float32, (batch_size, hidden_size))
    # #lstm_cell.zero_state(batch_size, dtype=tf.float32)#
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_data, initial_state = initial_state, dtype=tf.float32, time_major = False)
#    outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_data, initial_state=initial_state, dtype=tf.float32, time_major=True)   
 
    pass

def rnn_3():
    input_indices = [4, 3, 7, 9, 0, 1, 2, 5, 6, 8]
    input_indices = inflate_random_indice(input_indices, 2)
    x = tf.constant(2)
    y = tf.constant(5)
    y0 = tf.constant([10, 2, 3, 4, 5, 6])
    y1 = tf.reshape(y0, [-1, 3])
    z = tf.placeholder(tf.bool)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_STATE_LEN, name='lstm_cell', dynamic = False)
    lstm_input = tf.placeholder(tf.float32, (None, 128))
    combined_hidden_states = tf.placeholder(tf.float32, (None, 2, HIDDEN_STATE_LEN))

    lstm_input = tf.reshape(lstm_input, (-1, TIME_STEP, *(lstm_input.get_shape()[1:])))
    lstm_input = tf.unstack(lstm_input, axis = 1)
    last_output, last_hidden_state = tf.nn.static_rnn(cell=lstm_cell, inputs=lstm_input, initial_state = combined_hidden_states)
    # last_output, last_hidden_state = lstm_cell(inputs=lstm_input, state=(c_old, h_old))
#        last_output, last_hidden_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=lstm_input, initial_state = combined_hidden_states)
#        last_output = last_output[0]
    def f1():
        return tf.multiply(x, 16)
    def f2():
        return tf.add(y, 23)

    r = tf.cond(z, f1, f2)

    session = tf.Session()
    r_val, y1_val = session.run([r, y1], feed_dict = {z:True})


def test():

    dyn_rnn_2()


    pass


if __name__=='__main__':
#    test()
#    exit(0)
    if g_is_train:
        learn(num_steps=5000)
    else:
        play_game()