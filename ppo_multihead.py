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


EPSILON = 0.2

F = 8
C = 1

NUM_FRAME_PER_ACTION = 4
BATCH_SIZE = 64
EPOCH_NUM = 4
LEARNING_RATE = 1e-3
TIMESTEPS_PER_ACTOR_BATCH = 256*8
GAMMA = 0.99
LAMBDA = 0.95
NUM_STEPS = 5000
ENV_NAME = 'gym_moba:moba-v0'
RANDOM_START_STEPS = 4

global g_step

# Save model in pb format
g_save_pb_model = False

# Control if output to tensorboard
g_out_tb = True

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

def stable_softmax(logits, name): 
    a = logits - tf.reduce_max(logits, axis=-1, keepdims=True) 
    ea = tf.exp(a) 
    z = tf.reduce_sum(ea, axis=-1, keepdims=True) 
    return tf.div(ea, z, name = name) 

class Environment(object):
  def __init__(self):
    self.env = gym.make(ENV_NAME)
    self._screen = None
    self.reward = 0
    self.terminal = True
    self.random_start = RANDOM_START_STEPS
    self.obs = np.zeros(shape=(F,C), dtype=np.float)

  def get_action_number(self):
    return 9

  def step(self, action):
    self._screen, self.reward, self.terminal, info = self.env.step(action)
    self.obs[..., :-1] = self.obs[..., 1:]
    self.obs[..., -1] = self._screen
    return self.obs, self.reward, self.terminal, info

  def reset(self):
    self._screen = self.env.reset()
    ob, _1, _2, _3 = self.step([0, 0, 0]) 

    return ob

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
        ac = [0, 0, 0]

        new = True # marks if we're on first timestep of an episode
        ob = self.env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_unclipped_ret = 0 # unclipped return in current episode
        cur_ep_len = 0 # len of current episode
        ep_rets = [] # returns of completed episodes in this segment
        ep_unclipped_rets = [] # unclipped returns of completed episodes in this segment
        ep_lens = [] # lengths of ...

        # Initialize history arrays
        obs = np.array([np.zeros(shape=(F,C), dtype=np.float32) for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        unclipped_rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            ac, vpred = self.agent.predict(ob[np.newaxis, ...])
            #print('Action:', ac, 'Value:', vpred)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                        "ep_unclipped_rets": ep_unclipped_rets}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_unclipped_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, unclipped_rew, new, step_info = self.env.step(ac)
            rew = unclipped_rew
            # rew = float(np.sign(unclipped_rew))
            rews[i] = rew
            unclipped_rews[i] = unclipped_rew

            cur_ep_ret += rew
            cur_ep_unclipped_ret += unclipped_rew
            cur_ep_len += 1
            if new or step_info > 600:
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
        return ob, ac, atarg, tdlamret, seg

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

        self.input_dims = F
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
        self.layer_size = 128

        self._init_input()
        self._init_nn()
        self._init_op()

        self.session.run(tf.global_variables_initializer())        
        

    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(tf.float32, [None, self.input_dims, C], name='s')

            # Action shall have four elements, 
            self.a = tf.placeholder(tf.int32, [None, self.policy_head_num], name='a')

            self.cumulative_r = tf.placeholder(tf.float32, [None, ], name='cumulative_r')
            self.adv = tf.placeholder(tf.float32, [None, ], name='adv')
            self.importance_sample_arr_pl = tf.placeholder(tf.float32, [None, ], name='importance_sample')

            tf.summary.histogram("input_state", self.s)
            tf.summary.histogram("input_action", self.a)
            tf.summary.histogram("input_cumulative_r", self.cumulative_r)
            tf.summary.histogram("input_adv", self.adv)


    def _init_nn(self, *args):
        self.a_policy_new, self.a_policy_logits_new, self.value, self.summary_new = self._init_actor_net('policy_net_new', trainable=True)
        self.a_policy_old, self.a_policy_logits_old, _, self.summary_old = self._init_actor_net('policy_net_old', trainable=False)

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
            learning_rate = self.learning_rate * self.lrmult
            # Passing global_step to minimize() will increment it at each step.
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
            #self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)



    def _init_actor_net(self, scope, trainable=True):        
        my_initializer = None#tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            flat_output_size = F*C
            flat_output = tf.reshape(self.s, [-1, flat_output_size], name='flat_output')

            fc_W_1 = tf.get_variable(shape=[flat_output_size, self.layer_size], name='fc_W_1',
                trainable=trainable, initializer=my_initializer)
            fc_b_1 = tf.Variable(tf.zeros([self.layer_size], dtype=tf.float32), name='fc_b_1',
                trainable=trainable)

            tf.summary.histogram("fc_W_1", fc_W_1)
            tf.summary.histogram("fc_b_1", fc_b_1)

            output1 = tf.nn.relu(tf.matmul(flat_output, fc_W_1) + fc_b_1)

            fc_W_2 = tf.get_variable(shape=[self.layer_size, self.layer_size], name='fc_W_2',
                trainable=trainable, initializer=my_initializer)
            fc_b_2 = tf.Variable(tf.zeros([self.layer_size], dtype=tf.float32), name='fc_b_2',
                trainable=trainable)

            tf.summary.histogram("fc_W_2", fc_W_2)
            tf.summary.histogram("fc_b_2", fc_b_2)

            output2 = tf.nn.relu(tf.matmul(output1, fc_W_2) + fc_b_2)


            fc_W_3 = tf.get_variable(shape=[self.layer_size, self.layer_size], name='fc_W_3',
                trainable=trainable, initializer=my_initializer)
            fc_b_3 = tf.Variable(tf.zeros([self.layer_size], dtype=tf.float32), name='fc_b_3',
                trainable=trainable)

            tf.summary.histogram("fc_W_3", fc_W_3)
            tf.summary.histogram("fc_b_3", fc_b_3)

            output3 = tf.nn.relu(tf.matmul(output2, fc_W_3) + fc_b_3)
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

                fc_W_a = tf.get_variable(shape=[self.layer_size, output_num], name=weight_layer_name,
                    trainable=trainable, initializer=my_initializer)
                fc_b_a = tf.Variable(tf.zeros([output_num], dtype=tf.float32), name=bias_layer_name,
                    trainable=trainable)

                tf.summary.histogram(weight_layer_name, fc_W_a)
                tf.summary.histogram(bias_layer_name, fc_b_a)            

                a_logits = tf.matmul(output3, fc_W_a) + fc_b_a
                a_logits_arr.append(a_logits)                
                tf.summary.histogram(logit_layer_name, a_logits)

                a_prob = stable_softmax(a_logits, head_layer_name) #tf.nn.softmax(a_logits)
                a_prob_arr.append(a_prob)
                tf.summary.histogram(head_layer_name, a_prob)

            # value network
            fc1_W_v = tf.get_variable(shape=[self.layer_size, 1], name='fc1_W_v',
                trainable=trainable, initializer=my_initializer)
            fc1_b_v = tf.Variable(tf.zeros([1], dtype=tf.float32), name='fc1_b_v',
                trainable=trainable)

            tf.summary.histogram("fc1_W_v", fc1_W_v)
            tf.summary.histogram("fc1_b_v", fc1_b_v)

            value = tf.matmul(output3, fc1_W_v) + fc1_b_v
            value = tf.reshape(value, [-1, ], name = "value_output")
            tf.summary.histogram("value_head", value)
            merged_summary = tf.summary.merge_all()
            return a_prob_arr, a_logits_arr, value, merged_summary

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

    def predict(self, s):
        # Calculate a eval prob.
        tuple_val = self.session.run([self.value, self.a_policy_new[0], self.a_policy_new[1], self.a_policy_new[2]], feed_dict={self.s: s})
        value = tuple_val[0]
        chosen_policy = tuple_val[1:] 
        #chosen_policy = self.session.run(self.a_policy_new, feed_dict={self.s: s})
        actions = []
        for idx in range(self.policy_head_num):            
            ac = np.random.choice(range(chosen_policy[idx].shape[1]), p=chosen_policy[idx][0])
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
            #actions[2] = -1
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

    def learn_one_traj(self, timestep, ob, ac, atarg, tdlamret, seg, train_writer):
        global g_step
        self.session.run(self.update_policy_net_op)

        lrmult = max(1.0 - float(timestep) / self.num_total_steps, .0)

        Entropy_list = []
        KL_distance_list = []

        for _idx in range(EPOCH_NUM):
            indices = np.random.permutation(len(ob))
            inner_loop_count = (len(ob)//BATCH_SIZE)
            for i in range(inner_loop_count):
                temp_indices = indices[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # Minimize loss.
                _, entropy, kl_distance, summary_new_val, summary_old_val = self.session.run([self.optimizer, self.policy_entropy, self.kl_distance, self.summary_new, self.summary_old], {
                    self.lrmult : lrmult,
                    self.adv: atarg[temp_indices],
                    self.s: ob[temp_indices],
                    self.a: ac[temp_indices],
                    self.cumulative_r: tdlamret[temp_indices],
                })
                
                if g_out_tb and i == (inner_loop_count - 1) and _idx == (EPOCH_NUM -1):
                    g_step += 1
                    train_writer.add_summary(summary_new_val, g_step)
                    train_writer.add_summary(summary_old_val, g_step)

                Entropy_list.append(entropy)
                KL_distance_list.append(kl_distance)

        self.lenbuffer.extend(seg["ep_lens"])
        self.rewbuffer.extend(seg["ep_rets"])
        self.unclipped_rewbuffer.extend(seg["ep_unclipped_rets"])
        return np.mean(Entropy_list), np.mean(KL_distance_list)



def learn(num_steps=NUM_STEPS):
    global g_step
    g_step = 0
    session = tf.Session()

    action_space_map = {'action':7, 'move':8, 'skill':8}
    a_space_keys = ['action', 'move', 'skill']
    agent = Agent(session, action_space_map, a_space_keys)

    data_generator = Data_Generator(agent)
    train_writer = tf.summary.FileWriter('summary_log_gerry', graph=tf.get_default_graph()) 

    saver = tf.train.Saver(max_to_keep=1)
    if False == g_start_anew:
        model_file=tf.train.latest_checkpoint('ckpt/')
        if model_file != None:
            saver.restore(session,model_file)

    _save_frequency = 50
    max_rew = -1000000
    for timestep in range(num_steps):
        ob, ac, atarg, tdlamret, seg = data_generator.get_one_step_data()
        if g_enable_per:
            is_beta = g_is_beta_start + (timestep / num_steps) * (g_is_beta_end - g_is_beta_start)
            entropy, kl_distance = agent.learn_one_traj_per(timestep, ob, ac, atarg, tdlamret, seg, train_writer, is_beta)
        else:
            entropy, kl_distance = agent.learn_one_traj(timestep, ob, ac, atarg, tdlamret, seg, train_writer)

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
            "\tKL_distance:", '%.8f'%kl_distance)

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


if __name__=='__main__':
    bb = {1:'a', 2:'b', 3:'c'}
    #print(list(bb))
    a = list(map(lambda x: math.pow(x, -2), range(1, 10)))
    for val in a:
        print(val)
    print(a)

    try:
        # Write control file
        root_folder = os.path.split(os.path.abspath(__file__))[0]
        ctrl_file_path = '{}/ctrl.txt'.format(root_folder)
        file_handle = open(ctrl_file_path, 'w')
        if g_manual_ctrl_enemy:
            file_handle.write('0')
        else:
            file_handle.write('1')

        file_handle.close()
    except:
        pass	  

    if g_is_train:
        learn(num_steps=5000)
    else:
        play_game()