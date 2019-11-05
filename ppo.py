# coding=utf-8
"""
Some of code copy from
https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
"""
import sys
import copy
import numpy as np
import tensorflow as tf
from collections import deque
import gym, os
import random, cv2
import time

EPSILON = 0.2

F = 6
C = 4

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
    self._screen, self.reward, self.terminal, _ = self.env.step(action)
    self.obs[..., :-1] = self.obs[..., 1:]
    self.obs[..., -1] = self._screen
    return self.obs, self.reward, self.terminal, _

  def reset(self):
    self._screen = self.env.reset()
    ob, _1, _2, _3 = self.step(0) 
    for _ in range(random.randint(3, self.random_start - 1)):
      ob, _1, _2, _3 = self.step(0)

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
        ac = 0
        tac = 0
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
            ac, tac, vpred = self.agent.predict(ob[np.newaxis, ...])
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

            ob, unclipped_rew, new, _ = self.env.step(tac)
            rew = unclipped_rew
            # rew = float(np.sign(unclipped_rew))
            rews[i] = rew
            unclipped_rews[i] = unclipped_rew

            cur_ep_ret += rew
            cur_ep_unclipped_ret += unclipped_rew
            cur_ep_len += 1
            if new:
                if cur_ep_unclipped_ret == 0:
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

    def __init__(self, session, a_space, **options):

        self.session = session
        self.a_space = 9
        self.input_dims = F
        self.learning_rate = LEARNING_RATE
        self.num_total_steps = NUM_STEPS
        self.lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=100) # rolling buffer for clipped episode rewards
        self.unclipped_rewbuffer = deque(maxlen=100) # rolling buffer for unclipped episode rewards
        self.restrict_x_min = -0.15
        self.restrict_x_max = 0.15
        self.restrict_y_min = -0.15
        self.restrict_y_max = 0.15

        self._init_input()
        self._init_nn()
        self._init_op()

        self.session.run(tf.global_variables_initializer())


    def _init_input(self, *args):
        with tf.variable_scope('input'):
            self.s = tf.placeholder(tf.float32, [None, self.input_dims, C], name='s')
            self.a = tf.placeholder(tf.int32, [None, ], name='a')
            self.cumulative_r = tf.placeholder(tf.float32, [None, ], name='cumulative_r')
            self.adv = tf.placeholder(tf.float32, [None, ], name='adv')

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
            self.c_loss_func = tf.reduce_mean(tf.square(self.value - self.cumulative_r))

        with tf.variable_scope('actor_loss_func'):

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            new_policy_prob = tf.gather_nd(params=self.a_policy_new, indices=a_indices)   # shape=(None, )
            old_policy_prob = tf.gather_nd(params=self.a_policy_old, indices=a_indices)   # shape=(None, )
            #ratio = new_policy_prob/old_policy_prob
            ratio = tf.exp(tf.log(new_policy_prob) - tf.log(old_policy_prob))
            surr = ratio * self.adv                       # surrogate loss
            self.a_loss_func = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - EPSILON*self.lrmult, 1. + EPSILON*self.lrmult) * self.adv))

        with tf.variable_scope('kl_distance'):
            a0 = self.a_policy_logits_new - tf.reduce_max(self.a_policy_logits_new, axis=-1, keepdims=True)
            a1 = self.a_policy_logits_old - tf.reduce_max(self.a_policy_logits_old, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            ea1 = tf.exp(a1)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
            p0 = ea0 / z0
            self.kl_distance = tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

        with tf.variable_scope('policy_entropy'):
            a0 = self.a_policy_logits_new - tf.reduce_max(self.a_policy_logits_new , axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            self.policy_entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1))


        with tf.variable_scope('optimizer'):
            self.total_loss = self.a_loss_func + self.c_loss_func - 0.01*self.policy_entropy
            learning_rate = self.learning_rate * self.lrmult
            # Passing global_step to minimize() will increment it at each step.
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)
            #self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)



    def _init_actor_net(self, scope, trainable=True):
        my_initializer = None#tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            flat_output_size = F*C
            flat_output = tf.reshape(self.s, [-1, flat_output_size], name='flat_output')

            fc_W_1 = tf.get_variable(shape=[flat_output_size, 64], name='fc_W_1',
                trainable=trainable, initializer=my_initializer)
            fc_b_1 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='fc_b_1',
                trainable=trainable)

            tf.summary.histogram("fc_W_1", fc_W_1)
            tf.summary.histogram("fc_b_1", fc_b_1)

            output1 = tf.nn.relu(tf.matmul(flat_output, fc_W_1) + fc_b_1)

            fc_W_2 = tf.get_variable(shape=[64, 64], name='fc_W_2',
                trainable=trainable, initializer=my_initializer)
            fc_b_2 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='fc_b_2',
                trainable=trainable)

            tf.summary.histogram("fc_W_2", fc_W_2)
            tf.summary.histogram("fc_b_2", fc_b_2)

            output2 = tf.nn.relu(tf.matmul(output1, fc_W_2) + fc_b_2)


            fc_W_3 = tf.get_variable(shape=[64, 64], name='fc_W_3',
                trainable=trainable, initializer=my_initializer)
            fc_b_3 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='fc_b_3',
                trainable=trainable)

            tf.summary.histogram("fc_W_3", fc_W_3)
            tf.summary.histogram("fc_b_3", fc_b_3)

            output3 = tf.nn.relu(tf.matmul(output2, fc_W_3) + fc_b_3)

            # actor network
            fc1_W_a = tf.get_variable(shape=[64, self.a_space], name='fc1_W_a',
                trainable=trainable, initializer=my_initializer)
            fc1_b_a = tf.Variable(tf.zeros([self.a_space], dtype=tf.float32), name='fc1_b_a',
                trainable=trainable)

            tf.summary.histogram("fc1_W_a", fc1_W_a)
            tf.summary.histogram("fc1_b_a", fc1_b_a)            

            temp_val = tf.matmul(output3, fc1_W_a) + fc1_b_a
            
            # Add mask here
            masked_val = -20.0

            cond_x_less = tf.less(self.s[:,0,0], tf.constant(self.restrict_x_min))
            mask_x_less = tf.Variable([0, masked_val, 0, 0, masked_val, 0, masked_val, 0, 0], tf.float32)
            temp_val = tf.where(cond_x_less, tf.add(temp_val, mask_x_less), temp_val)
            
            
            cond_y_less = tf.less(self.s[:,0,1], tf.constant(self.restrict_y_min))
            mask_y_less = tf.Variable([masked_val, masked_val, masked_val, 0, 0, 0, 0, 0, 0], tf.float32)
            temp_val = tf.where(cond_y_less, tf.add(temp_val, mask_y_less), temp_val)

            cond_x_greater = tf.greater(self.s[:,0,0], tf.constant(self.restrict_x_max))
            mask_x_greater = tf.Variable([0, 0, 0, masked_val, 0, masked_val, 0, 0, masked_val], tf.float32)
            temp_val = tf.where(cond_x_greater, tf.add(temp_val, mask_x_greater), temp_val)

            cond_y_greater = tf.greater(self.s[:,0,1], tf.constant(self.restrict_y_max))
            mask_y_greater = tf.Variable([0, 0, 0, 0, 0, 0, masked_val, masked_val, masked_val], tf.float32)            
            temp_val = tf.where(cond_y_greater, tf.add(temp_val, mask_y_greater), temp_val)
            '''
            '''

            a_logits = temp_val
            tf.summary.histogram("a_logits", a_logits)
            a_prob = stable_softmax(a_logits, 'soft_logits') #tf.nn.softmax(a_logits)
            tf.summary.histogram("policy_head", a_prob)

            # value network
            fc1_W_v = tf.get_variable(shape=[64, 1], name='fc1_W_v',
                trainable=trainable, initializer=my_initializer)
            fc1_b_v = tf.Variable(tf.zeros([1], dtype=tf.float32), name='fc1_b_v',
                trainable=trainable)

            tf.summary.histogram("fc1_W_v", fc1_W_v)
            tf.summary.histogram("fc1_b_v", fc1_b_v)

            value = tf.matmul(output3, fc1_W_v) + fc1_b_v
            value = tf.reshape(value, [-1, ])
            tf.summary.histogram("value_head", value)
            merged_summary = tf.summary.merge_all()
            return a_prob, a_logits, value, merged_summary

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
        chosen_policy, value = self.session.run([self.a_policy_new, self.value], feed_dict={self.s: s})
        ac = np.random.choice(range(chosen_policy.shape[1]), p=chosen_policy[0])

        new_distrib = self.mask_invalid_action(s[0,:,0], chosen_policy[0])
           
        # Calculate action prob ratio between eval and target.
        tac = ac
        if new_distrib[ac] == 0:
            tac = 0
        return ac, tac, value

    def learn_one_traj(self, timestep, ob, ac, atarg, tdlamret, seg, train_writer):
        global g_step
        self.session.run(self.update_policy_net_op)

        lrmult = max(1.0 - float(timestep) / self.num_total_steps, .0)

        Entropy_list = []
        KL_distance_list = []
        for _ in range(EPOCH_NUM):
            indices = np.random.permutation(len(ob))
            for i in range(len(ob)//BATCH_SIZE):
                temp_indices = indices[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                # Minimize loss.
                _, entropy, kl_distance, summary_new_val, summary_old_val = self.session.run([self.optimizer, self.policy_entropy, self.kl_distance, self.summary_new, self.summary_old], {
                    self.lrmult : lrmult,
                    self.adv: atarg[temp_indices],
                    self.s: ob[temp_indices],
                    self.a: ac[temp_indices],
                    self.cumulative_r: tdlamret[temp_indices],
                })
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

    agent = Agent(session, 9)
    data_generator = Data_Generator(agent)
    train_writer = tf.summary.FileWriter('summary_log_gerry', graph=tf.get_default_graph()) 

    saver = tf.train.Saver(max_to_keep=1)
    model_file=tf.train.latest_checkpoint('ckpt/')
    if model_file != None:
        saver.restore(session,model_file)

    _save_frequency = 10
    max_rew = -1000000
    for timestep in range(num_steps):
        ob, ac, atarg, tdlamret, seg = data_generator.get_one_step_data()
        entropy, kl_distance = agent.learn_one_traj(timestep, ob, ac, atarg, tdlamret, seg, train_writer)
        max_rew = max(max_rew, np.max(agent.unclipped_rewbuffer))
        if timestep % _save_frequency == 0:
            saver.save(session,'ckpt/mnist.ckpt', global_step=g_step)

        print('Timestep:', timestep,
            "\tEpLenMean:", '%.3f'%np.mean(agent.lenbuffer),
            "\tEpRewMean:", '%.3f'%np.mean(agent.rewbuffer),
            "\tUnClippedEpRewMean:", '%.3f'%np.mean(agent.unclipped_rewbuffer),
            "\tMaxUnClippedRew:", max_rew,
            "\tEntropy:", '%.3f'%entropy,
            "\tKL_distance:", '%.8f'%kl_distance)

def play_game():
    session = tf.Session()
    agent = Agent(session, 9)

    saver = tf.train.Saver(max_to_keep=1)
    model_file=tf.train.latest_checkpoint('ckpt/')
    if model_file != None:
        saver.restore(session, model_file)

    env = Environment()
    
    ob = env.reset()

    while True:
        time.sleep(0.1)
        _, tac, _ = agent.predict(ob[np.newaxis, ...])
        print('Predict :{}'.format(tac))

        ob, reward, new, _ = env.step(tac)
        if new:
            print('Game is finishd, reward is:{}'.format(reward))
            ob = env.reset()

    pass

if __name__=='__main__':
    is_train = True
    try:
        # Write control file
        root_folder = os.path.split(os.path.abspath(__file__))[0]
        ctrl_file_path = '{}/ctrl.txt'.format(root_folder)
        file_handle = open(ctrl_file_path, 'w')
        if is_train:
            file_handle.write('1')
        else:
            file_handle.write('0')

        file_handle.close()
    except:
        pass	  

    if is_train:
        learn()
    else:
        play_game()