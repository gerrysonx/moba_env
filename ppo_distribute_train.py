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
import pickle
import json
from ppo_lstm import GetDataGeneratorAndTrainer


EPSILON = 0.2
# 3 * 3 + 2
ONE_HOT_SIZE = 10
STATE_SIZE = 11
F = STATE_SIZE + 1
EMBED_SIZE = 5
LAYER_SIZE = 128

C = 1
HERO_COUNT = 2
# Hero skill mask, to indicate if a hero skill is a directional one.
g_dir_skill_mask = [[False, False, False, False], [False, False, False, False]]

NUM_FRAME_PER_ACTION = 4
BATCH_SIZE = 512
EPOCH_NUM = 4
LEARNING_RATE = 8e-4
TIMESTEPS_PER_ACTOR_BATCH = 256*512
GAMMA = 0.99
LAMBDA = 0.95
NUM_STEPS = 5000
ENV_NAME = 'gym_moba:moba-multiplayer-v0'
RANDOM_START_STEPS = 4

global g_step

# Generating data worker count
g_data_generator_count = 3

# Use hero id embedding
g_embed_hero_id = False

# Save model in pb format
g_save_pb_model = False

# Control if output to tensorboard
g_out_tb = True

# Control if train or play
g_is_train = True
# True means start a new train task without loading previous model.
g_start_anew = True

# Control if use priority sampling
g_enable_per = False
g_per_alpha = 0.6
g_is_beta_start = 0.4
g_is_beta_end = 1


def get_one_step_data(timestep, work_thread_count):
    root_folder = os.path.split(os.path.abspath(__file__))[0]
    ob, ac, std_atvtg, tdlamret, lens, rets, unclipped_rets = [], [], [], [], [], [], []
    # Enumerate data files under folder
    data_folder_path = '{}/../distribute_collected_train_data/{}'.format(root_folder, timestep)
    collected_all_data_files = False
    while not collected_all_data_files:
        for root, _, files in os.walk(data_folder_path):
            if len(files) < work_thread_count:
                print('Already has {} files, waiting for the worker thread generate more data files.'.format(len(files)))
                break

            for file_name in files:
                full_file_name = '{}/{}'.format(root, file_name)
                with open(full_file_name, 'rb') as file_handle:
                    _seg = pickle.load(file_handle)
                    ob.extend(_seg["ob"])
                    ac.extend(_seg["ac"])
                    std_atvtg.extend(_seg["std_atvtg"])
                    tdlamret.extend(_seg["tdlamret"])
                    lens.extend(_seg["ep_lens"])
                    rets.extend(_seg["ep_rets"])
                    unclipped_rets.extend(_seg["ep_unclipped_rets"])    

            collected_all_data_files = True
            break        
        time.sleep(10)
    print('Successfully collected {} files, data size:{} from {}.'.format(len(files), len(ob), timestep))
    return np.array(ob), np.array(ac), np.array(std_atvtg), np.array(tdlamret), np.array(lens), np.array(rets), np.array(unclipped_rets)


def learn(num_steps=NUM_STEPS):
    root_folder = os.path.split(os.path.abspath(__file__))[0]
    global g_step
    agent, _, session = GetDataGeneratorAndTrainer()

    train_writer = tf.summary.FileWriter('summary_log_gerry', graph=tf.get_default_graph()) 
    saver = tf.train.Saver(max_to_keep=50)
    max_rew = -10000
    base_step = g_step
    for timestep in range(num_steps):
        g_step = base_step + timestep
        ob, ac, atarg, tdlamret, lens, rets, unclipped_rets = get_one_step_data(g_step, g_data_generator_count)

        entropy, kl_distance = agent.learn_one_traj(g_step, ob, ac, atarg, tdlamret, lens, rets, unclipped_rets, train_writer)

        max_rew = max(max_rew, np.max(agent.unclipped_rewbuffer))

        saver.save(session, '{}/../ckpt/mnist.ckpt'.format(root_folder), global_step=g_step + 1)

        '''
        summary0 = tf.Summary()
        summary0.value.add(tag='EpLenMean', simple_value=np.mean(agent.lenbuffer))
        train_writer.add_summary(summary0, g_step)

        summary1 = tf.Summary()
        summary1.value.add(tag='UnClippedEpRewMean', simple_value=np.mean(agent.unclipped_rewbuffer))
        train_writer.add_summary(summary1, g_step)
        '''
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

    while True:
        time.sleep(0.2)
        ac, _ = agent.greedy_predict(ob[np.newaxis, ...])
        print('Predict :{}'.format(ac))

        ob, unclipped_rew, new, _ = env.step(ac)
        if new:
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

def FindHeroSkills(hero_cfg_file_path, hero_id):
    hero_cfg_file = '{}/{}.json'.format(hero_cfg_file_path, hero_id)
    hero_skills = []
    with open(hero_cfg_file, 'r') as file_handle:
        hero_cfg = json.load(file_handle)
        hero_skills = hero_cfg['skills']
        return hero_skills
    
def GetSkillTypes(skill_cfg_file_path, hero_skills):
    skill_dir_type_check = []
    for skill_id in hero_skills:
        skill_dir_type = False
        skill_id = int(skill_id)
        if -1 != skill_id:
            skill_cfg_file = '{}/{}.json'.format(skill_cfg_file_path, skill_id)
            with open(skill_cfg_file, 'r') as file_handle:
                skill_cfg = json.load(file_handle)
                if 0 == skill_cfg['type']:
                    skill_dir_type = True
        skill_dir_type_check.append(skill_dir_type)
    return skill_dir_type_check

if __name__=='__main__':
    global g_step
    g_step = 0
    scene_id = 0

    if len(sys.argv) > 1:
        g_data_generator_count = int(sys.argv[1])
        
    if len(sys.argv) > 2:    
        g_step = int(sys.argv[2])

    if len(sys.argv) > 3:    
        scene_id = int(sys.argv[3])

    try:
        # Load train self heroes skill masks
        root_folder = os.path.split(os.path.abspath(__file__))[0]
        g_dir_skill_mask = []
        
        cfg_file_path = '{}/gamecore/cfg'.format(root_folder)
        training_map_file = '{}/maps/{}.json'.format(cfg_file_path, scene_id)
        hero_cfg_file_path = '{}/heroes'.format(cfg_file_path)
        skill_cfg_file_path = '{}/skills'.format(cfg_file_path)
        map_dict = None
        with open(training_map_file, 'r') as file_handle:
            map_dict = json.load(file_handle)

        for hero_id in map_dict['SelfHeroes']:
            hero_skills = FindHeroSkills(hero_cfg_file_path, hero_id)
            hero_skill_types = GetSkillTypes(skill_cfg_file_path, hero_skills)
            g_dir_skill_mask.append(hero_skill_types)
            
        HERO_COUNT = len(g_dir_skill_mask)

        # Write control file
        ctrl_file_path = '{}/ctrl.txt'.format(root_folder)
        file_handle = open(ctrl_file_path, 'w')
        if g_is_train:
            file_handle.write('1')
        else:
            file_handle.write('0')
        file_handle.write(' ')
        file_handle.write('{}'.format(scene_id))
        file_handle.close()
    except Exception as ex:
        pass	  

    if g_is_train:
        learn(num_steps=500)
    else:
        play_game()