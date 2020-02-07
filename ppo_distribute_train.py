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


g_step = 0


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


def learn(scene_id, num_steps):
    root_folder = os.path.split(os.path.abspath(__file__))[0]
    global g_step
    agent, _, session = GetDataGeneratorAndTrainer(scene_id)

    train_writer = tf.summary.FileWriter('{}/../summary_log_gerry'.format(root_folder), graph=tf.get_default_graph()) 
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

if __name__=='__main__':
    global g_step
    g_step = 0
    scene_id = 10

    if len(sys.argv) > 1:
        g_data_generator_count = int(sys.argv[1])
        
    if len(sys.argv) > 2:    
        g_step = int(sys.argv[2])

    if len(sys.argv) > 3:    
        scene_id = int(sys.argv[3])  

    learn(scene_id, num_steps=500)
