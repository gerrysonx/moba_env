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
from ppo_lstm import LoadModel

def dump_generated_data_2_file(file_name, seg):
    with open(file_name, 'wb') as file_handle:
        pickle.dump(seg, file_handle)

    pass

def generate_data():

    root_folder = os.path.split(os.path.abspath(__file__))[0]   
    _, data_generator, _ = GetDataGeneratorAndTrainer()
    _step = g_step
    while True:
        load_succ = LoadModel(_step)
        if load_succ:
            break

    while True:
        ob, ac, atarg, tdlamret, seg = data_generator.get_one_step_data()

        data_folder_path = '{}/../distribute_collected_train_data/{}'.format(root_folder, _step)
        if os.path.exists(data_folder_path):
            pass
        else:
            os.mkdir(data_folder_path)

        data_file_name = '{}/../distribute_collected_train_data/{}/seg_{}_{}.data'.format(root_folder, _step, g_worker_id, int(time.time()*100000))       
        dump_generated_data_2_file(data_file_name, seg)
        print('Timestep:{}, generated data:{}.'.format(_step, data_file_name))
        _step += 1

if __name__=='__main__':
    global g_step
    g_step = 0
    scene_id = 0
    if len(sys.argv) > 1:
        TIMESTEPS_PER_ACTOR_BATCH = int(sys.argv[1])
        
    if len(sys.argv) > 2:    
        g_worker_id = int(sys.argv[2])
        
    if len(sys.argv) > 3:    
        g_step = int(sys.argv[3])

    if len(sys.argv) > 4:
        scene_id = int(sys.argv[4])

    generate_data()