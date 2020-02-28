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
import shutil
from ppo_lstm import LoadModel
from ppo_lstm import GetDataGeneratorAndTrainer

g_step = 0
g_worker_id = 0

def dump_generated_data_2_file(file_name, seg):
    with open(file_name, 'wb') as file_handle:
        pickle.dump(seg, file_handle)

    pass

def delete_outdated_folders(step):
    # Delete folders within range [0, step)
    root_folder = os.path.split(os.path.abspath(__file__))[0]   
    if step < 1:
        return
    for idx in range(step):
        try:
            data_folder_path = '{}/../distribute_collected_train_data/{}'.format(root_folder, idx)
            shutil.rmtree(data_folder_path)
        except:
            pass
    pass

def generate_data(scene_id):

    root_folder = os.path.split(os.path.abspath(__file__))[0]   
    _, data_generator, session = GetDataGeneratorAndTrainer(scene_id)
    _step = g_step    

    while True:
        LoadModel(session, _step)
        seg = data_generator.get_one_step_data()

        data_folder_path = '{}/../distribute_collected_train_data/{}'.format(root_folder, _step)
        while True:
            if os.path.exists(data_folder_path):
                break
            else:
                try:
                    os.mkdir(data_folder_path)
                    # Need to delete all previous folders
                    delete_outdated_folders(_step)
                    break
                except:
                    print('mkdir {} failed, but we caught the exception.'.format(data_folder_path))
                    continue

        data_file_name = '{}/../distribute_collected_train_data/{}/seg_{}_{}.data'.format(root_folder, _step, g_worker_id, int(time.time()*100000))       
        dump_generated_data_2_file(data_file_name, seg)
        print('Timestep:{}, generated data:{}.'.format(_step, data_file_name))
        _step += 1

if __name__=='__main__':
    g_step = 0
    scene_id = 10
    if len(sys.argv) > 1:
        TIMESTEPS_PER_ACTOR_BATCH = int(sys.argv[1])
        
    if len(sys.argv) > 2:    
        g_worker_id = int(sys.argv[2])
        
    if len(sys.argv) > 3:    
        g_step = int(sys.argv[3])

    if len(sys.argv) > 4:
        scene_id = int(sys.argv[4])

    generate_data(scene_id)