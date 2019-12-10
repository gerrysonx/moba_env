# coding=utf-8
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from subprocess import call


if __name__=='__main__':
    root_folder = os.path.split(os.path.abspath(__file__))[0]

    horizon_total = 1024 * 512
    horizon_per_worker = 8192 * 4
    worker_count = horizon_total / horizon_per_worker

    train_full_path = '{}/ppo_distribute_train.py'.format(root_folder)
    generate_full_path = '{}/ppo_distribute_generate_data.py'.format(root_folder)

    call(["python", train_full_path, '{}'.format(worker_count)])
    for i in range(worker_count):
        call(["python", generate_full_path, '{}'.format(horizon_per_worker)])