import sys, os

import baselines.run as run
os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,tensorboard'
 
if __name__ == '__main__':
    
    run.main(['main.py', '--alg=ppo_moba', '--env=gym_moba:moba-multiplayer-v0', 
    '--network=multi_unit_mlp', '--num_timesteps=2e7', '--scene_id=10', '--is_train'])
    '''
    run.main(['main.py', '--alg=ppo2', '--env=Breakout-v0', 
    '--network=cnn', '--num_timesteps=2e7', '--scene_id=10', '--is_train'])
    '''

