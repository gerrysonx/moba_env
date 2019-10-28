import gym
import time
import subprocess
import os
import json
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class MobaEnv(gym.Env):
	metadata = {'render.modes':['human']}
	def restart_proc(self):
	#	print('moba_env restart_proc is called.')
		self.done = False
		self.state = np.zeros((6,))
		self.reward = 0
		self.info = None
				
		pass

	def __init__(self):
		# Create process, and communicate with std
		#root_folder = os.path.split(os.path.abspath(__name__))[0]
		self.proc = subprocess.Popen(['/home/gerrysun/work/ml-prjs/go-lang/moba/gamecore/gamecore', '-render=true'],
								stdin=subprocess.PIPE,
								stdout=subprocess.PIPE,
								stderr=subprocess.PIPE)



		self.restart_proc()
		print('moba_env initialized.')
		pass

	def step(self, action):
		#time.sleep(1)
		self.proc.stdin.write('{}\n'.format(action).encode())
		self.proc.stdin.flush()
		json_str = self.proc.stdout.readline()
		if json_str == None or len(json_str) == 0:
			print('json_str == None or len(json_str) == 0')
			self.done = True
			self.reward = 0
			return self.state, self.reward, self.done, self.info

		try:
			jobj = json.loads(json_str.decode("utf-8"))
		#	print('moba_env v2 step called with action:{}, result:{}'.format(action, jobj))
			if jobj['SelfWin'] != 0:
				self.done = True
				if 2 == jobj['SelfWin']:
					self.reward = 0
				elif -1 == jobj['SelfWin']:
					self.reward = -1
				else:
					self.reward = jobj['SelfWin']
			else:
				self.reward = 0
				self.done = False
		except:
			print('Parsing json failed, terminate this game.')
			self.done = True
			self.reward = 0
			return self.state, self.reward, self.done, self.info			
		norm_base = 1000.0	
		self.state[0] = jobj['SelfHeroPosX'] / norm_base
		self.state[1] = jobj['SelfHeroPosY'] / norm_base
		self.state[2] = jobj['SelfHeroHealth'] / norm_base

		self.state[3] = jobj['OppoHeroPosX'] / norm_base
		self.state[4] = jobj['OppoHeroPosY'] / norm_base
		self.state[5] = jobj['OppoHeroHealth'] / norm_base		
		return self.state, self.reward, self.done, self.info


	def reset(self):
		self.restart_proc()
		# To avoid deadlocks: careful to: add \n to output, flush output, use
		# readline() rather than read()
		self.proc.stdin.write(b'9\n')
		self.proc.stdin.flush()
		self.proc.stdout.readline()



	
		return self.state


	def render(self, mode='human'):
		pass

	def close(self):
		self.proc.stdin.close()
		self.proc.terminate()
		self.proc.wait(timeout=0.2)			
		pass
