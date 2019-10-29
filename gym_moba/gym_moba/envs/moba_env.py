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
		self.self_health = 0
		self.oppo_health = 0
		self.step_idx = 0	
		self.full_self_health = 0
		self.full_oppo_health = 0			
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
		self.step_idx += 1
		encoded_action_val = (self.step_idx << 4) + action
		self.proc.stdin.write('{}\n'.format(encoded_action_val).encode())
		self.proc.stdin.flush()

		while True:
			json_str = self.proc.stdout.readline()
			if json_str == None or len(json_str) == 0:
				print('json_str == None or len(json_str) == 0')
				self.done = True
				self.reward = 0
				return self.state, self.reward, self.done, self.info

			try:
				str_json = json_str.decode("utf-8")
				parts = str_json.split('@')
				if int(parts[0]) != self.step_idx:
					print('Step::We expect:{}, while getting {}'.format(self.step_idx, int(parts[0])))
					continue
				jobj = json.loads(parts[1])	
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
					harm_reward = 0.2
					self.reward = 0
					if self.self_health  > jobj['SelfHeroHealth']:
						self.reward -= harm_reward
					if self.oppo_health  > jobj['OppoHeroHealth']:
						self.reward += harm_reward

					self.oppo_health = jobj['OppoHeroHealth']
					self.self_health = jobj['SelfHeroHealth']

					self.done = False
				break
			except:
				print('Parsing json failed, terminate this game.')
				self.done = True
				self.reward = 0
				return self.state, self.reward, self.done, self.info	

		norm_base = 1000.0	
		self.state[0] = jobj['SelfHeroPosX'] / norm_base
		self.state[1] = jobj['SelfHeroPosY'] / norm_base
		self.state[2] = jobj['SelfHeroHealth'] / self.full_self_health 

		self.state[3] = jobj['OppoHeroPosX'] / norm_base
		self.state[4] = jobj['OppoHeroPosY'] / norm_base
		self.state[5] = jobj['OppoHeroHealth'] / self.full_oppo_health 		
		return self.state, self.reward, self.done, self.info


	def reset(self):
		self.restart_proc()
		# To avoid deadlocks: careful to: add \n to output, flush output, use
		# readline() rather than read()
		#self.proc.stdout.readline()
		self.proc.stdin.write(b'9\n')
		self.proc.stdin.flush()
		while True:
			json_str = self.proc.stdout.readline()
			if json_str == None or len(json_str) == 0:
				print('When resetting env, json_str == None or len(json_str) == 0')

				return self.state

			try:
				str_json = json_str.decode("utf-8")
				parts = str_json.split('@')
				if int(parts[0]) != self.step_idx:
					print('Reset::We expect:{}, while getting {}'.format(self.step_idx, int(parts[0])))
					continue
				jobj = json.loads(parts[1])	

				self.full_self_health = jobj['SelfHeroHealth']
				self.self_health = self.full_self_health
				self.full_oppo_health = jobj['OppoHeroHealth']
				self.oppo_health = self.full_oppo_health
				break
			except:
				print('When resetting env, parsing json failed.')
	
		return self.state


	def render(self, mode='human'):
		pass

	def close(self):
		self.proc.stdin.close()
		self.proc.terminate()
		self.proc.wait(timeout=0.2)			
		pass
