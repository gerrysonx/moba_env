import gym
import time
import subprocess
import os
import json
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class MobaMultiPlayerEnv(gym.Env):
	metadata = {'render.modes':['human']}
	def restart_proc(self):
	#	print('moba_env restart_proc is called.')
		self.done = False
		self.hero_count = 2
		self.state = np.zeros((self.hero_count, 12))
		self.reward = 0
		self.info = None
		self.self_0_health = 0
		self.self_1_health = 0
		self.oppo_health = 0
		self.step_idx = 0	
		self.full_self_0_health = 0
		self.full_self_1_health = 0
		self.full_oppo_health = 0
				
		pass

	def __init__(self):
		# Create process, and communicate with std
		is_train = True
		root_folder = os.path.split(os.path.abspath(__file__))[0]
		try:
			# Read control file			
			ctrl_file_path = '{}/../../../ctrl.txt'.format(root_folder)
			file_handle = open(ctrl_file_path, 'r')
			ctrl_str = file_handle.read()
			file_handle.close()		
			if int(ctrl_str) == 1:
				is_train = True
			else:
				is_train = False

		except:
			pass		
		
		manual_str = '-manual_enemy=false'
		if not is_train:
			manual_str = '-manual_enemy=true'
		my_env = os.environ.copy()
		my_env['TF_CPP_MIN_LOG_LEVEL'] = '3'
		gamecore_file_path = '{}/../../../gamecore/gamecore'.format(root_folder)
		self.proc = subprocess.Popen([gamecore_file_path, 
								'-render=true', '-gym_mode=true', '-debug_log=true', '-slow_tick=true', '-multi_player=true', '-scene=0', manual_str],
								stdin=subprocess.PIPE,
								stdout=subprocess.PIPE,
								stderr=subprocess.PIPE, env=my_env)



		self.restart_proc()
		print('moba_env initialized.')
		pass

	def step(self, total_actions):
		#time.sleep(1)
		# action is 4-dimension vector
		self.proc.stdin.write(b'2\n')
		self.proc.stdin.flush() 
		self.step_idx += 1
		for hero_idx in range(self.hero_count):
			action = total_actions[hero_idx]
			action_encode = 0
			action_encode = (action[0] << 12)

			move_dir_encode = 0
			if action[1] != -1:
				move_dir_encode = (action[1] << 8)	

			skill_dir_encode = 0
			if action[2] != -1:
			    skill_dir_encode = (action[2] << 4)	

			encoded_action_val = (self.step_idx << 16) + action_encode + move_dir_encode + skill_dir_encode
			self.proc.stdin.write('{}\n'.format(encoded_action_val).encode())
			self.proc.stdin.flush()

		# Wait for response.
		while True:
			json_str = self.proc.stdout.readline()
			#print('When step, recv game process response {}'.format(json_str))
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
						self.reward = 1#jobj['SelfWin']
						hp_reward = (jobj['SelfHero0Health'] + jobj['SelfHero1Health']) / (jobj['SelfHero0HealthFull'] + jobj['SelfHero1HealthFull'])
						self.reward += hp_reward
				else:
					harm_reward = 0.002
					self.reward = 0
					if self.self_0_health  > jobj['SelfHero0Health'] or self.self_1_health  > jobj['SelfHero1Health']:
						self.reward -= harm_reward
					if self.oppo_health  > jobj['OppoHeroHealth']:
						self.reward += harm_reward

					self.oppo_health = jobj['OppoHeroHealth']
					self.self_0_health = jobj['SelfHero0Health']
					self.self_1_health = jobj['SelfHero1Health']

					self.done = False
				break
			except:
				print('Parsing json failed, terminate this game.')
				self.done = True
				self.reward = 0
				return self.state, self.reward, self.done, self.step_idx	

		norm_base = 1000.0	
		# Player 1 perspective
		self.state[0][0] = jobj['SelfHero0PosX'] / norm_base - 0.5
		self.state[0][1] = jobj['SelfHero0PosY'] / norm_base - 0.5
		self.state[0][2] = jobj['SelfHero0Health'] / jobj['SelfHero0HealthFull'] - 0.5

		self.state[0][3] = jobj['SelfHero1PosX'] / norm_base - 0.5
		self.state[0][4] = jobj['SelfHero1PosY'] / norm_base - 0.5
		self.state[0][5] = jobj['SelfHero1Health'] / jobj['SelfHero1HealthFull'] - 0.5

		self.state[0][6] = jobj['OppoHeroPosX'] / norm_base - 0.5
		self.state[0][7] = jobj['OppoHeroPosY'] / norm_base - 0.5
		self.state[0][8] = jobj['OppoHeroHealth'] / jobj['OppoHeroHealthFull'] - 0.5

		self.state[0][9] = 0#jobj['SlowBuffState']
		self.state[0][10] = jobj['SlowBuffRemainTime']
		self.state[0][11] = 0

		# Player 2 perspective
		self.state[1][0] = jobj['SelfHero1PosX'] / norm_base - 0.5
		self.state[1][1] = jobj['SelfHero1PosY'] / norm_base - 0.5
		self.state[1][2] = jobj['SelfHero1Health'] / jobj['SelfHero1HealthFull'] - 0.5

		self.state[1][3] = jobj['SelfHero0PosX'] / norm_base - 0.5
		self.state[1][4] = jobj['SelfHero0PosY'] / norm_base - 0.5
		self.state[1][5] = jobj['SelfHero0Health'] / jobj['SelfHero0HealthFull'] - 0.5

		self.state[1][6] = jobj['OppoHeroPosX'] / norm_base - 0.5
		self.state[1][7] = jobj['OppoHeroPosY'] / norm_base - 0.5
		self.state[1][8] = jobj['OppoHeroHealth'] / jobj['OppoHeroHealthFull'] - 0.5

		self.state[1][9] = 0#jobj['SlowBuffState']
		self.state[1][10] = jobj['SlowBuffRemainTime']
		self.state[1][11] = 1


		return self.state, self.reward, self.done, self.step_idx


	def reset(self):
		self.restart_proc()
		# To avoid deadlocks: careful to: add \n to output, flush output, use
		# readline() rather than read()
		#self.proc.stdout.readline()
		self.proc.stdin.write(b'36864\n')
		self.proc.stdin.flush()
#		print('Send reset signal to game process.')
		while True:
			json_str = self.proc.stdout.readline()
	#		print('When reset, recv game process response {}'.format(json_str))
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

				self.full_self_0_health = jobj['SelfHero0HealthFull']
				self.self_0_health = jobj['SelfHero0Health']
				self.full_self_1_health = jobj['SelfHero1HealthFull']
				self.self_1_health = jobj['SelfHero1Health']
				self.full_oppo_health = jobj['OppoHeroHealthFull']
				self.oppo_health = jobj['OppoHeroHealth']
				break

			except:
				print('When resetting env, parsing json failed.')
				continue
	
		return self.state


	def render(self, mode='human'):
		pass

	def close(self):
		self.proc.stdin.close()
		self.proc.terminate()
		self.proc.wait(timeout=0.2)			
		pass
