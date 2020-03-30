import gym
import time
import subprocess
import os
import json
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

ONE_HERO_FEATURE_SIZE = 3
BATTLE_FIELD_SIZE = 1000.0

class MobaMultiPlayerEnv(gym.Env):
	metadata = {'render.modes':['human']}
	def restart_proc(self):
	#	print('moba_env restart_proc is called.')
		self.done = False
		self.hero_count = 0
		self.oppo_hero_count = 0

		self.state = None
		self.reward = 0
		self.info = None
		self.last_state = None
		self.step_idx = 0				
		pass

	def __init__(self):
		# Create process, and communicate with std
		is_train = True
		root_folder = os.path.split(os.path.abspath(__file__))[0]

		scene_id = 0
		try:
			# Read control file			
			ctrl_file_path = '{}/../../../ctrl.txt'.format(root_folder)
			file_handle = open(ctrl_file_path, 'r')
			ctrl_str = file_handle.read()
			segs = ctrl_str.split(' ')
			file_handle.close()		
			if int(segs[0]) == 1:
				is_train = True
			else:
				is_train = False

			scene_id = int(segs[1])

		except:
			pass		
		
		manual_str = '-manual_enemy=false'
		if not is_train:
			manual_str = '-manual_enemy=true'
		my_env = os.environ.copy()
		my_env['TF_CPP_MIN_LOG_LEVEL'] = '3'
		gamecore_file_path = '{}/../../../gamecore/gamecore'.format(root_folder)
		self.proc = subprocess.Popen([gamecore_file_path, 
								'-render=true', '-gym_mode=true', '-debug_log=true', '-slow_tick=false', 
								'-multi_player=true', '-scene={}'.format(scene_id), manual_str],
								stdin=subprocess.PIPE,
								stdout=subprocess.PIPE,
								stderr=subprocess.PIPE, env=my_env)

		# We shall parse the scene config file to get the input space and action space


		self.observation_space = spaces.Box(low=-1, high=1, shape=(100, 3))

        # Action space omits the Tackle/Catch actions, which are useful on defense
		self.action_space = spaces.Tuple((spaces.Discrete(3),))				

		self.restart_proc()
		print('moba_env initialized.')
		pass

	def fill_state(self, state, json_data):
		# View from the point view of each self
		self_hero_count = int(json_data['SelfHeroCount'])
		
		for hero_idx in range(self_hero_count):		
			feature_idx = 0	
			state[hero_idx][feature_idx] = (float(json_data['SelfHeroPosX'][hero_idx]) / BATTLE_FIELD_SIZE) - 0.5
			feature_idx += 1
			state[hero_idx][feature_idx] = (float(json_data['SelfHeroPosY'][hero_idx]) / BATTLE_FIELD_SIZE) - 0.5
			feature_idx += 1
			state[hero_idx][feature_idx] = (float(json_data['SelfHeroHealth'][hero_idx]) / float(json_data['SelfHeroHealthFull'][hero_idx])) - 0.5
			feature_idx += 1
			for _id_0 in range(self_hero_count):
				if _id_0 == hero_idx:
					continue
				state[hero_idx][feature_idx] = (float(json_data['SelfHeroPosX'][_id_0]) / BATTLE_FIELD_SIZE) - 0.5
				feature_idx += 1
				state[hero_idx][feature_idx] = (float(json_data['SelfHeroPosY'][_id_0]) / BATTLE_FIELD_SIZE) - 0.5
				feature_idx += 1
				state[hero_idx][feature_idx] = (float(json_data['SelfHeroHealth'][_id_0]) / float(json_data['SelfHeroHealthFull'][_id_0])) - 0.5
				feature_idx += 1

			oppo_hero_count = int(json_data['OppoHeroCount'])
			for _id_1 in range(oppo_hero_count):			
				state[hero_idx][feature_idx] = (float(json_data['OppoHeroPosX'][_id_1]) / BATTLE_FIELD_SIZE) - 0.5
				feature_idx += 1
				state[hero_idx][feature_idx] = (float(json_data['OppoHeroPosY'][_id_1]) / BATTLE_FIELD_SIZE) - 0.5
				feature_idx += 1
				state[hero_idx][feature_idx] = (float(json_data['OppoHeroHealth'][_id_1]) / float(json_data['OppoHeroHealthFull'][_id_1])) - 0.5
				feature_idx += 1

			pass

		pass

	def get_hp_remain_reward(self):
		total_self_hero_hp = 0
		for hero_idx in range(self.hero_count):
			total_self_hero_hp += self.state[hero_idx][2]

		return total_self_hero_hp

	def get_harm_reward(self):
		harm_reward = 0.002
		total_harm_reward = 0
		for hero_idx in range(self.hero_count):
			if self.state[hero_idx][2] < self.last_state[hero_idx][2]:
				total_harm_reward -= harm_reward

		start_feature_idx = self.hero_count * ONE_HERO_FEATURE_SIZE
		for hero_idx in range(self.oppo_hero_count):
			if self.state[0][start_feature_idx + hero_idx * ONE_HERO_FEATURE_SIZE + 2] < self.last_state[0][start_feature_idx + hero_idx * ONE_HERO_FEATURE_SIZE + 2]:
				total_harm_reward += harm_reward


		return total_harm_reward

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
		# Parse the state
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
				self.last_state = self.state	
				self.fill_state(self.state, jobj)

				if jobj['SelfWin'] != 0:
					self.done = True
					if 2 == jobj['SelfWin']:
						self.reward = 0
					elif -1 == jobj['SelfWin']:
						self.reward = -1
					else:
						self.reward = 1#jobj['SelfWin']

					# Add remain hp as reward
					hp_reward = self.get_hp_remain_reward()
					self.reward += hp_reward
				else:					
					self.reward = 0
					harm_reward = self.get_harm_reward()

					self.reward += harm_reward

					self.done = False
				

				break
			except:
				print('Parsing json failed, terminate this game.')
				self.done = True
				self.reward = 0
				return self.state, self.reward, self.done, self.step_idx

		return self.state, self.reward, self.done, self.step_idx


	def reset(self):
		self.restart_proc()
		# To avoid deadlocks: careful to: add \n to output, flush output, use
		# readline() rather than read()
		#self.proc.stdout.readline()
		self.proc.stdin.write(b'36864\n')
		self.proc.stdin.flush()

		while True:
			json_str = self.proc.stdout.readline()

			if json_str == None or len(json_str) == 0:
				raise Exception('When resetting env, json_str == None or len(json_str) == 0')

			try:
				str_json = json_str.decode("utf-8")
				parts = str_json.split('@')
				if int(parts[0]) != self.step_idx:
					print('Reset::We expect:{}, while getting {}'.format(self.step_idx, int(parts[0])))
					continue

				jobj = json.loads(parts[1])	

				self.hero_count = int(jobj['SelfHeroCount'])
				self.oppo_hero_count = int(jobj['OppoHeroCount'])

				state_feature_count = (self.hero_count + self.oppo_hero_count) * ONE_HERO_FEATURE_SIZE

				self.state = np.zeros((self.hero_count, state_feature_count))
				
				self.fill_state(self.state, jobj)				
				self.last_state = self.state
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
