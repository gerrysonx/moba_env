from gym.envs.registration import register

register(
	id='moba-v0',
	entry_point='gym_moba.envs:MobaEnv',
)

#register(
#	id='moba-extrahard-v0',
#	entry_point='gym_moba.envs:MobaExtraHardEnv',
#)
