import gym
import numpy as np

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

env = gym.make("MountainCar-v0")
env.reset()

# while this env provides a min/max - 
# IRL we may need to run observations for min/max may not be available at the start

# print the max position and max velocity possible
print("env.observation_space.high: [position, velocity]: ", env.observation_space.high)
# print the min position and min velocity possible
print("env.observation_space.low:  [position, velocity]: ", env.observation_space.low)

# break the position range into 20 buckets and the velocity range into 20 as well
DISCREET_OS_SIZE = [20,20]
# size of one bucket of [position, velocity]
discreet_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCREET_OS_SIZE
# create a 3D Q-table 
# [state (a combination of position and velocity), each action at the state]
    ''' so:
    pos 0, vel 0, act 0
    pos 0, vel 0, act 1
    pos 0, vel 0, act 2
    pos 1, vel 0, act 0
    pos 1, vel 0, act 1
    pos 1, vel 0, act 2
    pos 1, vel 1, act 0
    pos 1, vel 1, act 1
    pos 1, vel 1, act 2
    ... and so on...'''
q_table = np.random.uniform(low = -2, high = 0, size = (DISCREET_OS_SIZE + [env.action_space.n]))


done = False

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

discrete_state = get_discrete_state(env.reset())

while not done:
	action = 2 # go right by default
	new_state, reward, done, _ = env.step(action)
	print("reward: ", reward,", new_state: ",new_state)
	env.render()


