import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()
print("env.observation_space.high: ", env.observation_space.high)
print("env.observation_space.low: ", env.observation_space.low)

DISCREET_OS_SIZE = [20,20]
discreet_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCREET_OS_SIZE

q_table = np.random.uniform(low = -2, high = 0, size = (DISCREET_OS_SIZE + [env.action_space.n]))

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

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


