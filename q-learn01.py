import gym
import numpy as np

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

env = gym.make("MountainCar-v0")

# env.reset() # no longer needed, we'll use the get_discrete_state method to get the right bucket

# while this env provides a min/max - 
# IRL we may need to run observations for min/max may not be available at the start

# print the max position and max velocity possible
print("env.observation_space.high: [position, velocity]: ", env.observation_space.high)
# print the min position and min velocity possible
print("env.observation_space.low:  [position, velocity]: ", env.observation_space.low)

# break the position range into 20 buckets and the velocity range into 20 as well
DISCREET_OBSERVATION_SPACE_BUCKETS = [20,20]
# size of one bucket of [position, velocity]
discree_observation_space_window_size = (env.observation_space.high - env.observation_space.low)/DISCREET_OBSERVATION_SPACE_BUCKETS
# create a 3D Q table 
'''[state (a combination of position and velocity), each action at the state]
    so:
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
q_table = np.random.uniform(low = -2, high = 0, size = (DISCREET_OBSERVATION_SPACE_BUCKETS + [env.action_space.n]))

done = False

# find out which 'bucket' the current state falls in
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discree_observation_space_window_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

# one iteration in our q_learning cycle
def run_single_episode():
    discrete_state = get_discrete_state(env.reset()) # 'bucket' for initial state
    while not done:
    	#action = 2 # go right by default
        action = np.argmax(q_table[discrete_state])
    	# get the updated state, reward and if the sim was "done" from env.step
        new_state, reward, done, _ = env.step(action)
        # new state? we need to get it's bucket so:
        new_discreet_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discreet_state])
            current_q = q_table[discrete_state + (action,)]
            # calculate the new q_value for...
            # the particular action we just took (not the new discreet state!!!) in the specific bucket
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # update the q_value for the action we just took in the q table
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position: # our state is [position, velocity] -- we want to check if the car reached the flag...
            q_table[discrete_state+(action,)] = 0 # the 'done' value is 0 - our system needs to figure out the policy to get to 0

        discrete_state = new_discreet_state # onwards to the new state...

for episode in range(EPISODES):
    run_single_episode()


