# An introduction to Reinforcement Learning (specifically: Q-Learning) using OpenAI's Gym
Notes and code following the introductory reinforcement learning tutorial from @Sentdex: https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

# Learning reinforcement learning
Using OpenAI's gym for an introductory exploration of RL and specifically Q-Learning.
https://gym.openai.com/
Also: https://github.com/openai/gym

# Notes
Also here: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/   

For an environment:
* States: Observations on the state of the agent
* Actions: Actions that the agent performs
* Goals: What you want the agent to achieve  

For [OpenAI's gym MountainCar-v0 environment](https://github.com/openai/gym/wiki/MountainCar-v0), humans may need to know what action means what but we'll not tell the system - it needs to figure this out by itself. This is the "action space" for MountainCar-v0:
* 0 - push left
* 1 - no push
* 2 - push right
* the 'goal' is to get to the flag or 0.5

_Need to learn how the environments are defined_  

The "observation space" is found through ```step()``` or ```reset()``` - when you do ```env.reset()``` it returns an initial state, then you can step through various actions and see how the state changes.

For every action in every state the system assigns a *Q* value.

"observations" for MountainCar-v0 are of the type ```[position, velocity]```

At time t (or at step t), q-value is a function of s (state at time t) and a (action or actions at time t):  
q_new(time = t) = (1-alpha)*q_old + alpha*q_learned
q_learned = reward(time = t) + (discount_factor * estimated_optimal_future_value)
estimated_optimal_future_value = max (q_new(time = t+1)) for all actions at t=1


