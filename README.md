# An introduction to Reinforcement Learning (specifically: Q-Learning) using OpenAI's Gym
Notes and code following the introductory reinforcement learning tutorial from @Sentdex: https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

# Learning reinforcement learning
Using OpenAI's gym for an introductory exploration of RL and specifically Q-Learning.
https://gym.openai.com/
Also: https://github.com/openai/gym

# Notes
* Also here: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
* At time t (or at step t), q-value is a function of s (state at time t) and a (action or actions at time t):
* q_new(time = t) = (1-alpha)*q_old + alpha*q_learned
* q_learned = reward(time = t) + (discount_factor * estimated_optimal_future_value)
* estimated_optimal_future_value = max (q_new(time = t+1)) for all actions at t=1


