# An introduction to Reinforcement Learning (specifically: Q-Learning) using OpenAI's Gym
Notes and code following the introductory [reinforcement learning tutorial](https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7) from @Sentdex   
    
[Full text version on pythonprogramming.net](https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/)   

## Library references  
Using [OpenAI's gym](https://gym.openai.com/) for an introductory exploration of RL and specifically Q-Learning.   
Also: [gym github repo](https://github.com/openai/gym)    

## Notes   

Q-Learning is a "model-free" reinforcement learning algorithm.  

_Need to learn:  finite Markov decision process (FMDP)_  

For an environment:  
* States: Observations on the state of the agent  
* Actions: Actions that the agent performs  
* Goals: What you want the agent to achieve    

The "observation space" is found through ```step()``` or ```reset()``` - when you do ```env.reset()``` it returns an initial state, then you can step through various actions and see how the state changes.  

For every action in every state the system assigns a *Q* value. 
At a step:  
* The system looks at q-values for each of the actions that exist corresponding to the state
* Selects the action to perform next based on some rule (for e.g. perform the action with max q-value). 
* Based on the reward, q-values are updated for the state. 
* We only want to find out what "policy" (given a state, what action should the agent take) will take us to maximum reward (or goal).  

[That function](https://en.wikipedia.org/wiki/Q-learning#Algorithm) from [wikipedia](https://en.wikipedia.org/wiki/Q-learning):  
At *time* t (or at *step* t), q-value is a function of s (state at time t) and a (action or actions at time t):    
* ```q_new(time = t) = (1-alpha) * q_old + alpha * q_learned```   
* ```q_learned = reward(time = t) + (discount_factor * estimated_optimal_future_value)```   
* ```estimated_optimal_future_value = max (q_new(time = t+1)) for all actions at t=1```  
  
In [OpenAI's gym MountainCar-v0 environment](https://github.com/openai/gym/wiki/MountainCar-v0), humans may need to know what action means but the system or the algorithm doesn't care much - its focussed on figuring out the optimal "policy".   
This is the "action space" for MountainCar-v0:  
* 0 - push left  
* 1 - no push  
* 2 - push right  
* the 'goal' is to get to the flag or 0.5  
  
"Observations" for ```MountainCar-v0``` are of the type ```[position, velocity]``` but one should not care - IRL these may be anything. 

_Need to learn:  how the environments are defined_   
 


