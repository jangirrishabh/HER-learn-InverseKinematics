import gym
import gym_gazebo
env = gym.make('GazeboWAMemptyEnv-v0')
env.reset()
import time
# Check the env limits:
print(env.action_space) # Box(3,)
print(env.observation_space) # Box(9,)
from random import randint

for i in range(100):
    print ("about to")
    env.reset()
    print("Reset!")
    for x in range(200):
        # env.render()
        action = randint(0, 9)
        print("action = ", action)
        observation, reward, done, info = env.step(action) # take a random action
        #print("reward: ", reward, " observation: ", observation)
        if done: break
