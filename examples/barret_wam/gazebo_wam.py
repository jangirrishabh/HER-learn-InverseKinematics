import gym
import gym_gazebo
env = gym.make('GazeboWAMemptyEnv-v0')
import time
from random import randint

#env.reset()
env.reset()
print("Reset!")

# Check the env limits:
#print(env.action_space) # Box(3,)
#print(env.observation_space) # Box(9,)

for i in range(100):
    env.reset()
    print("Reset!")
    for x in range(1000):
        # env.render()
        action = [randint(0, 6), randint(0, 1)]
        #print("action = ", action)
        observation, reward, done, info = env.step(action) # take a random action
        #time.sleep(2)
        #print("reward: ", reward, " observation: ", observation)
        if done: break
