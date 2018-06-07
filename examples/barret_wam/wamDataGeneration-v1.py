import gym
import gym_gazebo
import time
import random
import numpy as np
import rospy
import roslaunch

from random import randint
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding


ep_returns = []
actions = []
observations = []
rewards = []
infos = []

def main():
    env = gym.make('GazeboWAMemptyEnv-v1')
    #env.seed()
    numItr = 100
    initStateSpace = "random"

    env.reset()
    print("Reset!")
    time.sleep(10)
    while len(actions) < numItr:
        obs = env.reset()
        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)

    fileName = "data_wam"

    fileName += "_" + initStateSpace

    fileName += "_" + str(numItr)

    fileName += ".npz"
    
    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)



def actionMapping(env, ac, diff):

    action = [0, 0, 0]

    for joint in range(len(ac)):
        if ac[joint]==0: 
            #action[joint] = -max(random.uniform((diff[joint]/9), (diff[joint]/3)), 0)
            action[joint] = -diff[joint]/3
        elif ac[joint]==1:
            action[joint] = diff[joint]/3

    return np.array(action)


def goToGoal(env, lastObs):

    difference = np.zeros(env.lenGoal)
    goalPosition = env.goalJS
    goalPositionConc = [goalPosition[0], goalPosition[1], goalPosition[3]] 



    #ep_return = 0
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    #episodeRews = []    
    obsData = env.lastObservationJS



    difference = np.array(obsData) - np.array(goalPositionConc)
    obsData = lastObs

    for _ in range(env._max_episode_steps):

        
        actionPreferance = np.zeros(len(difference))
        for joint in range(len(difference)):
            if difference[joint] > 0: #backward
                actionPreferance[joint] = 0
            elif difference[joint] < 0:
                actionPreferance[joint] = 1
            else: None   

        action = actionMapping(env, actionPreferance, np.absolute(difference))

        obsDataNew, reward, done, info = env.step(action)

        #obsData = env.lastObservationJS
        difference = np.array(env.lastObservationJS) - np.array(goalPositionConc)
        
        #ep_return += reward
        episodeObs.append(obsData)
        #episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)

        obsData = obsDataNew

        # print (" return: ", reward)
        # print ("actions : ", action)
        # print ("observation : ", obsData)
        #if done: break

    episodeObs.append(obsData)


    #ep_returns.append(ep_return)
    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    #rewards.append(np.array(episodeRews))

    

if __name__ == "__main__":
    main()
