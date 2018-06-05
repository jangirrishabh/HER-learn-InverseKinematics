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


"""Data generation for the case of a single block with wam pick and place"""

ep_returns = []
actions = []
observations = []
rewards = []
infos = []

def main():
    env = gym.make('GazeboWAMemptyEnv-v2')
    #env.seed()
    numItr = 100
    initStateSpace = "random"

    env.reset()
    print("Reset!")
    time.sleep(1)
    while len(actions) < numItr:
        obs = env.reset()
        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        objPosition = env.objInitialJS
        goToGoal(env, obs, objPosition)
        # if objPosition != None: 
        #     obs2 = goToObject(env, obs, objPosition)
        #     goToGoal(env, obs2)
        # else: 
        #     goToGoal(env, obs)
        #time.sleep(0)

    fileName = "data_wam_double"

    fileName += "_" + initStateSpace

    fileName += "_" + str(numItr)

    fileName += ".npz"
    
    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)



def actionMapping(env, ac, diff):

    action = [0, 0, 0, 0, 0]

    for joint in range(len(ac[:4])):
        if ac[joint]==0: 
            #action[joint] = -max(random.uniform((diff[joint]/9), (diff[joint]/3)), 0)
            action[joint] = -diff[joint]/3
        elif ac[joint]==1:
            #action[joint] = max(random.uniform((diff[joint]/9), (diff[joint]/3)), 0)
            action[joint] = diff[joint]/3

    action[4] = random.uniform(0.5, 1)
    return np.array(action)




def goToGoal(env, lastObs, objPosition):

    differenceObject = np.zeros(env.lenGoal)
    differenceGoal = np.zeros(env.lenGoal)
    goalPosition = env.goalJS
    goalPositionConc = [goalPosition[0], goalPosition[1], goalPosition[2], goalPosition[3]] 

    #ep_return = 0
    episodeAcs = []
    episodeObs = []
    episodeInfo = []
    #episodeRews = []    
    obsData = env.lastObservationJS
    objPositionConc = [objPosition[0], objPosition[1], objPosition[2], objPosition[3]]  #getting the initial object position as input
    differenceGoal = np.array(obsData) - np.array(goalPositionConc)
    differenceObject = np.array(obsData) - np.array(objPositionConc)
    obsData = lastObs


    #t = 0
    #while np.linalg.norm(difference) > 0.06:


    #while differenceObject.any()>env.distanceThresholdDemo:
    #t += 1

    for _ in range(env._max_episode_steps):
        
        actionPreferance = np.zeros(len(differenceObject))
        for joint in range(len(differenceObject)):
            if differenceObject[joint] > 0: #backward
                actionPreferance[joint] = 0
            elif differenceObject[joint] < 0:
                actionPreferance[joint] = 1
            else: None   

        action = actionMapping(env, actionPreferance, np.absolute(differenceObject))

        obsDataNew, reward, done, info = env.step(action)

        #obsData = env.lastObservationJS
        differenceObject = np.array(env.lastObservationJS) - np.array(objPositionConc)
        
        #ep_return += reward
        episodeObs.append(obsData)
        #episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)

        obsData = obsDataNew


    

    #print("Object reached, preparing to grasp")
    #time.sleep(0.05)
    # action = np.array([0, 0, 0, 0,-1]) #open
    # _, _, _, _ = env.step(action)
    episodeObs.append(obsData)
    action = np.array([0, 0, 0, 0, random.uniform(0.7, 1)])
    obsData, _, _, info = env.step(action)
    episodeAcs.append(action)
    episodeInfo.append(info)
    

    for _ in range(env._max_episode_steps):

        
        actionPreferance = np.zeros(len(differenceGoal))
        for joint in range(len(differenceGoal)):
            if differenceGoal[joint] > 0: #backward
                actionPreferance[joint] = 0
            elif differenceGoal[joint] < 0:
                actionPreferance[joint] = 1
            else: None   

        action = actionMapping(env, actionPreferance, np.absolute(differenceGoal))

        obsDataNew, reward, done, info = env.step(action)

        #obsData = env.lastObservationJS
        differenceGoal = np.array(env.lastObservationJS) - np.array(goalPositionConc)
        
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

    #time.sleep(1)
    action = np.array([0, 0, 0, 0, random.uniform(-1, -0.5)])
    obsData, _, _, info = env.step(action)
    episodeAcs.append(action)
    episodeInfo.append(info)
    episodeObs.append(obsData)


    for i in range(10): 
        action = np.array([0, -0.005, 0, 0, random.uniform(-1, -0.5)])
        obsData, _, _, info = env.step(action)
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsData)


    


    #ep_returns.append(ep_return)
    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)
    #rewards.append(np.array(episodeRews))

    

if __name__ == "__main__":
    main()
