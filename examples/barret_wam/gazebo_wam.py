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
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics


ep_returns = []
actions = []
observations = []
rewards = []

def main():
    env = gym.make('GazeboWAMemptyEnv-v0')
    #env.seed()
    

    env.reset()
    print("Reset!")
    time.sleep(10)
    while len(actions) < 200:
        obs = env.reset()
        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        #randomGoalPosition = env.getRandomGoal()
        #print("New Goal received!")
        goToGoal(env, obs)
        # action = randint(0, 13)
        # obsData, reward, done, info = env.step(action)
        #time.sleep(5)
        # for x in range(1000):
        #     # env.render()
        #     #action = [randint(0, 6), randint(0, 1)]
        #     #print("action = ", action)
        #     #observation, reward, done, info = env.step(action) # take a random action
        #     #time.sleep(2)
        #     #print("reward: ", reward, " observation: ", observation)

        #     randomGoalPosition = env.getRandomGoal()
        #     env.goToGoal(randomGoalPosition)
        #     if done: break

    np.savez_compressed('data_wam.npz', acs=actions, obs=observations, ep_rets=ep_returns, rews=rewards)
    #np.save('obs.npy', observations)
    #np.save('ep_rets.npy', ep_returns)
    #np.save('rews.npy', rewards)




def getInverseKinematics(env, goalPose): #get joint angles for reaching the goal position
    tempPose = Pose()
    #tempPose.header.frame_id = env.baseFrame
    tempPose.position.x = goalPose[0]
    tempPose.position.y = goalPose[1]
    tempPose.position.z = goalPose[2]
    tempPose.orientation.x = goalPose[3]
    tempPose.orientation.y = goalPose[4]
    tempPose.orientation.z = goalPose[5] 
    tempPose.orientation.w = goalPose[6]

    #print (tempPose)
    goalPoseStamped = PoseStamped()
    goalPoseStamped.header.frame_id = env.baseFrame
    goalPoseStamped.pose = tempPose
    rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_ik')
    try:
        getIK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_ik', QueryInverseKinematics)
        jointPositionsReturned = getIK(goalPoseStamped)
        #print ("Returned by IK " , jointPositionsReturned)
        return [jointPositionsReturned.joints.position[0], jointPositionsReturned.joints.position[1], jointPositionsReturned.joints.position[3]]
    except (rospy.ServiceException) as e:
        print ("Service call failed: %s"%e)



def actionMapping(env, ac):
    #trueAction = ac[0] + ac[1]*7

    action = [0, 0, 0]
    if ac[1]==1:
        action[ac[0]]=1
    elif ac[1]==0: #backward
         action[ac[0]]=-1

    return np.array(action)


def goToGoal(env, lastObs):
    done = False
    #goalPosition = getInverseKinematics(env, lastObs[(len(env.highConc)/2):])
    goalPosition = None
    
    goalPosition = getInverseKinematics(env, lastObs[env.lenGoal:])

    goalPositionConc = goalPosition
    
    reached = -1
    ep_return = 0
    episodeAcs = []
    episodeObs = []
    episodeRews = []
    #badDataFlag = False
    beginningTime = time.time()
    
    elapsedTimes = [0, 0, 0]

    while done == False:
        if goalPositionConc == None: 
            print ("Inverse kinematics failed ")
            break
        badDataFlag = False
        print ("resetting times ")
        startTimes = [0, 0, 0]
        for joint in range(3):
            obsData = env.lastObservation
            difference = np.array(lastObs[:env.lenGoal]) - np.array(goalPositionConc)
            start_time = time.time()
            startTimes[joint] = time.time()
            while (np.absolute(difference[joint]) > env.minDisplacement) and done == False:
                #print ("Difference in goal for joint : ", joint, " = ", difference[joint], done)
                if difference[joint] > 0: #take negative action, backward
                    action = actionMapping(env, [joint, 0])  
                elif difference[joint] < 0: #take positive action, forward
                    action = actionMapping(env, [joint, 1])
                else: #take no action
                    None
                obsData, reward, done, badDataFlag, moved = env.step(action)
                if badDataFlag: 
                    print ("starting again due to bad data ")
                    break
                if moved:#(np.absolute(reward) > 0.001) and moved:
                    #print ("Reward received :", reward)
                    ep_return += reward
                    episodeObs.append(obsData)
                    episodeRews.append(reward)
                    episodeAcs.append(action)
                lastObs = obsData[:env.lenGoal]
                difference = lastObs - goalPositionConc
                elapsed_time = time.time() - start_time
                if elapsed_time > env.waitTime: 
                    print("exiting from while loop")
                    break
            
            
            if badDataFlag: 
                print ("starting again due to bad data 2 ")
                break

            #elapsed_time = time.time() - beginningTime
            elapsedTimes[joint] = elapsedTimes[joint] + time.time() - startTimes[joint]
            if np.absolute(difference[joint]) < env.minDisplacement:
                reached = joint
                print ("Goal position reached for joint number ", joint)
                #startTimes[joint] = 0
            elif reached < 0 or (elapsedTimes > np.full(np.shape(elapsedTimes), (env.waitTime*2)) ).any() :
                print("Bad point, moving out of for loop")
                break
            else: 
                print("Moving to next joint ", "reached = ", reached, "elapsed time = ", elapsed_time, elapsedTimes, (elapsedTimes > np.full(np.shape(elapsedTimes), (env.waitTime*2)) ).any())

            if done==True:#and reached==6:
                ep_returns.append(ep_return)
                actions.append(np.array(episodeAcs))
                observations.append(np.array(episodeObs))
                rewards.append(np.array(episodeRews))
                print ("total episode return: ", ep_return)
                print ("total episode actions lenght: ", len(episodeAcs))
                print ("total episode observation length : ", len(episodeObs))
                break
            #else: reached = 100

        elapsed_time = time.time() - beginningTime
        if elapsed_time > env.waitTime: 
            print("exiting from the main while loop")
            break

        if badDataFlag: 
                print ("starting again due to bad data 3 ")
                break
        


if __name__ == "__main__":
    main()
