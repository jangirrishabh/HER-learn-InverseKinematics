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
    while len(actions) < 100:
        obs = env.reset()
        print("Reset!")
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)

    
    np.savez_compressed('data_wam.npz', acs=actions, obs=observations, ep_rets=ep_returns, rews=rewards)
 

def getInverseKinematics(env, goalPose): #get joint angles for reaching the goal position
    tempPose = Pose()
    tempPose.position.x = goalPose[0]
    tempPose.position.y = goalPose[1]
    tempPose.position.z = goalPose[2]
    tempPose.orientation.x = goalPose[3]
    tempPose.orientation.y = goalPose[4]
    tempPose.orientation.z = goalPose[5]
    tempPose.orientation.w = goalPose[6]

    goalPoseStamped = PoseStamped()
    goalPoseStamped.header.frame_id = env.baseFrame
    goalPoseStamped.pose = tempPose
    rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_ik')
    try:
        getIK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_ik', QueryInverseKinematics)
        jointPositionsReturned = getIK(goalPoseStamped)
        return [jointPositionsReturned.joints.position[0], jointPositionsReturned.joints.position[1], jointPositionsReturned.joints.position[3]]
    except (rospy.ServiceException) as e:
        print ("Service call failed: %s"%e)



def actionMapping(env, ac, diff):

    action = [0, 0, 0]

    for joint in range(len(ac)):
        if ac[joint]==0: 
            action[joint] = -random.uniform((diff[joint]/5), (diff[joint]/3))
        elif ac[joint]==1:
            action[joint] = random.uniform((diff[joint]/5), (diff[joint]/3))

    return np.array(action)


def goToGoal(env, lastObs):
    done = False
    FAIL = False
    difference = np.zeros(env.lenGoal)

    goalPosition = getInverseKinematics(env, lastObs[(env.lenGoal*2):])
    goalPositionConc = goalPosition
    if goalPositionConc == None: 
        print ("Inverse kinematics failed ")
        FAIL = True

    
    
    reached = -1
    ep_return = 0
    episodeAcs = []
    episodeObs = []
    episodeRews = []
    beginningTime = time.time()
    
    elapsedTimes = [0, 0, 0]
    startTimes = [0, 0, 0]
    obsData = env.lastObservationFull



    if not FAIL: difference = np.array(lastObs[:env.lenGoal]) - np.array(goalPositionConc)
    #print (" the differences in our views ", (np.absolute(difference) > np.array([env.minDisplacement, env.minDisplacement, env.minDisplacement])))
    while (np.absolute(difference) > np.array([env.minDisplacement, env.minDisplacement, env.minDisplacement])).any() and (done == False) and FAIL == False:
       # print (" INSIDE ", (np.absolute(difference) > np.array([env.minDisplacement, env.minDisplacement, env.minDisplacement])))
        badDataFlag = False


        actionPreferance = np.zeros(len(difference))
        for joint in range(len(difference)):
            if difference[joint] > 0: #backward
                actionPreferance[joint] = 0
            elif difference[joint] < 0:
                actionPreferance[joint] = 1
            else: None   

        action = actionMapping(env, actionPreferance, np.absolute(difference))

        obsData, reward, done, badDataFlag, moved = env.step(action)

        obsDataAugmented = obsData[env.lenGoal:(env.lenGoal*3)]
        #print ("obs data STEP function ", obsDataAugmented)
        assert len(obsDataAugmented) == env.lenGoal*2

        if badDataFlag: 
            print ("starting again due to bad data ")
            break
        if moved:#(np.absolute(reward) > 0.001) and moved:
            #print ("Reward received :", reward)
            ep_return += reward
            episodeObs.append(obsDataAugmented)
            episodeRews.append(reward)
            episodeAcs.append(action)

        lastObs = obsData[:env.lenGoal]
        difference = np.array(lastObs) - np.array(goalPositionConc)
        elapsed_time = time.time() - beginningTime
        if elapsed_time > env.waitTime: 
            print("exiting from while loop due to TIME")
            break

    if done==True:
        ep_returns.append(ep_return)
        actions.append(np.array(episodeAcs))
        observations.append(np.array(episodeObs))
        rewards.append(np.array(episodeRews))
        print ("total episode return: ", ep_return)
        print ("total episode actions lenght: ", len(episodeAcs))
        print ("total episode observation length : ", len(episodeObs))
    else: 
        #print ("OUTSIDE ", (np.absolute(difference) > np.array([env.minDisplacement, env.minDisplacement, env.minDisplacement])))
        print ("done not true still out of the while loop ")



if __name__ == "__main__":
    main()
