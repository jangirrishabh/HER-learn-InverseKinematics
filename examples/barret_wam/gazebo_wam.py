import gym
import gym_gazebo
import time
import random
import numpy as np

from random import randint
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics


def main():
    env = gym.make('GazeboWAMemptyEnv-v0')
    env.minDisplacement = 0.09
    env.baseFrame = 'iri_wam_link_base'
    env.waitTime = 15 #time to wait if link gets stuck in seconds
    env.seed()

    env.reset()
    print("Reset!")
    time.sleep(10)
    for i in range(100):
        obs = env.reset()
        print("Reset!")
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




# def getInverseKinematics(self, goalPose): #get joint angles for reaching the goal position
#     goalPoseStamped = goalPose
#     rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_ik')
#     try:
#         getIK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_ik', QueryInverseKinematics)
#         jointPositionsReturned = getIK(goalPoseStamped)
#         return jointPositionsReturned.position
#     except (rospy.ServiceException) as e:
#         print ("Service call failed: %s"%e)

def getForwardKinematics(env, goalPosition): #get catesian coordinates for joint Positions
    rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
    try:
        getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
        jointPoseReturned = getFK(goalPosition)
        return jointPoseReturned
    except (rospy.ServiceException) as e:
        print ("Service call failed: %s"%e)



def actionMapping(ac):
    trueAction = ac[0] + ac[1]*7
    return trueAction


def goToGoal(env, lastObs):
    goalPosition = lastObs[np.shape(env.high)[0]:]
    for joint in range(7):
        difference = lastObs[:np.shape(env.high)[0]] - goalPosition
        start_time = time.time()
        while np.absolute(difference[joint]) > env.minDisplacement:
            #print ("Difference in goal for joint : ", joint, " = ", difference[joint] , " and current is : ", lastObs[joint], " and desired ", goalPosition[joint] )
            if difference[joint] > 0: #take negative action, backward
                action = actionMapping([joint, 1])  
            elif difference[joint] < 0: #take positive action, forward
                action = actionMapping([joint, 0])
            else: #take no action
                None
            obsData, reward, done, info = env.step(action)
            if reward > 0.001:
                print ("Reward received :", reward)
            lastObs = obsData[:np.shape(env.high)[0]]
            difference = lastObs - goalPosition
            elapsed_time = time.time() - start_time
            if elapsed_time > env.waitTime: 
                print("exiting from while loop")
                break

        if np.absolute(difference[joint]) <= env.minDisplacement:
            print ("Goal position reached for joint number ", joint)
        else:
            print("exiting due to bad goal point")
            break

        # if done==True:
        #     break



if __name__ == "__main__":
    main()
