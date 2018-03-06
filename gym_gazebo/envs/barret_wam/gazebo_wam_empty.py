import gym
import rospy
import roslaunch
import time
import numpy as np
import random

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding
from numpy import linalg as LA
from iri_common_drivers_msgs.srv import QueryInverseKinematics
from iri_common_drivers_msgs.srv import QueryForwardKinematics



class GazeboWAMemptyEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "iri_wam.launch")
        #self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.publishers = ['pub1', 'pub2', 'pub3', 'pub4', 'pub5', 'pub6', 'pub7']

        self.publishers[0] = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=5)
        self.publishers[1] = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=5)
        self.publishers[2] = rospy.Publisher('/iri_wam/joint3_position_controller/command', Float64, queue_size=5)
        self.publishers[3] = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=5)
        self.publishers[4] = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.publishers[5] = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=5)
        self.publishers[6] = rospy.Publisher('/iri_wam/joint7_position_controller/command', Float64, queue_size=5) # discretely publishing motor actions for now
        self.pubMarker = rospy.Publisher('/goalPose', Marker, queue_size=100)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.minDisplacement = 0.09
        #self.minDisplacementPose = 0.09
        self.baseFrame = 'iri_wam_link_base'
        self.waitTime = 15
        self.homingTime = 5

        self.home = np.zeros(7)
        self.Actualhigh = np.array([2.5, 1.9, 2.7, 3.0, 1.14, 1.47, 3])
        self.Actuallow = np.array([-2.5, -1.9, -2.7, -0.8, -4.66, -1.47, -3])
        self.high = np.array([2.0, 1.5, 2.0, 2.5, 1.0, 1.47, 3])
        self.low = np.array([-2.0, -1.5, -2.0, -0.5, -4.0, -1.47, -3])
        #self.lowAction = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.highAction = np.array([3, 3, 3, 3, 3, 3, 3])
        self.checkDisplacement = np.array([self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement])

        self.lastObservation = None

        #self.action_space = spaces.MultiBinary(7)
        self.action_space = spaces.Box(-self.highAction, self.highAction)
        self.observation_space = spaces.Box(np.concatenate((self.low,self.low), axis=0), np.concatenate((self.high,self.high), axis=0))
        self.reward_range = (-np.inf, np.inf)


    def getForwardKinematics(self, goalPosition): #get catesian coordinates for joint Positions
        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
        try:
            getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
            jointPoseReturned = getFK(goalPosition)
            return jointPoseReturned
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)

    def getRandomGoal(self):  #sample from reachable positions
        frame_ID = self.baseFrame
        tempPosition = []
        for joint in range(7):
            tempPosition.append(random.uniform(self.low[joint], self.high[joint]))

        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = tempPosition
        tempPoseFK = self.getForwardKinematics(tempJointState)

        tempTemp = [tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z, tempPoseFK.pose.pose.orientation.x, tempPoseFK.pose.pose.orientation.y, tempPoseFK.pose.pose.orientation.z, tempPoseFK.pose.pose.orientation.w]

        return np.array(tempTemp)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        lastObs = self.lastObservation[:np.shape(self.high)[0]] # only the first seven observations, rest 7 give the goal 
    
        moved = False
        goalState = self.lastObservation[np.shape(self.high)[0]:] # the goal information was contained in the state observation

        pointToPose = Point()
        pointToPose.x = goalState[0]
        pointToPose.y = goalState[1]
        pointToPose.z = goalState[2]
        markerObj = Marker()
        markerObj.header.frame_id = self.baseFrame
        #markerObj.header.stamp = rospy.get_rostime()
        markerObj.id = 0
        markerObj.ns = 'iri_wam'
        markerObj.type = markerObj.SPHERE
        markerObj.action = markerObj.ADD
        markerObj.pose.position = pointToPose
        markerObj.pose.orientation.x = goalState[3]
        markerObj.pose.orientation.y = goalState[4]
        markerObj.pose.orientation.z = goalState[5]
        markerObj.pose.orientation.w = goalState[6]
        markerObj.color.r = 1.0
        markerObj.color.a = 1.0
        markerObj.scale.x = 0.09
        markerObj.scale.y = 0.09
        markerObj.scale.z = 0.09

        badDataFlag = False

        self.pubMarker.publish(markerObj)
        #print ("marker object two")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #print ("action received ", action )

        #print ("action received type ", type(actionU) )

        tempLastObs = np.copy(lastObs)
        for num, joint in enumerate(action):
            for itr in range(np.absolute(round(joint/self.minDisplacement))):
                self.publishers[num].publish(tempLastObs[num] + (joint/np.absolute(joint)) * self.minDisplacement)
                tempLastObs[num] = tempLastObs[num] + (joint/np.absolute(joint)) * self.minDisplacement
                


        # if   (action ==[0, 0, 0, 0]).all(): self.pub1.publish(lastObsForward[0])
        # elif (action ==[0, 0, 0, 1]).all(): self.pub2.publish(lastObsForward[1])
        # elif (action ==[0, 0, 1, 0]).all(): self.pub3.publish(lastObsForward[2])
        # elif (action ==[0, 0, 1, 1]).all(): self.pub4.publish(lastObsForward[3])
        # elif (action ==[0, 1, 0, 0]).all(): self.pub5.publish(lastObsForward[4])
        # elif (action ==[0, 1, 0, 1]).all(): self.pub6.publish(lastObsForward[5])
        # elif (action ==[0, 1, 1, 0]).all(): self.pub7.publish(lastObsForward[6])
        # elif (action ==[0, 1, 1, 1]).all(): self.pub1.publish(lastObsBackward[0])
        # elif (action ==[1, 0, 0, 0]).all(): self.pub2.publish(lastObsBackward[1])
        # elif (action ==[1, 0, 0, 1]).all(): self.pub3.publish(lastObsBackward[2])
        # elif (action ==[1, 0, 1, 0]).all(): self.pub4.publish(lastObsBackward[3])
        # elif (action ==[1, 0, 1, 1]).all(): self.pub5.publish(lastObsBackward[4])
        # elif (action ==[1, 1, 0, 0]).all(): self.pub6.publish(lastObsBackward[5])
        # elif (action ==[1, 1, 0, 1]).all(): self.pub7.publish(lastObsBackward[6])
        
        def roundOffList(lyst):
            newLyst = []
            for i in lyst:
                newLyst.append(round(i, 2))

            return np.array(newLyst)


        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=10)
                if (np.array(data.position)<=self.high).all() and (np.array(data.position)>=self.low).all():
                    stateArms = np.array(data.position)
                    state = np.concatenate((stateArms, goalState), axis=0) # get a random goal every time you reset
                    if (roundOffList(lastObs) != roundOffList(stateArms)).any(): 
                        moved = True
                    self.lastObservation = state
                else:
                    data = None
                    print ("Bad observation data received " )
                    badDataFlag = True
                    for joint in range(7):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
                        # self.pub2.publish(self.home[joint])
                        # self.pub3.publish(self.home[joint])
                        # self.pub4.publish(self.home[joint])
                        # self.pub5.publish(self.home[joint])
                        # self.pub6.publish(self.home[joint])
                        # self.pub7.publish(self.home[joint])

                    time.sleep(self.homingTime)
            except:
                pass

        

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")



        # to calcualte the reward we need to know the current state as tcp point so use Fk for that
        goalStatePose = Pose()
        goalStatePose.position.x = goalState[0]
        goalStatePose.position.y = goalState[1]
        goalStatePose.position.z = goalState[2]

        goalStatePose.orientation.x = goalState[3]
        goalStatePose.orientation.y = goalState[4]
        goalStatePose.orientation.z = goalState[5]
        goalStatePose.orientation.w = goalState[6]

        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = lastObs.tolist()
        lastObsPose = self.getForwardKinematics(tempJointState)



        tempJointState2 = JointState()
        tempJointState2.header.frame_id = self.baseFrame
        tempJointState2.position = stateArms.tolist()
        stateArmsPose = self.getForwardKinematics(tempJointState2)

        goalArmDifferenceLast = LA.norm(np.array([lastObsPose.pose.pose.position.x, lastObsPose.pose.pose.position.y, lastObsPose.pose.pose.position.z]) - np.array([goalStatePose.position.x, goalStatePose.position.y, goalStatePose.position.z]))
        goalArmDifference = LA.norm(np.array([stateArmsPose.pose.pose.position.x, stateArmsPose.pose.pose.position.y, stateArmsPose.pose.pose.position.z]) - np.array([goalStatePose.position.x, goalStatePose.position.y, goalStatePose.position.z]))
        

        diff = goalArmDifferenceLast - goalArmDifference 
        reward = diff

        #print ("Difference Total ", np.absolute(stateArms - goalState), ( np.absolute(stateArms - goalState) <= self.checkDisplacement).all() )
        #print(" gooal arm difference :", goalArmDifference)
        if ( goalArmDifference <= self.minDisplacement ):
            #print ("Difference Total ", np.absolute(stateArms - goalState), ( np.absolute(stateArms - goalState) <= self.checkDisplacement).all() )
            done = True
            reward += 10
        else: done = False
        




        return state, reward, done, badDataFlag, moved


    def _reset(self):

        # Resets the state of the environment and returns an initial observation.

        # rospy.wait_for_service('/gazebo/reset_simulation') # Reset simulation was causing problems, do not reset simulation
        # try:
        #     #reset_proxy.call()
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        print("In the reset LOOP")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


        rospy.wait_for_service('/iri_wam/controller_manager/switch_controller')
        try:
            change_controller = rospy.ServiceProxy('/iri_wam/controller_manager/switch_controller', SwitchController)
            ret = change_controller(['joint1_position_controller', 'joint2_position_controller', 'joint3_position_controller', 'joint4_position_controller', 'joint5_position_controller', 'joint6_position_controller', 'joint7_position_controller'], ['iri_wam_controller'], 2)
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


        for joint in range(7):
            self.publishers[joint].publish(self.home[joint]) #homing at every reset

        data = None
        stateFull = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        #state = None
        while (data is None):
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=10)
                if (np.array(data.position)<=self.high).all() and (np.array(data.position)>=self.low).all():
                    state = np.array(data.position)
                    #goalArray, goalPose = self.getRandomGoal()
                    stateFull = np.concatenate((state, self.getRandomGoal()), axis=0) # get a random goal every time you reset
                    self.lastObservation = stateFull
                    print("New Goal received!")
                    
                    
                else:
                    data = None
                    print ("Bad observation data received " )
                    for joint in range(7):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset

                    time.sleep(self.homingTime)
            except:
                pass
                

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

         #send the observed state to the robot

        #print ("State after reset si XXXXXXXXXXXXXXXXXXXXCCCCCCCCCCCCCCCCCCCCCCCC", stateFull )
        return stateFull
