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

        self.pub1 = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=5)
        self.pub2 = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=5)
        self.pub3 = rospy.Publisher('/iri_wam/joint3_position_controller/command', Float64, queue_size=5)
        self.pub4 = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=5)
        self.pub5 = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.pub6 = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=5)
        self.pub7 = rospy.Publisher('/iri_wam/joint7_position_controller/command', Float64, queue_size=5) # discretely publishing motor actions for now
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.minDisplacement = None
        self.minDisplacementPose = None
        self.baseFrame = None
        self.waitTime = None

        self.home = np.zeros(7)
        self.high = np.array([2.5, 1.9, 2.7, 3.0, 1.14, 1.47, 3])
        self.low = np.array([-2.5, -1.9, -2.7, -0.8, -4.66, -1.47, -3])
        self.minDisplacementCheck = 0.09
        self.checkDisplacement = np.array([self.minDisplacementCheck, self.minDisplacementCheck, self.minDisplacementCheck, self.minDisplacementCheck, self.minDisplacementCheck, self.minDisplacementCheck, self.minDisplacementCheck])

        self.envelopeAugmented = self.high - self.low 
        self.lastObservation = None

        #self.action_space = spaces.MultiBinary(7)
        self.action_space = spaces.Discrete(14)
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
        #print (self.getForwardKinematics(tempJointState))

        return np.array(tempPosition)

    def getRandomGoal2(self):  #sample from reachable positions
        frame_ID = self.baseFrame
        tempPosition = []
        for joint in range(7):
            tempPosition.append(random.uniform(self.low[joint], self.high[joint]))

        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = tempPosition
        tempPoseFK = self.getForwardKinematics(tempJointState)
        #print (tempPoseFK)

        tempTemp = [tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z, tempPoseFK.pose.pose.orientation.x, tempPoseFK.pose.pose.orientation.y, tempPoseFK.pose.pose.orientation.z, tempPoseFK.pose.pose.orientation.w]

        return np.array(tempTemp)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        lastObs = self.lastObservation[:np.shape(self.high)[0]] # only the first seven observations, rest 7 give the goal 
        lastObsForward = (lastObs + self.minDisplacement)
        lastObsBackward = (lastObs - self.minDisplacement)

        goalState = self.lastObservation[np.shape(self.high)[0]:] # the goal information was contained in the state observation


        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: self.pub1.publish(lastObsForward[0])
        elif action == 1: self.pub2.publish(lastObsForward[1])
        elif action == 2: self.pub3.publish(lastObsForward[2])
        elif action == 3: self.pub4.publish(lastObsForward[3])
        elif action == 4: self.pub5.publish(lastObsForward[4])
        elif action == 5: self.pub6.publish(lastObsForward[5])
        elif action == 6: self.pub7.publish(lastObsForward[6])
        elif action == 7: self.pub1.publish(lastObsBackward[0])
        elif action == 8: self.pub2.publish(lastObsBackward[1])
        elif action == 9: self.pub3.publish(lastObsBackward[2])
        elif action == 10: self.pub4.publish(lastObsBackward[3])
        elif action == 11: self.pub5.publish(lastObsBackward[4])
        elif action == 12: self.pub6.publish(lastObsBackward[5])
        elif action == 13: self.pub7.publish(lastObsBackward[6])
        

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=10)
                if (np.array(data.position)<=self.high).all() and (np.array(data.position)>=self.low).all():
                    stateArms = np.array(data.position)
                    state = np.concatenate((stateArms, goalState), axis=0) # get a random goal every time you reset
                    self.lastObservation = state
                else:
                    data = None
                    print ("Bad observation data received " )
                    for joint in range(7):
                        self.pub1.publish(self.home[joint]) #homing at every reset
                        self.pub2.publish(self.home[joint])
                        self.pub3.publish(self.home[joint])
                        self.pub4.publish(self.home[joint])
                        self.pub5.publish(self.home[joint])
                        self.pub6.publish(self.home[joint])
                        self.pub7.publish(self.home[joint])
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
        print(" gooal arm difference :", goalArmDifference)
        if ( goalArmDifference <= self.minDisplacementPose ):
            #print ("Difference Total ", np.absolute(stateArms - goalState), ( np.absolute(stateArms - goalState) <= self.checkDisplacement).all() )
            done = True
            reward += 10
        else: done = False
        




        return state, reward, done, {}


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
            self.pub1.publish(self.home[joint]) #homing at every reset
            self.pub2.publish(self.home[joint])
            self.pub3.publish(self.home[joint])
            self.pub4.publish(self.home[joint])
            self.pub5.publish(self.home[joint])
            self.pub6.publish(self.home[joint])
            self.pub7.publish(self.home[joint])

        data = None
        #state = None
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=10)
                if (np.array(data.position)<=self.high).all() and (np.array(data.position)>=self.low).all():
                    state = np.array(data.position)
                    stateFull = np.concatenate((state, self.getRandomGoal2()), axis=0) # get a random goal every time you reset
                    self.lastObservation = stateFull
                    # print("New Goal received! FIRSTTTTTTTT")
                    # PoseObtained = self.getRandomGoal2()
                    # print (PoseObtained)
                    print("New Goal received!")
                    #print ("New Authentic observation data received :")
                else:
                    data = None
                    print ("Bad observation data received " )
                    for joint in range(7):
                        self.pub1.publish(self.home[joint]) #homing at every reset
                        self.pub2.publish(self.home[joint])
                        self.pub3.publish(self.home[joint])
                        self.pub4.publish(self.home[joint])
                        self.pub5.publish(self.home[joint])
                        self.pub6.publish(self.home[joint])
                        self.pub7.publish(self.home[joint])
            except:
                pass
                

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

         #send the observed state to the robot

        return stateFull
