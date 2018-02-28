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
from std_msgs.msg import Float64
from controller_manager_msgs.srv import SwitchController
from gym.utils import seeding
from numpy import linalg as LA



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
        self.baseFrame = None
        self.waitTime = None

        self.home = np.zeros(7)
        self.high = np.array([2.5, 1.9, 2.7, 3.0, 1.14, 1.47, 3])
        self.low = np.array([-2.5, -1.9, -2.7, -0.8, -4.66, -1.47, -3])
        self.checkDisplacement = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

        self.envelopeAugmented = self.high - self.low 
        self.lastObservation = None

        #self.action_space = spaces.MultiBinary(7)
        self.action_space = spaces.Discrete(14)
        self.observation_space = spaces.Box(np.concatenate((self.low,self.low), axis=0), np.concatenate((self.high,self.high), axis=0))
        self.reward_range = (-np.inf, np.inf)

    

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

        #jointNumber = action[0]
        #jointAction = action[1]


        #if jointAction == 0: self.publishers[jointNumber].publish(lastObsForward[jointNumber])
        #elif jointAction == 1: self.publishers[jointNumber].publish(lastObsBackward[jointNumber])

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



        goalArmDifferenceLast = LA.norm(lastObs - goalState)
        goalArmDifference = LA.norm(stateArms - goalState)
        

        diff = goalArmDifferenceLast - goalArmDifference
        if diff>0:
            reward = diff
        else: reward = 0
        #reward = goalArmDifference

        if ( np.absolute(stateArms - goalState) <= self.checkDisplacement).all():
            done = True
            reward += 10
        else: done = False
        #done = False

        # if not done:
        #     if action == 0:
        #         reward = 5
        #     else:
        #         reward = 1
        # else:
        #     reward = -200




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
                    stateFull = np.concatenate((state, self.getRandomGoal()), axis=0) # get a random goal every time you reset
                    self.lastObservation = stateFull
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
