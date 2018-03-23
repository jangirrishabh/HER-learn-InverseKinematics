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
        self.publishers = ['pub1', 'pub2', 'pub4', 'pub3', 'pub5', 'pub6', 'pub7'] #publishers for the motor commands

        self.publishers[0] = rospy.Publisher('/iri_wam/joint1_position_controller/command', Float64, queue_size=5)
        self.publishers[1] = rospy.Publisher('/iri_wam/joint2_position_controller/command', Float64, queue_size=5)
        self.publishers[2] = rospy.Publisher('/iri_wam/joint4_position_controller/command', Float64, queue_size=5)
        self.publishers[3] = rospy.Publisher('/iri_wam/joint3_position_controller/command', Float64, queue_size=5)
        self.publishers[4] = rospy.Publisher('/iri_wam/joint5_position_controller/command', Float64, queue_size=5)
        self.publishers[5] = rospy.Publisher('/iri_wam/joint6_position_controller/command', Float64, queue_size=5)
        self.publishers[6] = rospy.Publisher('/iri_wam/joint7_position_controller/command', Float64, queue_size=5) # discretely publishing motor actions for now
        self.pubMarker = rospy.Publisher('/goalPose', Marker, queue_size=5)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.minDisplacement = 0.09 #minimum discrete distance by a single action
        self.minDisplacementPose = 0.10 # minimum distance to check for the goal reaching state
        self.baseFrame = 'iri_wam_link_base' 
        self.waitTime = 3 # time to wait for arm to reach a joint until its considered a bad point
        self.homingTime = 0.5 # time given for homing
        self.lenGoal = 3 # goal position list length
        self.REWARD = 0 # reward received upon reachnig the goal
        self.rewScaleFactor = 10 # multiply the rewards by this factor
        self.actionTime = 0.05 #time delay after every published action
        self.DATA = 0
        self.TRAIN = not self.DATA
        self.BC = not self.DATA
        self.REWARD_TYPE = "dense" #sparse or dense

        self.home = np.zeros(4) # what position is the homing
        # self.Xacrohigh = np.array([2.6, 2.0, 2.8, 3.1, 1.24, 1.57, 2.96])
        # self.Xacrolow = np.array([-2.6, -2.0, -2.8, -0.9, -4.76, -1.57, -2.96])
        # self.IKlow = np.array([-2.6, -1.94, -2.73, -0.88, -4.74, -1.55, -2.98])
        # self.IKhigh = np.array([2.6, 1.94, 2.73, 3.08, 1.22, 1.55, 2.98])
        
        self.low = np.array([-2.6, -1.42, -2.73, -0.88, -4.74, -1.55, -2.98])
        self.high = np.array([2.6, 1.42, 2.73, 3.08, 1.22, 1.55, 2.98])

        self.lowConc = np.array([-2.6, -1.42, -0.88, -2.6, -1.42, -0.88]) #1, 2, 4, 1, 2, 4
        self.highConc = np.array([2.6, 1.42, 3.08, 2.6, 1.42, 3.08])

        self.lowConcObs = np.array([-2.6, -1.42, -0.88]) #1, 2, 4
        self.highConcObs = np.array([2.6, 1.42, 3.08])

        #self.samplelow = np.array([-2.4, -1.4, -2.03, -0.68, -4.04, -1.05, -2.08])
        #self.samplehigh = np.array([2.4, 1.4, 2.03, 2.78, 1.0, 1.05, 2.08])

        self.samplelow = np.array([-2.0, -1.0, -2.03, -0.58, -4.04, -1.05, -2.08])
        self.samplehigh = np.array([2.0, 1.0, 2.03, 2.08, 1.0, 1.05, 2.08])
        #self.high = np.array([5.2, 2.8, 5.4, 3.96, 6.96, 3.1, 5.96])        
        self.lowAction = [-1, -1, -1]
        self.highAction = [1, 1, 1]
        self.checkDisplacement = np.array([self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement, self.minDisplacement])

        self.lastObservation = None
        self.lastObservationFull = None

        #self.action_space = spaces.MultiBinary(7)
        self.action_space = spaces.Box(-1., 1., shape=(len(self.highAction),))
        self.observation_space = spaces.Box(self.lowConc, self.highConc)
        self.reward_range = (-np.inf, np.inf)


    def getForwardKinematics(self, goalPosition): #get catesian coordinates for joint Positions
        rospy.wait_for_service('/iri_wam/iri_wam_ik/get_wam_fk')
        try:
            getFK = rospy.ServiceProxy('/iri_wam/iri_wam_ik/get_wam_fk', QueryForwardKinematics)
            jointPoseReturned = getFK(goalPosition)
            return jointPoseReturned
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)


    def getGripperPosition(self, joints):
        frame_ID = self.baseFrame
        tempJointState = JointState()
        tempJointState.header.frame_id = self.baseFrame
        tempJointState.position = joints

        tempPoseFK = self.getForwardKinematics(tempJointState)

        return [tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z]

    def getRandomGoal(self):  #sample from reachable positions
        frame_ID = self.baseFrame
        tempPoseFK = None

        while tempPoseFK==None:
            tempPosition = []
            for joint in range(4):
                tempPosition.append(random.uniform(self.samplelow[joint], self.samplehigh[joint]))

            tempPosition.append(0)
            tempPosition.append(0)
            tempPosition.append(0)
            tempPosition[2] = 0
            tempJointState = JointState()
            tempJointState.header.frame_id = self.baseFrame
            tempJointState.position = tempPosition

            tempPoseFK = self.getForwardKinematics(tempJointState)
            if tempPoseFK!=None:
                tempTemp = [tempPoseFK.pose.pose.position.x, tempPoseFK.pose.pose.position.y, tempPoseFK.pose.pose.position.z, tempPoseFK.pose.pose.orientation.x, tempPoseFK.pose.pose.orientation.y, tempPoseFK.pose.pose.orientation.z, tempPoseFK.pose.pose.orientation.w]
                return tempTemp

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def homing(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        for joint in range(3):
            self.publishers[joint].publish(self.home[joint]) #homing at every reset
        time.sleep(self.homingTime)
        print ("HOMIED")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

    def _step(self, action):
        #print ("action received before4 clipping ", action )
        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print ("action received ", action )

        for i in range(len(action)):
            if action[i] > 0:
                action[i] = max(action[i], self.minDisplacement)
            elif action[i] <0:
                action[i] = -max(np.absolute(action[i]), self.minDisplacement)


        #print ("action Augmentedss ", action )
        lastObs = list(self.lastObservation[:self.lenGoal]) 
    
        moved = False
        badDataFlag = False
        goalState = list(self.lastObservation[self.lenGoal:]) # the goal information was contained in the state observation
        goalStateFull = list(self.lastObservationFull[(self.lenGoal*2):])
        assert len(goalState) == self.lenGoal
        #print ("goalState :", goalState, type(goalState))
        #print ("goalStateFull :", goalStateFull, type(goalStateFull))
        

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


        for num, joint in enumerate(action):
            self.publishers[num].publish(lastObs[num] + joint)
            
        if self.TRAIN: time.sleep(self.actionTime)
        
        def roundOffList(lyst):
            newLyst = []
            for i in lyst:
                newLyst.append(round(i, 4))

            return newLyst

        data = None
        state = list(self.lastObservationFull)
        while data is None:
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=1)
                gripperPos = self.getGripperPosition(data.position)
                dataConc = [data.position[0], data.position[1], data.position[3]]
                if ((np.array(dataConc)<=self.highConcObs).all()) and ((np.array(dataConc)>=self.lowConcObs).all()):
                    stateConc = gripperPos + goalState # get a random goal every time you reset
                    stateShort = dataConc + goalState
                    state = dataConc + gripperPos + goalStateFull  # get a random goal every time you reset
                    if (np.array(roundOffList(lastObs)) != np.array(roundOffList(dataConc))).any(): 
                        moved = True
                    self.lastObservation = stateShort
                    self.lastObservationFull = state
                else:
                    data = None
                    print ("Bad observation data received STEP" )
                    badDataFlag = True
                    for joint in range(3):
                        self.publishers[joint].publish(self.home[joint]) #homing at every reset
                    time.sleep(self.homingTime)
            except:
                pass


        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # to calcualte the reward we need to know the current state as tcp point so use Fk for that


        lastObsPose = None
        while lastObsPose==None:
            tempJointState = JointState()
            tempJointState.header.frame_id = self.baseFrame
            tempJointState.position = [lastObs[0], lastObs[1], 0, lastObs[2], 0, 0, 0]
            lastObsPose = self.getForwardKinematics(tempJointState)
            if lastObsPose == None: print ("here is the fuck up in last pose, ", lastObs, len(lastObs))

        stateArmsPose = None
        while stateArmsPose==None:
            tempJointState2 = JointState()
            tempJointState2.header.frame_id = self.baseFrame
            tempJointState2.position = [dataConc[0], dataConc[1], 0, dataConc[2], 0, 0, 0]
            stateArmsPose = self.getForwardKinematics(tempJointState2)
            if stateArmsPose == None: print ("here is the fuck up in state arms pose, ", dataConc, ) 

        goalArmDifferenceLast = LA.norm(np.array([lastObsPose.pose.pose.position.x, lastObsPose.pose.pose.position.y, lastObsPose.pose.pose.position.z]) - np.array([goalState[0], goalState[1], goalState[2]]))
        goalArmDifference = LA.norm(np.array([stateArmsPose.pose.pose.position.x, stateArmsPose.pose.pose.position.y, stateArmsPose.pose.pose.position.z]) - np.array([goalState[0], goalState[1], goalState[2]]))
        

        diff = goalArmDifferenceLast - goalArmDifference 
        #print ("How far from thegoal ", goalArmDifference )


        reward = self.REWARD
        # if self.REWARD_TYPE == "sparse":
        #     reward = -self.REWARD
        # elif self.REWARD_TYPE == "dense":
        #     if diff>0 and moved:
        #         reward = (1/(1+ np.exp(goalArmDifference)))
        #     elif diff<0 and moved:
        #         reward = -(1/(1+ np.exp(goalArmDifference)))
        #     else:
        #         reward = -self.REWARD


        #print ("reaward received, ", reward)
        #print ("Difference Total ", np.absolute(stateArms - goalState), ( np.absolute(stateArms - goalState) <= self.checkDisplacement).all() )
        #print(" gooal arm difference :", goalArmDifference)

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
        markerObj.pose.orientation.w = 1.0
        markerObj.scale.x = 0.09
        markerObj.scale.y = 0.09
        markerObj.scale.z = 0.09


        if ( goalArmDifference <= (self.minDisplacementPose*3) ):
            markerObj.color.r = 0.5
            markerObj.color.g = 0.5
            markerObj.color.a = 1.0

        elif ( goalArmDifference <= (self.minDisplacementPose) ):
            markerObj.color.g = 1.0
            markerObj.color.a = 1.0

        else: 
            markerObj.color.r = 1.0
            markerObj.color.a = 1.0
            

        self.pubMarker.publish(markerObj)


        # if ( goalArmDifference <= (self.minDisplacementPose) ):
        #     #print ("Difference Total ", np.absolute(stateArms - goalState), ( np.absolute(stateArms - goalState) <= self.checkDisplacement).all() )
        #     done = True
            
        #     if self.REWARD_TYPE == "sparse":
        #         reward = self.REWARD
        #     elif self.REWARD_TYPE == "dense":
        #         reward = self.REWARD*self.rewScaleFactor
        #     #print ("Dhone dhana dan dan ho gaya ", stateConc)
        # else: 
        #     #reward = -self.REWARD
        #     done = False
        
        done = bool(goalArmDifference <= (self.minDisplacementPose))

        #print (" REWARD RECEIVED ", reward )

        if self.DATA: return np.array(state), reward, done, badDataFlag, moved  # when generating data through planning
        elif self.TRAIN: return np.array(stateConc), reward, done, {} #to train through gail


    def _reset(self):

        # Resets the state of the environment and returns an initial observation.

        # rospy.wait_for_service('/gazebo/reset_simulation') # Reset simulation was causing problems, do not reset simulation
        # try:
        #     #reset_proxy.call()
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        #print("In the reset LOOP")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")


        # rospy.wait_for_service('/iri_wam/controller_manager/list_controllers')
        # try:
        #     readController = rospy.ServiceProxy('/iri_wam/controller_manager/list_controllers', Empty)
        #     control = readController()
        #     #print (control)
        #     #onHai = bool(controllers.iri_wam_controller.state == "stopped")
        # except (rospy.ServiceException) as e:
        #     print ("Service call failed: %s"%e)

        #if not onHai:
        rospy.wait_for_service('/iri_wam/controller_manager/switch_controller')
        try:
            change_controller = rospy.ServiceProxy('/iri_wam/controller_manager/switch_controller', SwitchController)
            ret = change_controller(['joint1_position_controller', 'joint2_position_controller', 'joint3_position_controller', 'joint4_position_controller', 'joint5_position_controller', 'joint6_position_controller', 'joint7_position_controller'], ['iri_wam_controller'], 2)
        except (rospy.ServiceException) as e:
            print ("Service call failed: %s"%e)

        if self.TRAIN:
            for joint in range(3):
                self.publishers[joint].publish(self.home[joint]) #homing at every reset
            time.sleep(self.homingTime)

        data = None
        state = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        stateFull = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 , 0.1, 0.1, 0.1]
        while (data is None):
            try:
                data = rospy.wait_for_message('/joint_states', JointState, timeout=1)
                gripperPos = self.getGripperPosition(data.position)
                dataConc = [data.position[0], data.position[1], data.position[3]]
                if (np.array(dataConc)<=self.highConcObs).all() and (np.array(dataConc)>=self.lowConcObs).all():
                    goalArrayFull = self.getRandomGoal()
                    state = gripperPos + goalArrayFull[:self.lenGoal] # get a random goal every time you reset
                    stateShort = dataConc + goalArrayFull[:self.lenGoal]
                    stateFull = dataConc + gripperPos + goalArrayFull
                    self.lastObservation = stateShort
                    self.lastObservationFull = stateFull
                    print("New Goal received!")
                else:
                    data = None
                    print ("Bad observation data received " )
                    for joint in range(3):
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

        if self.DATA: return np.array(stateFull)
        if self.TRAIN: return np.array(state)