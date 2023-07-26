import gym
from gym import spaces
from gym.utils import seeding
import rospy
import numpy as np
from geometry_msgs.msg import Point
from std_srvs.srv import Empty
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from actionlib import SimpleActionClient
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

class UR5eEnv(gym.Env):
    def __init__(self):
        super(UR5eEnv, self).__init__()

        # Example when using continuous actions
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        print("Initialized action space.")

        # Define observation space: joint states (6) + object position (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        print("Initialized observation space.")

        # ROS node
        rospy.init_node('rl_node', anonymous=True)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        print("Waiting for controller server.")
        self.client = SimpleActionClient('/scaled_pos_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.client.wait_for_server()

        print("Controller server found.")
        
        self.joint_states = JointState()
        rospy.Subscriber("/pos_joint_traj_controller/state", JointTrajectoryControllerState, self.joint_states_callback)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.joint_states = JointState()

        # Replace with your object's actual position
        self.object_position = np.array([0.5, 0, 0.2])
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        # Wait for up to 5 seconds for the first transforms to become available.
        rospy.sleep(5)

    def step(self, action):
        # Execute this step by publishing a message to the robot
        self.execute_action(action)

        # Get an observation
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward(observation)

        # Check if done
        done = self.check_if_done(observation)

        return observation, reward, done, {}

    def reset(self):
        self.reset_proxy()
        observation = self.get_observation()
        return observation

    def render(self, mode='human', close=False):
        pass

    def execute_action(self, action):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = action.tolist()
        point.time_from_start = rospy.Duration(1.0)
        goal.trajectory.points.append(point)
        self.client.send_goal(goal)
        self.client.wait_for_result()

    def get_observation(self):
        print(self.object_position)
        if len(self.joint_states.desired.positions) == 0:
            self.joint_states.desired.positions = [0.0] * 6
        print(self.joint_states.desired.positions)
        
        return np.concatenate((self.joint_states.desired.positions[:7], self.object_position))

    def calculate_reward(self, observation):
        # Reward based on distance to the object
        end_effector_position = self.get_end_effector_position()
        dist_to_object = np.linalg.norm(self.object_position - end_effector_position)

        # Huge reward for accomplishing the task
        if dist_to_object < 0.05:  # Assuming 5cm is close enough
            return 1000.0

        # Penalize according to the distance to the object
        return -dist_to_object

    def check_if_done(self, observation):
        end_effector_position = self.get_end_effector_position()
        dist_to_object = np.linalg.norm(self.object_position - end_effector_position)

        # Done if the task is accomplished
        if dist_to_object < 0.05:  # Assuming 5cm is close enough
            return True

        return False

    def get_end_effector_position(self):
        # Placeholder. Replace with actual method to get end effector position
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        # print(self.tfBuffer.__get_frames("").__slots__)
        try:
            # Look up the transform from 'base_link' to 'end_effector' and apply it to our pose
            pose_transformed = self.tfBuffer.transform(pose, 'tool0', rospy.Duration(1.0))
            print()
            print(pose_transformed)
            return [pose_transformed.pose.position.x, pose_transformed.pose.position.y, pose_transformed.pose.position.z]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to compute forward kinematics {e}")
            return np.array([0, 0, 0])

    def joint_states_callback(self, data):
        self.joint_states = data
