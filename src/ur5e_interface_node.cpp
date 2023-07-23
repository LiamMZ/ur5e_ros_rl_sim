#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "sensor_msgs/JointState.h"

class RobotInterfaceNode
{
public:
  RobotInterfaceNode()
  {
    // Initialize the publisher and subscriber
    pub_ = nh_.advertise<std_msgs::Float64MultiArray>("joint_commands", 10);
    sub_ = nh_.subscribe("joint_states", 10, &RobotInterfaceNode::jointStateCallback, this);
  }

  void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    // Do something with the joint state data
    ROS_INFO("Received joint state data");
  }

  void sendCommand(const std_msgs::Float64MultiArray& command)
  {
    pub_.publish(command);
  }

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
};

int main(int argc, char** argv)
{
  // Initialize the node
  ros::init(argc, argv, "my_robot_rl_node");

  RobotInterfaceNode robot_interface_node;

  // Spin
  ros::spin();

  return 0;
}
