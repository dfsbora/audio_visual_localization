#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import argparse


class TurtleBotController:
    def __init__(self, angular_speed=0.5):
        """
        Initializes the TurtleBotController node.

        :param angular_speed: Maximum angular speed (rad/s) for rotation.
        """
        # Initialize the ROS node
        rospy.init_node('turtlebot_orientation_controller', anonymous=True)

        # Robot speed parameters
        self.max_angular_speed = angular_speed
        self.min_angular_speed = 0.15

        # Orientation variables
        self.current_odom_yaw = None
        self.target_yaw = None
        self.yaw_threshold = math.radians(5.0)
        self.in_movement = False

        self.rate = rospy.Rate(10)

        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.angle_sub = rospy.Subscriber('/sound_source_angle', Float32, self.target_angle_callback)
        rospy.sleep(2)

    def odom_callback(self, msg):
        """
        Callback function for odometry data. Updates the current yaw angle.
        """
        orientation_q = msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.current_odom_yaw = yaw

    def target_angle_callback(self, msg):
        """
        Callback function for angle messages. Sets the absolute target yaw based on the received angle.
        """
        relative_yaw = math.radians(msg.data)
        if self.current_odom_yaw is not None and not self.in_movement:
            self.target_yaw = self.current_odom_yaw + relative_yaw
            self.target_yaw = math.atan2(math.sin(self.target_yaw), math.cos(self.target_yaw))

    def rotate_to_orientation(self):
        """
        Rotates the TurtleBot to achieve the desired orientation.
        """
        if self.target_yaw is None or self.current_odom_yaw is None:
            rospy.logwarn("Target yaw or current yaw is not initialized.")
            return

        vel_msg = Twist()
        self.in_movement = True  # Avoid changing direction mid-movement

        while not rospy.is_shutdown():
            yaw_error = self.target_yaw - self.current_odom_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

            if abs(yaw_error) < 0.1:
                vel_msg.angular.z = 0
                self.cmd_vel_pub.publish(vel_msg)
                self.in_movement = False
                self.rate.sleep()
                continue


            # Set proportional speed, but clip to a minimum
            p_angular_speed = self.max_angular_speed * yaw_error
            if abs(p_angular_speed) < self.min_angular_speed:
                vel_msg.angular.z = self.min_angular_speed * (1 if p_angular_speed >= 0 else -1)
            else:
                vel_msg.angular.z = p_angular_speed

            self.cmd_vel_pub.publish(vel_msg)
            self.rate.sleep()

    def print_angle(self):
        self.current_odom_yaw = 0
        while not rospy.is_shutdown():
            print("Angle: ", self.target_yaw)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        # USE COMMAND LINE
        """
        parser = argparse.ArgumentParser(description='Rotate TurtleBot to a specified angle.')
        parser.add_argument('angle', type=float, help='The angle in degrees to rotate the TurtleBot.')
        args = parser.parse_args()
        relative_yaw = math.radians(args.angle)
        controller = TurtleBotController()
        controller.rotate_to_orientation(relative_yaw)
        exit()
        """

        # USE ROS MESSAGE
        controller = TurtleBotController()
        controller.rotate_to_orientation()

    except rospy.ROSInterruptException:
        pass
