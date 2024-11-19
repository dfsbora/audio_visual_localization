#!/usr/bin/env python

"""
ROS node to publish simulated angles to test robot movement
"""

import rospy
import random
from std_msgs.msg import Float32


class AngleSimulator:
    def __init__(self):
        """
        Initializes the AngleSimulator node.
        - Sets up the ROS node with the name 'angle_simulator'.
        - Initializes a publisher for the topic '/sound_source_angle' with message type Float32.
        """
        rospy.init_node('angle_simulator', anonymous=True)
        self.angle_pub = rospy.Publisher('/sound_source_angle', Float32, queue_size=10)
        self.rate = rospy.Rate(1)
        self.last_angle = 0.0

    def generate_similar_angle(self):
        """Generate an angle similar to the last published angle."""
        variation = random.uniform(-5, 5)
        new_angle = self.last_angle + variation
        new_angle = max(min(new_angle, 90), -90)
        return new_angle

    def publish_angles(self):
        """
        Publishes angles to the '/sound_source_angle' topic.
        - Randomly decides whether to publish a completely random angle or one similar to the last angle.
        - Publishes the angle as a Float32 message.
        """
        while not rospy.is_shutdown():
            if random.random() < 0.3:
                simulated_angle = random.uniform(-90, 90)
            else:
                simulated_angle = self.generate_similar_angle()

            rospy.loginfo(f"Publishing simulated angle: {simulated_angle:.2f} degrees")

            self.angle_pub.publish(Float32(simulated_angle))
            self.last_angle = simulated_angle
            self.rate.sleep()

    def publish_45_angles(self):
        """
        Publishes a fixed set of angles (-45 and 45 degrees) to the '/sound_source_angle' topic.
        """
        while not rospy.is_shutdown():
            for simulated_angle in [45, -45]:
                rospy.loginfo(f"Publishing simulated angle: {simulated_angle:.2f} degrees")
                self.angle_pub.publish(Float32(simulated_angle))
                self.rate.sleep()


if __name__ == '__main__':
    try:
        simulator = AngleSimulator()
        simulator.publish_45_angles()
    except rospy.ROSInterruptException:
        pass
