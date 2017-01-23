#! /usr/bin/env python
import roslib
import tf
import sys, os
import rospy

from tf.transformations import quaternion_from_euler
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, Point, Pose, Point32, Polygon, PoseArray
from sensor_msgs.msg import JointState

from scitos_ptu.msg import *
from mongodb_store.message_store import MessageStoreProxy
from record_skeletons_action.msg import ViewInfo
class view_manager(object):
    def __init__(self):
        """Query and investigate robot views
        """
        self.pub_viewpoints = rospy.Publisher('activity_viewpoints', PoseArray, queue_size=10)
        self.views_msg_store = MessageStoreProxy(collection='activity_view_stats')
        self.query_viewpoints()


    def query_viewpoints(self):
        """Query the database and retrieve the good views:
        1. where nav did fail
        2. which recorded and received consenst.
        """
        # query = {}
        ret = self.views_msg_store.query(ViewInfo._type)
        view_points = []
        for view, meta in ret:
            view_points.append(view.robot_pose)
        self.msg = PoseArray(poses = view_points)
        self.msg.header.frame_id = "/map"

    def publish(self):
        self.pub_viewpoints.publish(self.msg)


if __name__ == "__main__":
    rospy.init_node('record_skelton_view_evaluation')

    vm = view_manager()

    while not rospy.is_shutdown():
        rate = rospy.Rate(10) # 10hz
        vm.publish()
        rate.sleep()
