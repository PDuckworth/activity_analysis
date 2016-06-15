#! /usr/bin/env python

import sys
import rospy
import actionlib
from learn_activities import Offline_ActivityLearning
from human_activities.msg import LearningAction, LearningResult

class Learning_server(object):
    def __init__(self):

        # Start server
        rospy.loginfo("Learning Human ACtivites action server")
        self._as = actionlib.SimpleActionServer("LearningHumanActivities", LearningHumanActivitiesAction,
            execute_cb=self.execute, auto_start=False)
        self._as.start()


    def execute(self, goal):

        ol = Offline_ActivityLearning()

        """get SOMA2 Objects""""
        if not self._as.is_preempt_requested():
            ol.get_soma_objects()

        """load skeleton detections over all frames"""
        if not self._as.is_preempt_requested():
            ol.get_events

        """encode all the observations using QSRs"""
        if not self._as.is_preempt_requested():
            ol.encode()

        """create histograms with global code book"""
        if not self._as.is_preempt_requested():
            ol.make_histograms()

        """create tf-idf and LSA classes"""
        if not self._as.is_preempt_requested():
            ol.learn()

        self._as.set_succeeded(LearningResult())


if __name__ == "__main__":
    rospy.init_node('LearningHumanActivities_server')

    Learning_server()
    rospy.spin()
