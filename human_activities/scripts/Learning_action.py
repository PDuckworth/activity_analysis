#! /usr/bin/env python

import sys
import rospy
import actionlib
from learn_activities import Offline_ActivityLearning
from human_activities.msg import LearningAction, LearningResult

class Learning_server(object):
    def __init__(self, name= "LearnHumanActivities"):

        # Start server
        rospy.loginfo("Learning Human ACtivites action server")
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, LearningHumanActivitiesAction,
            execute_cb=self.execute, auto_start=False)
        self._as.start()


    def execute(self, goal):
        rerun = 1
        parallel = 1
        singular_val_threshold = 10

        assign_clstr = 0.01

        low_instances = 5
        n_iters = 1000
        create_images = False
        dirichlet_params = (0.5, 0.03)
        class_threshold = 0.3

        while not self._as.is_preempt_requested():
            ol = Offline_ActivityLearning(rerun_all=rerun)

            """get SOMA2 Objects""""
            ol.get_soma_objects()

            """load skeleton detections over all frames"""
            ol.get_events()

            """encode all the observations using QSRs"""
            ol.encode_qsrs(parallel)

            """create histograms with global code book"""
            ol.make_temp_histograms_sequentially()

            ol.make_term_doc_matrix(parallel, low_instances)

            """create tf-idf and LSA classes"""
            ol.learn_activities(singular_val_threshold, assign_clstr)

            """learn a topic model of activity classes"""
            ol.learn_topic_model_activities(n_iters, create_images, dirichlet_params, class_threshold)

            print "\n completed learning phase"
        self._as.set_succeeded(LearningResult())


if __name__ == "__main__":
    rospy.init_node('learning_human_activities_server')

    Learning_server(name = "LearnHumanActivities")
    rospy.spin()
