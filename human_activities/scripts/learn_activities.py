#!/usr/bin/env python

"""Analyse Episode store using Machine Learning techniques (offline)"""

__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

import rospy
import os, sys, time
import getpass
import numpy as np
import cPickle as pickle
from create_events import *

class Offline_ActivityLearning(object):

    def __init__(self):
        print "starting activity learning"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/SafeZone'
        self.mean_window = 5

    def encode_qsrs(self):
        for cnt, self.recording in enumerate(self.path):
            print "encoding file: ", self.recording
            get_event(self.recording, self.path, self.mean_window)
        return True

    def make_histograms(self):
        return True

    def learn(self):
        return True




if __name__ == "__main__":
    rospy.init_node("Offline Activity Learner")

    o = Offline_ActivityLearning()

    o.encode_qsrs()
    #o.make_histograms()
    #o.learn()
