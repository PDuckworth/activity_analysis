#!/usr/bin/env python

"""Analyse Episode store using Machine Learning techniques (offline)"""

__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

import rospy
import os, sys, time
import getpass
import numpy as np
import cPickle as pickle
import multiprocessing as mp
from create_events import *
from encode_qsrs import *


class Offline_ActivityLearning(object):

    def __init__(self):
        print "starting activity learning"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
        self.mean_window = 5

    def get_events(self):
        """multiprocessing?"""
        path = os.path.join(self.path, 'SafeZone')
        for recording in [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]:
            print "encoding file: ", recording
            get_event(recording, path, self.mean_window)

    def encode_qsrs(self):
        """check for any events which are not QSRs yet"""
        path = os.path.join(self.path, 'Events')
        list_of_events = [i for i in sorted(os.listdir(path)) ]
        print list_of_events

        # num_procs = mp.cpu_count()
        # pool = mp.Pool(num_procs)
        # chunk_size = int(np.ceil(len(os.listdir(directory))/float(num_procs)))
        # pool.map(worker_qsrs, list_of_events, chunk_size)
        # pool.close()
        # pool.join()

        for cnt, i in enumerate(list_of_events):
            print i
            worker_qsrs(i)


    def make_histograms(self):
        return True

    def learn(self):
        return True




if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")

    o = Offline_ActivityLearning()

    o.get_events()
    o.encode_qsrs()
    #o.make_histograms()
    #o.learn()
