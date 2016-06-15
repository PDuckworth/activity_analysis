#!/usr/bin/env python

"""Analyse Episode store using Machine Learning techniques (offline)"""

__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

#import rospy
import os, sys, time
import getpass
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import human_activities.create_events as ce
import human_activities.encode_qsrs as eq

class Offline_ActivityLearning(object):

    def __init__(self, reduce_frame_rate=2, joints_mean_window=5, qsr_mean_window=5):
        print "initialise activity learning action"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'

        """FILTERS all effect each other. Frame rate reduction applied first. Filter to qsrs last."""
        self.reduce_frame_rate = reduce_frame_rate
        self.joints_mean_window = joints_mean_window
        self.qsr_mean_window = qsr_mean_window


    def get_soma_objects(self):
        """srv call to mongo and get the list of objects and locations"""
        #todo: make this SOMA2 instead of hardcoded object locations
        self.soma_objects = ce.get_soma_objects()
        print self.soma_objects.keys()

    def get_events(self):
        """multiprocessing?"""
        path = os.path.join(self.path, 'SafeZone')
        for recording in [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]:
            print "geting skeleton data: ", recording
            waypoint = recording.split('_')[-1]
            ce.get_event(recording, path, self.soma_objects[waypoint], self.reduce_frame_rate, self.joints_mean_window)

    def encode_qsrs(self):
        """check for any events which are not QSRs yet"""
        path = os.path.join(self.path, 'Events')
        list_of_events = [(i, path, self.soma_objects[i.strip(".p").split("_")[-1]], self.qsr_mean_window) \
            for i in sorted(os.listdir(path)) ]

        # num_procs = mp.cpu_count()
        # pool = mp.Pool(num_procs)
        # chunk_size = int(np.ceil(len(os.listdir(path))/float(num_procs)))
        # pool.map(eq.worker_qsrs, list_of_events, chunk_size)
        # pool.close()
        # pool.join()

        for cnt, event in enumerate(list_of_events):
            print "encoding QSRs: ", event[0]
            eq.worker_qsrs(event)


    def make_histograms(self):
        return True

    def learn(self):
        return True




if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")

    o = Offline_ActivityLearning()

    o.get_soma_objects()
    o.get_events()
    o.encode_qsrs()
    #o.make_histograms()
    #o.learn()
