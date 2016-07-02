#!/usr/bin/env python

"""Analyse Episode store using Machine Learning techniques (offline)"""

__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

#import rospy
import os, sys, time
import yaml
import roslib
import getpass
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import human_activities.create_events as ce
import human_activities.encode_qsrs as eq
import human_activities.histograms as h

class Offline_ActivityLearning(object):

    def __init__(self, rerun_all=0, reduce_frame_rate=2, joints_mean_window=5, qsr_mean_window=3):
        print "initialise activity learning action"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'

        """FILTERS all effect each other. Frame rate reduction applied first. Filter to qsrs last."""
        self.reduce_frame_rate = reduce_frame_rate
        self.joints_mean_window = joints_mean_window
        self.qsr_mean_window = qsr_mean_window
        self.processed_path = os.path.join(self.path, 'QSR_Worlds')
        self.load_config()

        if rerun_all:
            for f in os.listdir(self.processed_path):
                print f, os.path.join(self.processed_path, f)
                os.remove(os.path.join(self.processed_path, f))

    def load_config(self):
        """load the config file from the data recordings (from tsc)"""
        try:
            config_filepath = os.path.join(roslib.packages.get_pkg_dir("skeleton_tracker"), "config")
            self.config = yaml.load(open(os.path.join(config_filepath, 'config.ini'), 'r'))
            print "config loaded:", self.config.keys()
        except:
            try:
                config_filepath = '/home/' + getpass.getuser() + '/catkin_ws/src/skeleton_tracker/config'
                self.config = yaml.load(open(os.path.join(config_filepath, 'config.ini'), 'r'))
                print "config loaded:", self.config.keys()
            except:
                print "no config file found"

        # make this smarter - use Somaroi?
        self.soma_roi_config = {'KitchenTableLow':'Kitchen', 'KitchenTableHigh':'Kitchen', 'KitchenCounter1':'Kitchen', 'KitchenCounter2':'Kitchen', 'KitchenCounter3':'Kitchen',
                                'ReceptionDesk':'Reception', 'HospActRec1':'Hospitality', 'HospActRec4':'Hopotality', 'CorpActRec3':'Corporate', 'SuppActRec1': 'Support' }

    def get_soma_objects(self):
        """srv call to mongo and get the list of objects and locations"""
        #todo: make this SOMA2 instead of hardcoded object locations
        self.soma_objects = ce.get_soma_objects()
        print self.soma_objects.keys()

    def get_events(self):
        """multiprocessing?"""
        print "\nEvents"
        path = os.path.join(self.path, 'SafeZone')
        for recording in os.listdir(path):
            if os.path.isdir(os.path.join(path, recording)) and not os.path.isfile(os.path.join(self.processed_path, recording+".p")):
                waypoint = recording.split('_')[-1]
                region = self.soma_roi_config[waypoint]
                # if waypoint == "KitchenCounter1":
                # print "geting skeleton data: ", recording, waypoint, region
                # print "s", self.soma_objects[region]
                # print "p", self.config[waypoint]
                ce.get_event(recording, path, self.soma_objects[region], self.config[waypoint], self.reduce_frame_rate, self.joints_mean_window)
            else:
                print "already processed: %s" % recording

    def encode_qsrs(self, parallel=0):
        """check for any events which are not QSRs yet"""
        print "\nQSRS"
        path = os.path.join(self.path, 'Events')
        list_of_events = []
        for recording in sorted(os.listdir(path)):
            waypoint = recording.split('_')[-1].strip(".p")
            region = self.soma_roi_config[waypoint]
            if not os.path.isfile(os.path.join(self.processed_path, recording)):
                list_of_events.append((recording, path, self.soma_objects[region], self.qsr_mean_window))

        if parallel:
            num_procs = mp.cpu_count()
            pool = mp.Pool(num_procs)
            chunk_size = int(np.ceil(len(os.listdir(path))/float(num_procs)))
            pool.map(eq.worker_qsrs, list_of_events, chunk_size)
            pool.close()
            pool.join()
        else: # for sequential debugging:
            for cnt, event in enumerate(list_of_events):
                print "encoding QSRs: ", event[0]
                eq.worker_qsrs(event)

    def make_temp_histograms_sequentially(self):
        self.len_of_code_book = h.create_temp_histograms(self.path)
        return True

    def make_term_doc_matrix(self):
        path = os.path.join(self.path, 'Histograms')
        list_of_histograms =[ (recording, path) for recording in sorted(os.listdir(path)) ]
        print list_of_histograms
        sys.exit(1)

        if parallel:
            num_procs = mp.cpu_count()
            pool = mp.Pool(num_procs)
            chunk_size = int(np.ceil(len(os.listdir(path))/float(num_procs)))
            pool.map(eq.worker_qsrs, list_of_events, chunk_size)
            pool.close()
            pool.join()
        else: # for sequential debugging:
            for cnt, event in enumerate(list_of_events):
                print "encoding QSRs: ", event[0]
                eq.worker_qsrs(event)


        #h.create_term_doc_matrix(path, )
        return True


    def learn(self):
        return True




if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")

    rerun = 1
    parallel = 0

    o = Offline_ActivityLearning(rerun_all=rerun)

    o.get_soma_objects()
    o.get_events()
    o.encode_qsrs(parallel)
    o.make_temp_histograms_sequentially()
    # o.make_term_doc_matrix()

    #o.learn()
