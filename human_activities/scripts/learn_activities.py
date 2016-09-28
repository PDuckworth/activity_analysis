#!/usr/bin/env python
"""Analyse Episode store using Machine Learning techniques (offline)"""
__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

import os, sys, time
import yaml
import roslib
import getpass
import rospy
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import human_activities.create_events as ce
import human_activities.encode_qsrs as eq
import human_activities.histograms as h
import human_activities.tfidf as tfidf

from mongodb_store.message_store import MessageStoreProxy
from skeleton_tracker.msg import skeleton_message, skeleton_complete

class Offline_ActivityLearning(object):

    def __init__(self, rerun_all=0, reduce_frame_rate=2, joints_mean_window=5, qsr_mean_window=3):
        print "initialise activity learning action class"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'

        self._message_store = rospy.get_param("~message_store", "people_skeleton")
        self._database = rospy.get_param("~database", "message_store")
        self._store_client = MessageStoreProxy(collection=self._message_store, database=self._database)

        self.query = {} #"number_of_detections":100}

        query = {"date":"2016-09-28"}
        ret =  self._store_client.query(skeleton_complete._type, query)


        print len(ret)
        print ret[0].uuid, ret[0].time, ret[0].date
        sys.exit(1)
        """NEED A WAY OF QUERYING and UPLOADING TO MONGO"""

        """FILTERS all effect each other. Frame rate reduction applied first. Filter to qsrs last."""
        self.reduce_frame_rate = reduce_frame_rate
        self.joints_mean_window = joints_mean_window
        self.qsr_mean_window = qsr_mean_window
        self.events_path = os.path.join(self.path, 'Events')
        self.processed_path = os.path.join(self.path, 'QSR_Worlds')
        self.load_config()

        if rerun_all:
            """Dont re-process a skeleton if its already in QSR world format."""
            for f in os.listdir(self.processed_path):
                # print f, os.path.join(self.processed_path, f)
                os.remove(os.path.join(self.processed_path, f))
            for f in os.listdir(self.events_path):
                os.remove(os.path.join(self.events_path, f))

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

        # make this smarter - use SomaRoi?
        self.soma_roi_config = {'KitchenTableLow':'Kitchen', 'KitchenTableHigh':'Kitchen',
                                'KitchenCounter1':'Kitchen', 'KitchenCounter2':'Kitchen', 'KitchenCounter3':'Kitchen',
                                'KitchenDemo':'Kitchen',
                                'ReceptionDesk':'Reception', 'HospActRec1':'Hospitality',
                                'HospActRec4':'Hospitality', 'CorpActRec3':'Corporate', 'SuppActRec1': 'Support' }

    def get_soma_objects(self):
        """srv call to mongo and get the list of objects and locations"""
        #todo: make this SOMA2 instead of hardcoded object locations
        self.soma_objects = ce.get_soma_objects()
        print self.soma_objects.keys()




    def get_events(self):
        """multiprocessing?"""
        print "\nEvents"

        """NEED TO GET SKELETON MSGS FROM MONGO"""
    def _retrieve_logs(self):
        query = {"soma_roi_id":str(self.roi)}
        logs = self._client.message_store.spatial_qsr_models.find(query)

        path = os.path.join(self.path, 'consent')

        for recording in os.listdir(path):
            if os.path.isdir(os.path.join(path, recording)) and not os.path.isfile(os.path.join(self.processed_path, recording+".p")):
                print recording
                # waypoint = recording.split('_')[-1]
                # region = self.soma_roi_config[waypoint]
                region = "Kitchen"
                # if waypoint == "KitchenCounter1":
                # print "geting skeleton data: ", recording, waypoint, region
                # print "s", self.soma_objects[region]
                # print "p", self.config[waypoint]
                # print ">>", recording, waypoint, region
                ce.get_event(recording, path, self.soma_objects[region], self.config[region], self.reduce_frame_rate, self.joints_mean_window)
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

    def make_term_doc_matrix(self, low_instances=3):
        print "\nTERM FREQ MAT:"
        path = os.path.join(self.path, 'Histograms')
        accumulate_path = os.path.join(self.path, 'accumulate_data/features')

        try:
            len_of_code_book = self.len_of_code_book
        except AttributeError as e:
            print "temp histogram method not run. length fixed to 18"
            len_of_code_book = 18  # change this to check for the longest histgoram in the directory :(

        list_of_histograms =[ (recording, path, len_of_code_book) for recording in sorted(os.listdir(path)) ]
        parallel = 1
        if parallel:
            num_procs = mp.cpu_count()
            pool = mp.Pool(num_procs)
            chunk_size = int(np.ceil(len(os.listdir(path))/float(num_procs)))
            joint_results = pool.map(h.worker_padd, list_of_histograms, chunk_size)
            pool.close()
            pool.join()
        else: # for sequential debugging:
            joint_results = []
            for cnt, event in enumerate(list_of_histograms):
                print "adding to feature space: ", event[0]
                joint_results.append(h.worker_padd(event))

        labels = [lab for (lab, hist) in joint_results]
        f = open(accumulate_path + "/labels.p", "w")
        pickle.dump(labels, f)
        f.close()
        features = np.vstack([hist for (lab, hist) in joint_results])
        new_features = h.remove_low_instance_graphlets(self.path, features, low_instances)
        # for i in new_features:
        #     print ">>", i[:35]
        return True


    def learn_activities(self, singular_val_threshold=2.0, assign_clstr=0.1):
        tf_idf_scores = tfidf.get_tf_idf_scores(self.path)
        #print tf_idf_scores
        tfidf.get_svd_learn_clusters(self.path, tf_idf_scores, singular_val_threshold, assign_clstr)

        return True




if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")

    rerun = 0
    parallel = 0
    singular_val_threshold = 9
    assign_clstr = 0.01

    o = Offline_ActivityLearning(reduce_frame_rate=3, rerun_all=rerun)
    o.get_soma_objects()

    o.get_events()
    o.encode_qsrs(parallel)
    o.make_temp_histograms_sequentially()
    o.make_term_doc_matrix(low_instances=5)
    o.learn_activities(singular_val_threshold, assign_clstr)
