#!/usr/bin/env python
"""Analyse Episode store using Machine Learning techniques (offline)"""
__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

import os, sys, time
import yaml
import roslib
import getpass, datetime
import rospy
import numpy as np
import cPickle as pickle
import multiprocessing as mp
from soma2_msgs.msg import SOMA2Object   # this might be soma_msg in strands repo
from soma_manager.srv import SOMA2QueryObjs
from mongodb_store.message_store import MessageStoreProxy
from skeleton_tracker.msg import skeleton_message
#from activity_data.msg import SkeletonComplete

import human_activities.create_events as ce
import human_activities.encode_qsrs as eq
import human_activities.histograms as h
import human_activities.tfidf as tfidf
import topic_models as tm

class Offline_ActivityLearning(object):

    def __init__(self, soma_map="", soma_conf="", rerun_all=0, reduce_frame_rate=2, joints_mean_window=5, qsr_mean_window=3):
        print "initialise activity learning action class"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
        self.date = str(datetime.datetime.now().date())

        self._message_store = rospy.get_param("~message_store", "people_skeleton")
        self._database = rospy.get_param("~database", "message_store")
        self._store_client = MessageStoreProxy(collection=self._message_store, database=self._database)

        self.query = {} #"number_of_detections":100}
        self.soma_map = soma_map
        self.soma_conf = soma_conf

        """FILTERS all effect each other. Frame rate reduction applied first. Filter to qsrs last."""
        self.reduce_frame_rate = reduce_frame_rate
        self.joints_mean_window = joints_mean_window
        self.qsr_mean_window = qsr_mean_window
        self.events_path = os.path.join(self.path, 'Learning', 'Events')
        self.processed_path = os.path.join(self.path, 'Learning', 'QSR_Worlds')
        self.hist_path = os.path.join(self.path, 'Learning', 'Histograms')
        self.accu_path = os.path.join(self.path, 'Learning', 'accumulate_data')
        #self.load_config()

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
        """srv call to mongo and get the list of new objects and locations"""

        msg_store = MessageStoreProxy(database="soma2data", collection="soma2")
        objs = msg_store.query(SOMA2Object._type, message_query={"map_name":self.soma_map,"config":self.soma_conf})
        print "queried soma2 objects >> ", objs
        self.soma_objects = ce.get_soma_objects()
        print "hard coded objects >> ", [self.soma_objects[r].keys() for r in self.soma_objects.keys()]


    def get_skeletons_from_mongodb(self):
        """Query the database for the skeleton pose sequences"""
        query = {"date":self.date}
        print "query:", query
        ret =  self._store_client.query(skeleton_complete._type, query)   # ret is a list of all queried msgs, returned as tuples. (msg, meta)

        # for (complete_skel_msg, meta) in ret:
            # do stuff
        return ret

    def get_events(self):
        print "\ngetting new Events"
        path = os.path.join(self.path, 'no_consent')
        for d_cnt, date in sorted(enumerate(os.listdir(path))):

            if os.path.isdir(os.path.join(self.processed_path, date)):
                print "%s already processed" % date
                continue

            directory = os.path.join(path, date)
            for recording in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, recording)):
                    region = "Kitchen"
                    ce.get_event(recording, directory, self.soma_objects[region], self.reduce_frame_rate, self.joints_mean_window)
                else:
                    print "already processed: %s" % recording
        print "done."

    def encode_qsrs(self, parallel=0):
        """check for any events which are not QSRs yet"""
        print "\ncalculating new QSRs"

        list_of_events = []
        for date in sorted(os.listdir(self.events_path)):
            if os.path.isdir(os.path.join(self.processed_path, date)):
                print "%s already processed" % date
                continue

            path = os.path.join(self.events_path, date)
            print ">", path
            for recording in sorted(os.listdir(path)):
                region = "Kitchen"   #todo: remove this region
                if not os.path.isfile(os.path.join(self.processed_path, recording)):
                    list_of_events.append((recording, path, self.soma_objects[region], self.qsr_mean_window))

        if len(list_of_events) > 0:
            if parallel:
                num_procs = mp.cpu_count()
                pool = mp.Pool(num_procs)
                chunk_size = int(np.ceil(len(list_of_events)/float(num_procs)))
                pool.map(eq.worker_qsrs, list_of_events, chunk_size)
                pool.close()
                pool.join()
            else: # for sequential debugging:
                for cnt, event in enumerate(list_of_events):
                    print "encoding QSRs: ", event[0]
                    eq.worker_qsrs(event)
        print "done"

    def make_temp_histograms_sequentially(self):
        """find the length of the code book, i.e. all unique code words"""
        print "\nfind all unique code words"
        self.len_of_code_book = h.create_temp_histograms(self.path)
        return True

    def make_term_doc_matrix(self, parallel=0, low_instances=3):
        """generate a term frequency matrix using the unique code words and the local histograms/graphlets"""
        print "\ngenerating term-frequency matrix:"

        try:
            len_of_code_book = self.len_of_code_book
        except AttributeError as e:
            print ">>> temp histogram method not run. exit()"
            sys.exit(1)  # change this to serch for the longest histgoram in the directory :(

        list_of_histograms = []
        for d_cnt, date in sorted(enumerate(os.listdir(self.hist_path))):
            directory = os.path.join(self.hist_path, date)
            print directory
            for recording in sorted(os.listdir(directory)):
                list_of_histograms.append((recording, directory, len_of_code_book))

        if parallel:
            num_procs = mp.cpu_count()
            pool = mp.Pool(num_procs)
            chunk_size = int(np.ceil(len(list_of_histograms)/float(num_procs)))
            results = pool.map(h.worker_padd, list_of_histograms, chunk_size)
            pool.close()
            pool.join()
        else: # for sequential debugging:
            results = []
            for cnt, event in enumerate(list_of_histograms):
                print "adding to feature space: ", event[0]
                results.append(h.worker_padd(event))

        ## MIGHT WANT TO DUMP THE LOCATION HERE?
        uuids = [uuid for (uuid, hist) in results]
        f = open(self.accu_path + "/list_of_uuids.p", "w")
        pickle.dump(uuids, f)
        f.close()

        features = np.vstack([hist for (uuid, hist) in results])
        # features = np.vstack(results)
        new_features = h.recreate_data_with_high_instance_graphlets(self.path, features, low_instances)
        return True


    def learn_activities(self, singular_val_threshold=2.0, assign_clstr=0.1):
        tf_idf_scores = tfidf.get_tf_idf_scores(self.path)
        #print tf_idf_scores
        tfidf.get_svd_learn_clusters(self.path, tf_idf_scores, singular_val_threshold, assign_clstr)

        return True


    def learn_topic_model_activities(self, n_iters, create_images, dirichlet_params, class_threshold):

        n_topics = 10
        segmented_out, dictionary_codebook = tm.run_topic_model(self.path, n_iters, n_topics, dbg, create_images, dirichlet_params, class_threshold)
        (doc_topic, topic_word, code_book, pred_labels) = segmented_out
        return True


if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")

    rerun = 0
    parallel = 1
    singular_val_threshold = 10
    assign_clstr = 0.01

    n_iters = 10
    create_images = False
    dirichlet_params = (0.5, 0.03)
    class_threshold = 0.3

    o = Offline_ActivityLearning(reduce_frame_rate=3, rerun_all=rerun)
    o.get_soma_objects()
    o.get_events()
    o.encode_qsrs(parallel)
    o.make_temp_histograms_sequentially()
    o.make_term_doc_matrix(parallel, low_instances=1)
    o.learn_activities(singular_val_threshold, assign_clstr)
    o.learn_topic_model_activities(n_iters, create_images, dirichlet_params, class_threshold  )
