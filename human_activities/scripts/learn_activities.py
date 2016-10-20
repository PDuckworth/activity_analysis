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
from soma2_msgs.msg import SOMA2Object
from soma_manager.srv import SOMA2QueryObjs
from mongodb_store.message_store import MessageStoreProxy
from skeleton_tracker.msg import skeleton_message
#from activity_data.msg import SkeletonComplete

import human_activities.create_events as ce
import human_activities.encode_qsrs as eq
import human_activities.histograms as h
import human_activities.tfidf as tfidf
import human_activities.topic_model as tm

class Offline_ActivityLearning(object):

    def __init__(self, soma_map="", soma_conf="", rerun_all=0):
        print "initialise activity learning action class"

        self.load_config()  # loads all the learning parameters from a config file
        self.make_filepaths()  # Define all file paths

        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
        self.date = str(datetime.datetime.now().date())

        self._message_store = rospy.get_param("~message_store", "people_skeleton")
        self._database = rospy.get_param("~database", "message_store")
        self._store_client = MessageStoreProxy(collection=self._message_store, database=self._database)

        self.query = {} #"number_of_detections":100}
        self.soma_map = soma_map
        self.soma_conf = soma_conf

        """If you want to rerun all the learning each time (not necessary)"""
        if rerun_all:
            """Removes the skeleton pose sequence event and encoded qsr world."""
            for f in os.listdir(self.qsr_path):
                os.remove(os.path.join(self.qsr_path, f))
            for f in os.listdir(self.events_path):
                os.remove(os.path.join(self.events_path, f))

    def load_config(self):
        """load a config file for all the learning parameters"""
        try:
            self.config = yaml.load(open(roslib.packages.get_pkg_dir('human_activities') + '/config/config.ini', 'r'))
            print "config loaded:", self.config.keys()

            return True
        except:
            print "no config file found in /human_activities/config/config.ini"
            return False

    def make_filepaths(self):
        self.events_path = os.path.join(self.path, 'Learning', 'Events')
        self.qsr_path = os.path.join(self.path, 'Learning', 'QSR_Worlds')
        self.hist_path = os.path.join(self.path, 'Learning', 'Histograms')
        self.accu_path = os.path.join(self.path, 'Learning', 'accumulate_data', self.date)
        if not os.path.isdir(self.accu_path): os.system('mkdir -p ' + self.accu_path)

        self.lsa_path = os.path.join(self.accu_path, "LSA")
        if not os.path.exists(self.lsa_path): os.makedirs(self.lsa_path)
        self.lda_path = os.path.join(self.accu_path, "LDA")
        if not os.path.exists(self.lda_path): os.makedirs(self.lda_path)

    def get_soma_rois(self):
        """Find all the ROIs and restrict objects using membership.
        """
        soma_map = "collect_data_map_cleaned"
        # soma_config = "test"
        # query = {"map":soma_map, "config":soma_config}
        all_rois = []
        ret = self.soma_roi_store.query(SOMA2ROIObject._type)
        for (roi, meta) in ret:
            if roi.map_name != soma_map: continue
            if roi.geotype != "Polygon": continue
            all_rois.append(roi)
        return all_rois

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
        ret =  self._store_client.query(query, skeleton_complete._type)   # ret is a list of all queried msgs, returned as tuples. (msg, meta)

        # for (complete_skel_msg, meta) in ret:
            # do stuff
        return ret

    def get_events(self):
        """
        Encodes each skeleton detection into an Event() class
        FILTERS run sequentially (all effect each other):
         - Frame rate reduction applied first.
         - Filter to qsrs last.
        """

        print "\ngetting new Events"
        path = os.path.join(self.path, 'no_consent')
        for d_cnt, date in sorted(enumerate(os.listdir(path))):

            if os.path.isdir(os.path.join(self.events_path, date)):
                print "%s already processed" % date
                continue

            directory = os.path.join(path, date)
            for recording in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, recording)):

                    # Can we reduce this list of objects using ROI information?
                    try:
                        use_objects = {}
                        for region, objects in self.soma_objects.items():
                            for ob, position in objects.items():
                                use_objects[ob] = position

                        ce.get_event(recording, directory, use_objects, self.config['events'])
                    except:
                        print "recording: %s in: %s is broken." %(recording, directory)
                else:
                    print "already processed: %s" % recording
        print "done."

    def encode_qsrs(self, parallel=0):
        """check for any events which are not QSRs yet"""
        print "\ncalculating new QSRs: %s" % self.config['qsrs']['which_qsr']

        list_of_events = []
        for date in sorted(os.listdir(self.events_path)):
            if os.path.isdir(os.path.join(self.qsr_path, date)):
                print "%s already processed" % date
                continue

            path = os.path.join(self.events_path, date)
            print " >", date
            for recording in sorted(os.listdir(path)):
                # region = "Kitchen"   #todo: remove this region
                if not os.path.isfile(os.path.join(self.qsr_path, recording)):
                    list_of_events.append((recording, path, self.soma_objects.values(), self.config['qsrs']))

        if len(list_of_events) > 0:
            if self.config['qsrs']['parallel']:
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

        self.len_of_code_book = h.create_temp_histograms(self.qsr_path, self.accu_path)
        return True

    def make_term_doc_matrix(self):
        """generate a term frequency matrix using the unique code words and the local histograms/graphlets"""
        print "\ngenerating term-frequency matrix:"

        try:
            len_of_code_book = self.len_of_code_book
        except AttributeError as e:
            print ">>> temp histogram method not run. exit()"
            sys.exit(1)  # change this to serch for the longest histgoram in the directory ??

        list_of_histograms = []
        for d_cnt, date in sorted(enumerate(os.listdir(self.hist_path))):
            directory = os.path.join(self.hist_path, date)
            print " >", date
            for recording in sorted(os.listdir(directory)):
                list_of_histograms.append((recording, directory, len_of_code_book))

        if self.config['hists']['parallel']:
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

        uuids = [uuid for (uuid, hist) in results]
        f = open(self.accu_path + "/list_of_uuids.p", "w")
        pickle.dump(uuids, f)
        f.close()

        # features = np.vstack(results)
        features = np.vstack([hist for (uuid, hist) in results])
        new_features = h.recreate_data_with_high_instance_graphlets(self.accu_path, features, self.config['hists']['low_instances'])
        return True

    def learn_lsa_activities(self):
        """run tf-idf and LSA on the term frequency matrix. """
        print "\nrunning tf-idf weighting, and LSA:"

        tf_idf_scores = tfidf.get_tf_idf_scores(self.accu_path)
        U, Sigma, VT = tfidf.get_svd_learn_clusters(self.accu_path, tf_idf_scores, self.config['lsa']['singular_val_threshold'], self.config['lsa']['assign_clstr'])
        tfidf.dump_lsa_output(self.lsa_path, (U, Sigma, VT))
        print "number of LSA activities learnt: %s. left: %s. right:%s" % (len(Sigma), U.shape, VT.shape)
        print "LSA - done."
        return True

    def learn_topic_model_activities(self):
        """learn a topic model using LDA. """
        print "\nLearning a topic model with LDA:"

        doc_topic, topic_word = tm.run_topic_model(self.accu_path, self.config['lda'])

        tm.dump_lda_output(self.lda_path, doc_topic, topic_word)
        print "Topic Modelling - done.\n"
        return True


if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")

    rerun = 0

    o = Offline_ActivityLearning(rerun_all=rerun)
    o.get_soma_objects()
    o.get_events()
    o.encode_qsrs()
    o.make_temp_histograms_sequentially()
    o.make_term_doc_matrix()
    o.learn_lsa_activities()
    o.learn_topic_model_activities()

    print "\n completed learning phase"
