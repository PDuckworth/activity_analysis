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
from std_msgs.msg import String, Header
from soma_msgs.msg import SOMAObject, SOMAROIObject
from soma_manager.srv import *
from skeleton_tracker.msg import skeleton_message
#from activity_data.msg import SkeletonComplete
from shapely.geometry import Polygon, Point
from mongodb_store.message_store import MessageStoreProxy
import human_activities.create_events as ce
import human_activities.encode_qsrs as eq
import human_activities.histograms as h
import human_activities.tfidf as tfidf
import human_activities.topic_model as tm
import human_activities.onlineldavb as onlineldavb
import human_activities.utils as utils
from human_activities.msg import FloatList, oLDA

class Offline_ActivityLearning(object):

    def __init__(self, path, recordings):
        print "initialise activity learning action class"

        self.path = path
        self.recordings = recordings
        self.load_config()  # loads all the learning parameters from a config file
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
        self.make_init_filepaths()  # Define all file paths
        self.recordings = recordings

        # self.soma_map = rospy.get_param("~soma_map", "collect_data_map_cleaned")
        # self.soma_config = rospy.get_param("~soma_config", "test")  # objects
        # self.roi_config = rospy.get_param("~roi_config", "test")    # regions
        self.soma_map = self.config['events']['soma_map']
        self.soma_config = self.config['events']['soma_config']
        self.roi_config = self.config['events']['roi_config']

        # SOMA services
        rospy.loginfo("Wait for soma roi service")
        rospy.wait_for_service('/soma/query_rois')
        self.soma_query = rospy.ServiceProxy('/soma/query_rois',SOMAQueryROIs)
        rospy.loginfo("Done")

        self.soma_id_store = MessageStoreProxy(database='message_store', collection='soma_activity_ids_list')
        self.soma_store = MessageStoreProxy(database="somadata", collection="object")
        self.olda_msg_store = MessageStoreProxy(database='message_store', collection='activity_topic_models')

    def load_config(self):
        """load a config file for all the learning parameters"""
        try:
            self.config = yaml.load(open(roslib.packages.get_pkg_dir('human_activities') + '/config/config.ini', 'r'))
            print "config loaded:", self.config.keys()
        except:
            print "no config file found in /human_activities/config/config.ini"

    def make_init_filepaths(self):
        self.events_path = os.path.join(self.path, 'Learning', 'Events')
        self.qsr_path = os.path.join(self.path, 'Learning', 'QSR_Worlds')
        self.hist_path = os.path.join(self.path, 'Learning', 'Histograms')
        self.accu_path = os.path.join(self.path, 'Learning', 'accumulate_data')
        if not os.path.isdir(self.accu_path): os.system('mkdir -p ' + self.accu_path)

    def get_soma_rois(self):
        """Use only objects in the same ROI as the detection (if possible)
        """
        self.rois = {}
        print "self.roi_config: ", self.roi_config
        query = SOMAQueryROIsRequest(query_type=0, roiconfigs=[self.roi_config], returnmostrecent = True)

        for roi in self.soma_query(query).rois:
            if roi.map_name != self.soma_map: continue
            #if roi.config != self.roi_config: continue
            #if roi.geotype != "Polygon": continue
            k = roi.type + "_" + roi.id
            k = k.replace(" ", "")
            self.rois[k] = Polygon([ (p.position.x, p.position.y) for p in roi.posearray.poses])
        print "ROIs: ", self.rois.keys()

    def get_soma_objects(self):
        """srv call to mongo and get the list of new objects and locations"""
        ids = self.soma_id_store.query(String._type)
        ids = [id_[0].data for id_ in ids]
        print "SOMA IDs used to observe >> ", ids

        objs = self.soma_store.query(SOMAObject._type, message_query = {"id":{"$in": ids}})
        all_objects = {}
        self.soma_objects = {}
        for (ob, meta) in objs:
            if ob.id in ids:
                k = ob.type + "_" + ob.id
                p = ob.pose.position
                all_objects[k] = (p.x, p.y, p.z)
        #print "\nall_objects> ", all_objects.keys()

        for r, poly in self.rois.items():
            self.soma_objects[r] = {}
            for (ob, (x,y,z)) in all_objects.items():
                if poly.contains(Point([x, y])):
                    self.soma_objects[r][ob] = (x,y,z)
        print "objects >> ", [self.soma_objects[k].keys() for k in self.soma_objects.keys()]
        # print "\nobjects >> ", [self.soma_objects[r].keys() for r in self.soma_objects.keys()]

    # def get_skeletons_from_mongodb(self):
    #     """Query the database for the skeleton pose sequences"""

    # self._message_store = rospy.get_param("~message_store", "people_skeleton")
    # self._database = rospy.get_param("~database", "message_store")
    # self._store_client = MessageStoreProxy(collection=self._message_store, database=self._database)
    #     query = {"date":self.date}
    #     print "query:", query
    #     ret =  self._store_client.query(query, skeleton_complete._type)   # ret is a list of all queried msgs, returned as tuples. (msg, meta)
    #
    #     return ret

    def get_events(self, folder, uuid):
        """
        Encodes each skeleton detection into an Event() class
        FILTERS run sequentially (all effect each other):
         - Frame rate reduction applied first.
         - Filter to qsrs last.
        """
        path = os.path.join(self.path, self.recordings, folder)
        return ce.get_event(uuid, path, self.soma_objects, self.config['events'])
        # try:
        # use_objects = {}
        # for region, objects in self.soma_objects.items():
        #     for ob, position in objects.items():
        #         use_objects[ob] = position
        # except:
        #     print "recording: %s in: %s something is broken." %(uuid, path)

    def encode_qsrs_sequentially(self, folder, file_name):
        """very sequential version of encode qsrs"""
        path = os.path.join(self.events_path, folder)
        event = (file_name, path, self.config['qsrs'])
        eq.worker_qsrs(event)
        #for uuid in sorted(os.listdir(path), reverse=False):
        #    if rec in uuid:
        #        event = (uuid, path, self.config['qsrs'])
        #        eq.worker_qsrs(event)

    def encode_qsrs(self, folder):
        """check for any events which are not QSRs yet"""
        # print "\ncalculating new QSRs: %s" % self.config['qsrs']['which_qsr']
        path = os.path.join(self.events_path, folder)
        list_of_events = []
        for recording in sorted(os.listdir(path)):
            if not os.path.isfile(os.path.join(self.qsr_path, recording)):
                list_of_events.append((recording, path, self.config['qsrs']))

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
        print "qsrs - done"

    def make_histograms_online(self, this_run_date, last_olda):
        """Create a codebook for all previously seen unique code words"""

        print "\nfinding all unique code words"
        # codebook = np.array([])
        # graphlets = np.array([])
        #print type(last_olda.iGraphs)

        if last_olda.date == "":
            graphlets = np.array([])
        else:
            #print pickle.loads(last_olda.iGraphs)
            graphlets = np.array(pickle.loads(last_olda.iGraphs))

        codebook = np.array(last_olda.code_book)
        # graphlets = np.array(last_olda.iGraphs)

        #print "codebook:", codebook
        # print "load codebook from: %s " % last_run_date
        # prev_accu_path = os.path.join(self.accu_path, last_run_date)
        # with open(os.path.join(prev_accu_path, "code_book_all.p"), 'r') as f:
        #     codebook = pickle.load(f)
        # with open(os.path.join(prev_accu_path, "graphlets_all.p"), 'r') as f:
        #     graphlets = pickle.load(f)

        #qsr_path = os.path.join(self.qsr_path, date)
        codebook, graphlets, ret = h.create_temp_histograms(self.qsr_path, codebook, graphlets, self.batch)

        accu_path = os.path.join(self.accu_path, this_run_date)
        if not os.path.isdir(accu_path): os.system('mkdir -p ' + accu_path)

        print "current code book shape:", codebook.shape
        self.current_len_of_codebook = codebook.shape[0]

        last_olda.code_book = list(codebook)
        last_olda.iGraphs = pickle.dumps(graphlets)

        uuids, wordids, wordcts = ret

        f = open(accu_path + "/list_of_uuids.p", "w")
        pickle.dump(uuids, f)
        f.close()

        # new_features = h.recreate_data_with_high_instance_graphlets(accu_path, features, self.config['hists']['low_instances'])
        f = open(os.path.join(accu_path, "feature_space.p"), "w")
        pickle.dump((wordids, wordcts), f)
        f.close()

        # f = open(os.path.join(accu_path, "code_book_all.p"), "w")
        # pickle.dump(codebook, f)
        # f.close()
        #
        # f = open(os.path.join(accu_path, "graphlets_all.p"), "w")
        # pickle.dump(graphlets, f)
        # f.close()
        return last_olda, ret

    def make_term_doc_matrix(self, learning_date):
        """generate a term frequency matrix using the unique code words and the histograms/graphlets not yet processed"""
        print "\ngenerating term-frequency matrix: ",

        # try:
        #     len_of_code_book = self.current_len_of_codebook
        # except AttributeError as e:
        #     print ">>> unknown length of codebook. exit"
        #     return False

        list_of_histograms = []
        for rec in self.batch:
            (date,time,uuid) = rec.split("_")
            hist_path = os.path.join(self.hist_path, date)
            if os.path.isfile(os.path.join(hist_path, rec+".p")):
                list_of_histograms.append((rec, hist_path)) #, len_of_code_book))

        #for recording in sorted(os.listdir(hist_path)):
        #    list_of_histograms.append((recording, hist_path, len_of_code_book))
        # print "LEN OF LEARN ", len(list_of_histograms)
        if self.config['hists']['parallel']:
            print " -- in parallel."
            num_procs = mp.cpu_count()
            pool = mp.Pool(num_procs)
            chunk_size = int(np.ceil(len(list_of_histograms)/float(num_procs)))
            results = pool.map(h.worker_padd, list_of_histograms, chunk_size)
            pool.close()
            pool.join()
        else: # for sequential debugging:
            results = []
            print " -- sequentially."
            for cnt, event in enumerate(list_of_histograms):
                # print "building term-freq: ", event[0]
                results.append(h.worker_padd(event))

        # feature space is now a tuple of lists: (wordids, feature_counts) - wont work with LSA
        # feature_space = np.vstack([rets for (uuid, rets) in results])
        wordids = [ids for (uuid, (ids, cts)) in results]
        wordcts = [cts for (uuid, (ids, cts)) in results]
        uuids = [uuid for (uuid, rets) in results]

        feature_space = (wordids, wordcts)

        accu_path = os.path.join(self.accu_path, learning_date)

        f = open(accu_path + "/list_of_uuids.p", "w")
        pickle.dump(uuids, f)
        f.close()

        # new_features = h.recreate_data_with_high_instance_graphlets(accu_path, features, self.config['hists']['low_instances'])
        f = open(os.path.join(accu_path, "feature_space.p"), "w")
        pickle.dump(feature_space, f)
        f.close()
        return uuids, feature_space

    def learn_lsa_activities(self):
        """run tf-idf and LSA on the term frequency matrix. """
        print "\nrunning tf-idf weighting, and LSA:"

        accu_path = os.path.join(self.accu_path, self.date)
        lsa_path = os.path.join(accu_path, "LSA")
        if not os.path.exists(lsa_path): os.makedirs(lsa_path)

        tf_idf_scores = tfidf.get_tf_idf_scores(accu_path)
        U, Sigma, VT = tfidf.get_svd_learn_clusters(accu_path, tf_idf_scores, self.config['lsa']['singular_val_threshold'], self.config['lsa']['assign_clstr'])
        tfidf.dump_lsa_output(lsa_path, (U, Sigma, VT))
        print "number of LSA activities learnt: %s. left: %s. right:%s" % (len(Sigma), U.shape, VT.shape)
        print "LSA - done."
        return True

    def learn_topic_model_activities(self):
        """learn a topic model using LDA. """
        print "\nLearning a topic model with LDA:"

        accu_path = os.path.join(self.accu_path, self.date)
        lda_path = os.path.join(accu_path, "LDA")
        if not os.path.exists(lda_path): os.makedirs(lda_path)

        doc_topic, topic_word = tm.run_topic_model(accu_path, self.config['lda'])

        tm.dump_lda_output(lda_path, doc_topic, topic_word)
        print "Topic Modelling - done.\n"
        return True

    def online_lda_activities(self, date_folder, feature_space, msg):
        """learn online LDA topic model.
        note: # feature space is now a tuple of lists: (wordids, feature_counts)
        """
        print "\nLearning a topic model distributions with online LDA:"

        #olda = self.msg_to_olda(last_olda_ret)
        codebook = np.array(msg.code_book)
        accu_path = os.path.join(self.accu_path, date_folder)
        lda_path = os.path.join(accu_path, "oLDA")
        if not os.path.exists(lda_path): os.makedirs(lda_path)

        #load all the required data
        # code_book, graphlets, feature_space = utils.load_learning_files_all(accu_path)
        # with open(accu_path + "/feature_space.p", 'r') as f:
        #     feature_space = pickle.load(f)

        #make the oLDA object class
        #vocab = msg.code_book
        D = 1000
        if msg.date == "":
            print "initialise olda:"
            # The number of documents to analyze each iteration
            batchsize = len(feature_space)
            # The total number of documents (or an estimate of all docs)
            # The number of topics
            K = self.config['olda']['n_topics']
            olda = onlineldavb.OnlineLDA(codebook, K, D, 1./K, 1./K, 1., 0.7, 0)
        else:
            olda = onlineldavb.OnlineLDA(codebook, msg.K, msg.D, msg.alpha, msg.eta, msg.tau0, msg.kappa, msg.updatect)
            print "lamda has shape:", olda._lambda.shape
            print "lamda needs shape:", olda._lambda.shape[0], len(codebook)
            olda.add_new_features(len(codebook))

            # print "load oLDA from: ", last_run_date
            # prev_lda_path = os.path.join(self.accu_path, last_run_date, "oLDA")
            # # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
            # if last_run_date == "":
            #     print "initialise olda:"
            #     olda = onlineldavb.OnlineLDA(code_book, K, D, 1./K, 1./K, 1., 0.7)
            # else:
            #     #load the previous OLDA class
            #     with open(prev_lda_path + "/olda.p", 'r') as f:
            #         olda = pickle.load(f)

        wordids=feature_space[0]
        wordcts=feature_space[1]

        # print ">", wordids
        # print ">", wordcts

        # feature_counts = np.array([])
        # for cnt, v in enumerate(feature_space):
        #     # print "cnt: ", cnt
        #     nonzeros=np.nonzero(v)
        #     available_features=nonzeros
        #     wordids.append(available_features)
        #     feature_counts=v[nonzeros]
        #     wordcts.append(feature_counts)
        # print "avail features %s, feature_cnts: %s" %(available_features, feature_counts)

        # print "wordids %s, \nwordcts: %s" %(wordids, wordcts)
        (gamma, bound) = olda.update_lambda(wordids, wordcts)

        # Compute an estimate of held-out perplexity
        perwordbound = bound * wordcts[-1].shape[0] / (D * sum(map(sum, wordcts)))
        print 'date: %s:  rho_t = %f,  held-out perplexity estimate = %f. LDA - Done\n' % \
            (date_folder, olda._rhot, np.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.

        np.savetxt(lda_path + '/lambda.dat', olda._lambda)
        np.savetxt(lda_path + '/gamma.dat', gamma)

        lambda_msg = [FloatList(data = list(l)) for l in olda._lambda]

        new_msg = oLDA(K=olda._K, W=olda._W, alpha=olda._alpha, eta=olda._eta, tau0=olda._tau0, kappa=olda._kappa, updatect=olda._updatect)
        new_msg.type = "oLDA"
        new_msg.date =str(datetime.datetime.now().date())
        new_msg.time =str(datetime.datetime.now().time())
        new_msg.lambda_ = lambda_msg
        new_msg.code_book = msg.code_book
        new_msg.iGraphs = msg.iGraphs

        f = open(lda_path + "/olda.p", "w")
        pickle.dump(olda, f)
        f.close()
        return new_msg, gamma

if __name__ == "__main__":
    #rospy.init_node("Offline Activity Learner")
    o = Offline_ActivityLearning()

    o.self.directories_to_learn_from()
    o.get_soma_objects()
    o.get_events()
    o.encode_qsrs()
    o.make_temp_histograms_online()
    # o.make_term_doc_matrix()
    # o.learn_lsa_activities()
    # o.learn_topic_model_activities()

    print "\n completed learning phase"
