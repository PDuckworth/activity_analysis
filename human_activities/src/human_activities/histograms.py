#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import scipy
import numpy as np
import cPickle as pickle
import multiprocessing as mp
from create_events import *


def create_temp_histograms(path):
    """sequentially create a temporary histogram whilst
    generating the codebook from observations"""

    qsr_path = os.path.join(path, 'QSR_Worlds')
    acc_path = os.path.join(path, 'accumulate_data/features')
    for p in [qsr_path, acc_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    global_codebook = np.array([])
    all_graphlets = np.array([])

    print "HISTROGRAMS:"
    for e_cnt, event_file in sorted(enumerate(os.listdir(qsr_path))):
        e = load_e(qsr_path, event_file)

        print e_cnt, e.waypoint, e.uuid, len(e.qsr_object_frame.qstag.graphlets.histogram)

        try:
            if len(e.qsr_object_frame.qstag.graphlets.histogram) == 0: continue
            e.temp_histogram = np.array([0] * (global_codebook.shape[0]))
            #feature_spaces = [e.qsr_object_frame.qstag.graphlets, e.qsr_joint_frame.qstag.graphlets]
            feature_spaces = [e.qsr_object_frame.qstag.graphlets]

            for f in feature_spaces:
                for freq, hash in zip(f.histogram, f.code_book):
                    try:
                        ind = np.where(global_codebook == hash)[0][0]
                        e.temp_histogram[ind] += freq
                    # If the hash doesn't exist in the global codebook yet - add it
                    except IndexError:
                        global_codebook = np.append(global_codebook, hash)
                        e.temp_histogram = np.append(e.temp_histogram, freq)
                        all_graphlets = np.append(all_graphlets, f.graphlets[hash])
            # print global_codebook, e.temp_histogram, all_graphlets
            # print "t:", e.temp_histogram
            save_event(e, "Histograms")
        except:
            print "broken - check feature space"

    print "Final code book shape:", global_codebook.shape
    f = open(acc_path + "/code_book_all.p", "w")
    pickle.dump(global_codebook, f)
    f.close()

    f = open(acc_path + "/graphlets_all.p", "w")
    pickle.dump(all_graphlets, f)
    f.close()
    return global_codebook.shape[0]


def worker_padd(chunk):
    (event_file, histogram_directory, lenth_codebook) = chunk
    # print "    ", event_file

    e = load_e(histogram_directory, event_file)
    e.global_histogram = np.array([[0]*lenth_codebook])

    ind = len(e.temp_histogram)
    #Overlay the global histogram with the temp histogram
    e.global_histogram[0][:ind] = e.temp_histogram
    # save_event(e)
    return (e.waypoint, e.global_histogram[0])



def remove_low_instance_graphlets(path, feature_space=None, low_instance=1):
    """This invloves a lot of loading and saving of feature spaces.
    But essentially, it takes the feature space created over all events, and removes any
    feature that is not witnessed a minimum number of times (low_instance param).
    It then loads the code_book, and graphlets_book, to remove the features from there also
    (resaving with a modified filename)
    """

    try:
        remove_inds = np.where(feature_space.sum(axis=0) <= low_instance)[0]
    except AttributeError:
        with open(path + "accumulate_data/features/feature_space_all.p", 'r') as f:
            feature_space = pickle.load(f)
        remove_inds = np.where(feature_space.sum(axis=0) <= low_instance)[0]

    #load code book:
    with open(path + "accumulate_data/features/code_book_all.p", 'r') as f:
        code_book = pickle.load(f)

    #load graphlets
    with open(path + "accumulate_data/features/graphlets_all.p", 'r') as f:
        graphlets = pickle.load(f)

    for ind in sorted(list(remove_inds), reverse=True):
        feature_space = scipy.delete(feature_space, ind, 1)
        code_book = scipy.delete(code_book, ind, 0)
        graphlets = scipy.delete(graphlets, ind, 0)

    #save feature space
    f = open(path + "accumulate_data/features/feature_space.p", "w")
    pickle.dump(feature_space, f)
    f.close()

    #save code_book
    f = open(path + "accumulate_data/features/code_book.p", "w")
    pickle.dump(code_book, f)
    f.close()

    #save graphlets
    f = open(path + "accumulate_data/features/graphlets.p", "w")
    pickle.dump(graphlets, f)
    f.close()
    print "removed low (%s) instance graphlets" % low_instance
    print "new shape = ", feature_space.shape
    return feature_space
