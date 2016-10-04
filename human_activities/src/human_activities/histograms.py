#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import scipy
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import utils as utils


def create_temp_histograms(qsr_path, accu_path):
    """sequentially create a temporary histogram whilst
    generating the codebook from observations"""

    global_codebook = np.array([])
    all_graphlets = np.array([])

    for d_cnt, dir_ in sorted(enumerate(os.listdir(qsr_path))):
        directory = os.path.join(qsr_path, dir_)
        print "dir: ", directory, d_cnt, dir_

        for e_cnt, event_file in sorted(enumerate(os.listdir(directory))):
            e = utils.load_e(directory, event_file)

            if len(e.qsr_object_frame.qstag.graphlets.histogram) == 0:
                print "removed:", e_cnt, event_file
                continue
            e.temp_histogram = np.array([0] * (global_codebook.shape[0]))

            print "  ", d_cnt, e_cnt, e.uuid, "len:", len(e.qsr_object_frame.qstag.graphlets.histogram) #, len(e.qsr_joints_frame.qstag.graphlets.histogram)
            # feature_spaces = [e.qsr_object_frame.qstag.graphlets, e.qsr_joints_frame.qstag.graphlets]
            feature_spaces = [e.qsr_object_frame.qstag.graphlets]#, e.qsr_joints_frame.qstag.graphlets]

            for cnt, f in enumerate(feature_spaces):
                for freq, hash in zip(f.histogram, f.code_book):
                    try:
                        ind = np.where(global_codebook == hash)[0][0]
                        e.temp_histogram[ind] += freq
                    # If the hash doesn't exist in the global codebook yet - add it
                    except IndexError:
                        global_codebook = np.append(global_codebook, hash)
                        e.temp_histogram = np.append(e.temp_histogram, freq)
                        all_graphlets = np.append(all_graphlets, f.graphlets[hash])
                        # print "\n>", hash, f.graphlets[hash]

            # print global_codebook, e.temp_histogram, all_graphlets
            utils.save_event(e, "Learning/Histograms")

    print "Code book shape:", global_codebook.shape
    f = open(os.path.join(accu_path, "code_book_all.p"), "w")
    pickle.dump(global_codebook, f)
    f.close()

    f = open(os.path.join(accu_path, "graphlets_all.p"), "w")
    pickle.dump(all_graphlets, f)
    f.close()

    return global_codebook.shape[0]

def worker_padd(chunk):
    (event_file, histogram_directory, lenth_codebook) = chunk
    # print "    ", event_file

    e = utils.load_e(histogram_directory, event_file)
    e.global_histogram = np.array([[0]*lenth_codebook])

    ind = len(e.temp_histogram)
    #Overlay the global histogram with the temp histogram
    e.global_histogram[0][:ind] = e.temp_histogram
    # utils.save_event(e)
    return (e.uuid, e.global_histogram[0])


def recreate_data_with_high_instance_graphlets(accu_path, feature_space=None, low_instance=1):
    """This invloves a lot of loading and saving.
    But essentially, it takes the feature space created over all events, and removes any
    feature that is not witnessed a minimum number of times (low_instance param).
    It then loads the code_book, and graphlets book, to remove the features from there also
    (resaving with a different name)
    """

    ## Number of rows with non zero element :
    keep_rows = np.where((feature_space != 0).sum(axis=0) > low_instance)[0]
    remove_inds = np.where((feature_space != 0).sum(axis=0) <= low_instance)[0]

    print "orig feature space: %s. remove: %s. new space: %s." % (len(feature_space.sum(axis=0)), len(remove_inds), len(keep_rows))

    #keep only the columns of the feature space which have more than low_instance number of occurances.
    selected_features = feature_space.T[keep_rows]
    new_feature_space = selected_features.T

    f = open(os.path.join(accu_path, "feature_space.p"), "w")
    pickle.dump(new_feature_space, f)
    f.close()
    print "new feature space shape: ", new_feature_space.shape

    # # Code Book (1d np array of hash values)
    with open(os.path.join(accu_path, "code_book_all.p"), 'r') as f:
        code_book = pickle.load(f)
    new_code_book = code_book[keep_rows]
    f = open(os.path.join(accu_path, "code_book.p"), "w")
    pickle.dump(new_code_book, f)
    f.close()
    print "  new code book len: ", len(new_code_book)

    # # Graphlets book (1d np array of igraphs)
    with open(os.path.join(accu_path, "graphlets_all.p"), 'r') as f:
        graphlets = pickle.load(f)
    new_graphlets = graphlets[keep_rows]
    f = open(os.path.join(accu_path, "graphlets.p"), "w")
    pickle.dump(new_graphlets, f)
    f.close()

    print "  new graphlet book len: ", len(new_graphlets)
    print "removed low (%s) instance graphlets" % low_instance
    print "shape = ", new_feature_space.shape
    return new_feature_space
