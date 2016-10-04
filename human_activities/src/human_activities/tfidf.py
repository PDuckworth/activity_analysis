#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import getpass
import time
import cPickle as pickle
import numpy as np
import math
import multiprocessing as mp
import itertools
import matplotlib.pyplot as plt

from create_events import *
import utils as utils

from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.fixes import bincount
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist, euclidean


def get_tf_idf_scores(path, input_data=None, vis=False):

    try:
        (code_book, graphlets, data) = input_data
    except:
        code_book, graphlets, data = utils.load_all_learning_files(path)

    """BINARY COUNTING OF FEATURES:"""
    feature_freq = (data != 0).sum(axis=0)  # TF: document_frequencies
    (N, f) = data.shape                     # Number of documents, and number of features
    print "nuber of documents = %s, number of features = %s " % (N, f)

    """
    ## Inverse document frequency scores
    ## LONG HAND
    # idf_scores=[]
    # for i in feature_freq:
    #     try:
    #         idf_scores.append(math.log((N /float(i))))
    #     except:
    #         idf_scores.append(0)
    """
    idf_scores = [(math.log((N / float(i)))) if i > 0 else 0 for i in feature_freq]

    tf_idf_scores = np.array([[]])
    for histogram in data:
        #freq = 1+math.log(freq) if freq>0 else 0  #log normalisation of Term Frequency
        foo = [idf_scores[cnt]*(math.log(1+freq)) for cnt, freq in enumerate(histogram)]
        try:
            tf_idf_scores = np.append(tf_idf_scores, np.array([foo]), axis=0)
        except ValueError:
            tf_idf_scores = np.array([foo])

    print "tf-idf shape:", tf_idf_scores.shape
    return tf_idf_scores






def get_svd_learn_clusters(path, data=None, sing_threshold=2.0, assign_clstr=0.1, vis=False):
    """First runs the decomposition for maximum number of singular values.
    Then reruns on a subset > than some value"""

    # with open(path + "Learning/accumulate_data/labels.p", 'r') as f:
    #     labels = pickle.load(f)

    with open(path + "Learning/accumulate_data/graphlets.p", 'r') as f:
        graphlets = pickle.load(f)

    (N, f) = data.shape
    all_components = min(N,f)
    U, Sigma, VT = randomized_svd(data, n_components=all_components, n_iter=5, random_state=None)

    # print "Sigma:", Sigma
    best_components = sum(Sigma > sing_threshold)
    U, Sigma, VT = randomized_svd(data, n_components=best_components, n_iter=5, random_state=None)

    # print "\npredicted classes:", [np.argmax(doc) for doc in U]
    pred_labels = [np.argmax(doc) if np.max(doc) > assign_clstr else 100 for doc in U]
    # print "predicted classes:", pred_labels

    graph_path = os.path.join(path, 'Learning', 'accumulate_data')
    utils.screeplot(graph_path, Sigma, all_components, vis)

    max_= 0
    min_=100
    for i in VT:
        if max(i)>max_: max_ = max(i)
        if min(i)<min_: min_ = min(i)

    print "\n%s Activities Learnt" % best_components
    activities_path = os.path.join(path, 'Learning', "accumulate_data", "activities")
    if not os.path.exists(activities_path):
        os.makedirs(activities_path)
        f = open(os.path.join(activities_path, "v_singular_mat.p"), "w")
        pickle.dump(VT, f)
        f.close()

    print "\nVT: "
    for i, vocabulary in enumerate(VT):
        print i, vocabulary

        if vis:
            for c, v in enumerate(vocabulary):
                if v > 0.1:
                    print "\n",c,  graphlets[c]

        utils.genome(graph_path, vocabulary, [min_, max_], i)
    return


if __name__ == "__main__":
    """	Load the feature space,
    (i.e. a histogram over the global codebook, for each recording)
    Maintain the labels, and graphlet iGraphs (for the learning/validation)
    """

    ##DEFAULTS:
    path = '/home/' + getpass.getuser() + '/SkeletonDataset/'

    tf_idf_scores = get_tf_idf_scores(path)
    get_svd_learn_clusters(path, tf_idf_scores, assign_clstr=0.5, vis=False)
