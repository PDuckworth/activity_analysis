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

from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.fixes import bincount
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist, euclidean


def load_all_files(path, which_features = "features"):
    print "loading files..."

    with open(path + "accumulate_data/features/code_book.p", 'r') as f:
        code_book = pickle.load(f)

    with open(path + "accumulate_data/features/graphlets.p", 'r') as f:
        graphlets = pickle.load(f)

    with open(path + "accumulate_data/features/feature_space.p", 'r') as f:
        data = pickle.load(f)
    return code_book, graphlets, data


def get_tf_idf_scores(path, input_data=None, vis=False):

    try:
        (code_book, graphlets, data) = input_data
    except:
        code_book, graphlets, data = load_all_files(path)

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

    print "new shape:", tf_idf_scores.shape
    return tf_idf_scores


def genome(path, data1, yrange, it):
    t = np.arange(0.0, len(data1), 1)
    #print "data", min(data1), max(data1), sum(data1)/float(len(data1))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlim([-1,65])
    ax1.vlines(t, [0], data1)
    ax1.set_xlabel('code words', fontsize=20)
    ax1.set_ylim(yrange)
    title = 'Latent Concept %s' % it
    ax1.set_title(title, fontsize=25)
    ax1.grid(True)
    filename = path + "accumulate_data/graphs/genome_%s.png" %it
    print filename
    fig.savefig(filename, bbox_inches='tight')
    plt.show()


def screeplot(sigma, comps, div=2):
    y = sigma
    x = np.arange(len(y)) + 1

    plt.subplot(2, 1, 1)
    plt.plot(x, y, "o-", ms=2)
    xticks = ["Comp." + str(i) if i % div == 0 else "" for i in x]

    plt.xticks(x, xticks, rotation=45, fontsize=20)

    # plt.yticks([0, .25, .5, .75, 1], fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("Variance", fontsize=20)
    plt.xlim([0, len(y)])
    plt.title("Plot of the variance of each Singular component", fontsize=25)
    plt.axvspan(10, 11, facecolor='g', alpha=0.5)
    plt.show()

def get_svd_learn_clusters(path, data=None, sing_threshold=2.0, assign_clstr=0.1):
    """First runs the decomposition for maximum number of singular values.
    Then reruns on a subset > than some value"""

    with open(path + "accumulate_data/features/labels.p", 'r') as f:
        labels = pickle.load(f)

    with open(path + "accumulate_data/features/graphlets.p", 'r') as f:
        graphlets = pickle.load(f)

    (N, f) = data.shape
    all_components = min(N,f)
    U, Sigma, VT = randomized_svd(data, n_components=all_components, n_iter=5, random_state=None)

    print "Sigma:", Sigma
    best_components = sum(Sigma > sing_threshold)
    U, Sigma, VT = randomized_svd(data, n_components=best_components, n_iter=5, random_state=None)
    # print "U:"
    # for i in U:
    #     print i[:best_components]

    print "\npredicted classes:", [np.argmax(doc) for doc in U]
    pred_labels = [np.argmax(doc) if np.max(doc) > assign_clstr else 100 for doc in U]
    print "predicted classes:", pred_labels

    for cluster in range(best_components)+[100]:
        cluster_labels = []
        inds = [1 if p == cluster else 0 for p in pred_labels]
        for cnt, i in enumerate(inds):
            if i is 1:
                cluster_labels.append(labels[cnt])
        print "\ncluster %s: %s instances" % (cluster, len(cluster_labels))
        print cluster_labels

    # screeplot(Sigma, all_components)

    #for i, j in itertools.combinations(VT, 2):
    max_= 0
    min_=100
    for i in VT:
        if max(i)>max_: max_ = max(i)
        if min(i)<min_: min_ = min(i)

    for i, vocabulary in enumerate(VT):
        print i, vocabulary

        for c, v in enumerate(vocabulary):
            if v > 0.1:
                print "\n",c,  graphlets[c]

        genome(path, vocabulary, [min_, max_], i)

    print "\n%s Activities Learnt" % best_components
    activities_path = os.path.join(path, "accumulate_data", "activities")
    if not os.path.exists(activities_path):
        os.makedirs(activities_path)
    f = open(os.path.join(activities_path, "v_singular_mat.p"), "w")
    pickle.dump(VT, f)
    f.close()


if __name__ == "__main__":
    """	Load the feature space,
    (i.e. a histogram over the global codebook, for each recording)
    Maintain the labels, and graphlet iGraphs (for the learning/validation)
    """

    ##DEFAULTS:
    path = '/home/' + getpass.getuser() + '/SkeletonDataset/SafeZone'

    tf_idf_scores = get_tf_idf_scores(path)
    print tf_idf_scores
    get_svd_learn_clusters(path, tf_idf_scores)
