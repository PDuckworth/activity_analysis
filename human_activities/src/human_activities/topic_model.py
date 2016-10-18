#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import copy
import getpass
import time
import cPickle as pickle
import numpy as np
import scipy
import lda
import math
import matplotlib.pyplot as plt
from pylab import *

import utils as utils

#import investigating_prepare_LDAvis_func as pp
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.fixes import bincount
from sklearn.pipeline import make_pipeline
# from sklearn import metrics


def test_topic_model(n_iters):
    import lda.datasets
    X = lda.datasets.load_reuters()
    vocab = lda.datasets.load_reuters_vocab()
    titles = lda.datasets.load_reuters_titles()
    print "data:", type(X), X.shape, X.sum()
    print "codebook:", type(vocab), len(vocab)

    model = lda.LDA(n_topics=20, n_iter=n_iters, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 30
    for i, topic_dist in enumerate(topic_word):
        print i, topic_dist

        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print topic_words
        print type(topic_words)
        print('Topic {}: {}'.format(i, ' '.join( topic_words)))


def object_nodes(graph):
    object_nodes = []
    num_of_eps = 0
    for node in graph.vs():
        if node['node_type'] == 'object':
            if node['name'] not in ["hand", "torso"]:
                object_nodes.append(node['name'])
        if node['node_type'] == 'spatial_relation':
            num_of_eps+=1

    return object_nodes, num_of_eps

def learn_topic_model(X, vocab, graphlets, config, dbg=False):

    alpha = config['dirichlet_params']['alpha']
    eta = config['dirichlet_params']['eta']
    model = lda.LDA(n_topics=config['n_topics'], n_iter=config['n_iters'], random_state=1, alpha=alpha, eta=eta)

    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 30

    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)

    try:
        print "phi: %s. theta: %s. nd: %s. vocab: %s. Mw: %s" \
        %( model.topic_word_.shape, model.doc_topic_.shape, doc_lengths.shape, len(graphlets.keys()), len(feature_freq))
        data = {'topic_term_dists': model.topic_word_,
                'doc_topic_dists': model.doc_topic_,
                'doc_lengths': len(graphlets.keys()),
                'vocab': graphlets.keys(),
                'term_frequency': X}

        import pyLDAvis
        vis_data = pyLDAvis.prepare(model.topic_word_, model.doc_topic_, doc_lengths, graphlets.keys(), feature_freq)
        # vis_data = pp.prepare(model.topic_word_, model.doc_topic_, doc_lengths, graphlets.keys(), feature_freq)
        html_file = "../LDAvis/Learnt_Models/topic_model_" + id + ".html"
        pyLDAvis.save_html(vis_data, html_file)
        print "PyLDAVis ran. output: %s" % html_file

        """investigate the objects used in the topics"""
        print("\ntype(topic_word): {}".format(type(topic_word)))
        print("shape: {}".format(topic_word.shape))
        topics = {}
        for i, topic_dist in enumerate(topic_word):
            objs = []
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            #print('Topic {}: {}'.format(i, ' '.join( [repr(i) for i in topic_words] )))
            for j in [graphlets[k] for k in topic_words]:
                objs.extend(object_nodes(j)[0])
            topics[i] = objs
            if dbg:
                print('Topic {}: {}'.format(i, list(set(objs))))

    except ImportError:
        print "No module pyLDAvis. Cannot visualise topic model"

    """investigate the highly probably topics in each document"""
    doc_topic = model.doc_topic_
    # #Each document's most probable topic - don't have the UUIDs, so dont use this.
    # pred_labels = []
    # for n in range(doc_topic.shape[0]):
    #     if max(doc_topic[n]) > config['class_thresh']:
    #         topic_most_pr = doc_topic[n].argmax()
    #         pred_labels.append(topic_most_pr)

    return doc_topic, topic_word #, pred_labels


def investigate_features(dictionary_codebook):
    cnt = 0
    for i, j in dictionary_codebook.items():
        for k in object_nodes(j)[0]:
            if k in ["knee"]:
                cnt+=1
                print cnt,  i, object_nodes(j)


def create_codebook_images(codebook, im_path=None, dbg=False):
    for cnt, (g_name, g) in enumerate(codebook.items()):
        if dbg: print "\n", hash , g
        if cnt % 1000 is 0: print cnt
        # if not os.path.isfile(os.path.join(im_path, g_name+".png")):
        graphlet2dot(g, g_name, im_path)

def get_dic_codebook(code_book, graphlets, create_graphlet_images=False):

    dictionary_codebook = {}
    for hash, graph in zip(code_book, graphlets):
        g_name = "{:20.0f}".format(hash).lstrip()
        dictionary_codebook[g_name] = graph

    if create_graphlet_images:
        #investigate_features(dictionary_codebook)
        image_path = '/home/' + getpass.getuser() + '/Dropbox/Programming/Lucie/LDAvis/Learnt_Models/images'
        create_codebook_images(dictionary_codebook, image_path, dbg)
    return dictionary_codebook

def run_topic_model(accu_path, config):

    code_book, graphlets, data = utils.load_all_learning_files(accu_path)
    dictionary_codebook = {}
    try:
        import pyLDAvis
        dictionary_codebook = get_dic_codebook(code_book, graphlets, config['create_images'])
    except ImportError:
        print "No module pyLDAvis. Cannot visualise topic model"

    print "sum of all data:", data.shape, data.sum()
    vocab = [ "{:20.0f}".format(hash).lstrip() for hash in list(code_book) ]
    # print "vocab:", len(vocab)

    doc_topic, topic_word  = learn_topic_model(data, vocab, dictionary_codebook, config)
    print " per document topic proportions: ", doc_topic.shape
    print " per topic word distributions: ", topic_word.shape

    return doc_topic, topic_word

def dump_lda_output(path, doc_topic, topic_word):
    f = open(os.path.join(path, "doc_topic.p"), "w")
    pickle.dump(doc_topic, f)
    f.close()

    f = open(os.path.join(path, "topic_word.p"), "w")
    pickle.dump(topic_word, f)
    f.close()

    """Plot a graph for each topic word distribution (vocabulary)"""
    max_, min_ = 0, 100
    min_=100
    for i in topic_word:
        if max(i)>max_: max_ = max(i)
        if min(i)<min_: min_ = min(i)

    for i, vocabulary in enumerate(topic_word):
        title = 'Topic %s' % i
        utils.genome(path, vocabulary, [min_, max_], title)

    # f = open(os.path.join(path, "pred_labels.p"), "w")
    # pickle.dump(pred_labels, f)
    # f.close()


if __name__ == "__main__":

    n_iters = 1000
    all_topics = xrange(11,12)
    dbg = False
    create_images = False
    dirichlet_params = (0.5, 0.03)

    date = str(datetime.datetime.now().date())
    path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
    accu_path = os.path.join(path, 'Learning', 'accumulate_data', date, 'LDA')

    all_topics = [10]
    class_threshold = 0.3
    results = {}

    for n_topics in all_topics:
        print "\ntopics = %s. " % n_topics
        doc_topic, topic_word = run_topic_model(path, n_iters, n_topics,  create_images, dirichlet_params, class_threshold)

        dump_lda_output(accu_path, doc_topic, topic_word)
        # results[n_topics] = print_results(true_labels, pred_labels, n_topics)

    # print "\nRESULTS:"
    # for n_topics in class_thresholds:
    #     r = results[n_topicss]
    #     print "#Topics=%s (%s. LL=%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f"  % (r[0], r[1], r[8], r[2], r[3], r[4], r[5], r[6], r[7])
