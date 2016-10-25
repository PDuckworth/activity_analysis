#!/usr/bin/env python
import sys, os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
import getpass, datetime
from sklearn import metrics

def print_results(true_labels, pred_labels, num_clusters):
    (h, c, v) =  metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)

    print "#Topics=%s (%s). v-measure: %0.3f. homo: %0.3f. comp: %0.3f. MI: %0.3f. NMI: %0.3f. Acc: %0.3f" \
      % (num_clusters, len(pred_labels), v, h, c,
        metrics.mutual_info_score(true_labels, pred_labels),
        metrics.normalized_mutual_info_score(true_labels, pred_labels),
        metrics.accuracy_score(true_labels, pred_labels))

def get_results():

    path = '/home/' + getpass.getuser() + '/SkeletonDataset/Learning/accumulate_data'
    date = str(datetime.datetime.now().date())
    # date = "2016-10-24"

    filename = os.path.join(path, date, "list_of_uuids.p")
    with open(filename, 'r') as f:
        uuids = pickle.load(f)
    gammas = np.loadtxt(os.path.join(path, date, "onlineLDA", 'gamma.dat'))
    #print "uuids: ",  uuids
    #print "gammas:", gammas

    ground_truth, pred_labels = [], []
    for dist, uuid in zip(gammas, uuids):
        # print uuid
        # print "_".join(uuid.split("_")[1:])
        # print dist
        act = "_".join(uuid.split("_")[1:])
        ground_truth.append(act)
        pred_labels.append(np.argmax(dist))

    #print "\n"
    #print "GT:", ground_truth
    #print "pred:", pred_labels
    #print "\n"

    print_results(ground_truth, pred_labels, 10)


if __name__ == "__main__":
    get_results()
