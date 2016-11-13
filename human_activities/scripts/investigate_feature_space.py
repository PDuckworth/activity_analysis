#!/usr/bin/env python
__author__ = "Paul Duckworth"

import rospy
import os, sys
import argparse
import numpy as np
import cPickle as pickle
import getpass, datetime

if __name__ == "__main__":

    rospy.init_node("offline_feature_space")

    parser = argparse.ArgumentParser(description='Offline feature space Investigation')
    parser.add_argument('date', type=str, help='date of learned topic model')
    args = parser.parse_args()
    date = args.date
    rospy.loginfo("Offline Human Activity Investigation: %s" % date)

    path = '/home/' + getpass.getuser() + '/SkeletonDataset/Learning/accumulate_data'

    data_path = os.path.join(path, date)
    print "path load: %s " % data_path

    #load all the required data
    with open(data_path + "/feature_space.p", 'r') as f:
        data = pickle.load(f)

    print ">>", data.shape

    for i in data:
        print ">", i
