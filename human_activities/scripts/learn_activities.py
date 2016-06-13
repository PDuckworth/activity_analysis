#!/usr/bin/env python

"""Analyse Episode store using Machine Learning techniques (offline)"""

__author__ = "Paul Duckworth"
__copyright__ = "Copyright 2015, University of Leeds"

import rospy
import os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import argparse
import itertools
import getpass

import numpy as np
from scipy import spatial
import cPickle as pickle

from geometry_msgs.msg import Pose, Quaternion
from human_trajectory.msg import Trajectory, Trajectories
from soma_trajectory.srv import TrajectoryQuery, TrajectoryQueryRequest, TrajectoryQueryResponse
from soma_geospatial_store.geospatial_store import *

import novelTrajectories.config_utils as util
import relational_learner.obtain_trajectories as ot
import relational_learner.trajectory_analysis as th
import relational_learner.graphs_handler as gh

from novelTrajectories.traj_data_reader import *
from relational_learner.learningArea import *

def Mongodb_to_list(res):
    """Convert the EpisodeMsg into a list of episodes
       episodeMsg[]
           string obj1
           string obj1_type
           string obj2
           string obj2_type
           string spatial_relation
           int32 start_frame
           int32 end_frame
    """

    ep_list = []
    for i in res:
        ep = (str(i["obj1"]), str(i["obj1_type"]), str(i["obj2"]), \
              str(i["obj2_type"]), str(i["spatial_relation"]), \
              int(i["start_frame"]), int(i["end_frame"]))
        ep_list.append(ep)
    return ep_list


def keep_percentage_of_data(data, upper=0.0, lower=0.1, vis=False):
    a = list(reversed(sorted(data, key=data.get)))
    l = len(a)
    upper_ = int(l*upper)
    lower_ = int(l*lower)
    best_uuids = a[upper_:lower_]

    print "%s - %s percent selected (sorted by highest ratio). %s returned." % (upper*100, lower*100, len(best_uuids))
    query = ot.make_query(best_uuids)
    q = ot.query_trajectories(query)
    q.get_poses()

    if vis:
        d = data.values()
        plt.hist(d, bins=20)
        plt.xlim([0, max(d)])
        plt.title("Trajectory Displacement Histogram")
        plt.xlabel('displacement/#poses')
        plt.ylabel('count')
        plt.show()

    return q, best_uuids


def run_all(plotting=False, episode_store='relational_episodes'):
    (directories, config_path, input_data, date) = util.get_learning_config()
    (data_dir, qsr, trajs, activity_graph_dir, learning_area) = directories
    (soma_map, soma_config) = util.get_map_config(config_path)

    gs = GeoSpatialStoreProxy('geospatial_store', 'soma')
    msg_store = GeoSpatialStoreProxy('message_store', episode_store)

    # *******************************************************************#
    #                  Regions of Interest Knowledge                     #
    # *******************************************************************#
    rospy.loginfo('Getting Region Knowledge from roslog...')
    roi_knowledge, roi_temp_list = region_knowledge(soma_map, soma_config, \
                                                    sampling_rate=10, plot=plotting)

    # *******************************************************************#
    #               Filter trajectories which were deemed                #
    #          noise by using only people_trajectory msg store           #
    #           and filter on their distance covered per pose            #
    # *******************************************************************#
    # Filter on diplacement/poses ratio
    # Use if running online. Need to filter the message store trajectories before getting the corresponding episodes
    store='people_trajectory'
    all_data, ratios_dict = ot.filtered_trajectorys(just_ids=False, msg_store=store)
    q, best_uuids = keep_percentage_of_data(ratios_dict, upper=0.0, lower=0.1, vis=plotting)

    """
    # *******************************************************************#
    #              Analyse the shape of the Trajectories                 #
    #                                                                    #
    # *******************************************************************#
    rospy.loginfo('Generating Heatmap of trajectories...')
    uuid_pose_dict = { your_key: [your_key] for your_key in best_uuids}

    dt = th.Discretise_Trajectories(data=uuid_pose_dict, bin_size=0.2, filter_vel=1, verbose=False)
    dt.heatmap_run(vis=plotting, with_analysis=plotting)

    rospy.loginfo('Generating endpoint/destination points...')
    interest_points = dt.plot_polygon(vis=plotting, facecolor='green', alpha=0.4)
    print "interesting points include:\n", interest_points
    dt.markov_chain.display_and_save(layout='nx', view=True, path=trajs)

    """
    # *******************************************************************#
    #                  Obtain Episodes in ROI                            #
    # *******************************************************************#
    rospy.loginfo("0. Running ROI query from message_store")
    for roi in gs.roi_ids(soma_map, soma_config):
        str_roi = "roi_%s" % roi
        #if roi != '1': continue

        print '\nROI: ', gs.type_of_roi(roi, soma_map, soma_config), roi
        query = {"soma_roi_id": str(roi)}

        res = msg_store.find_projection(query, {"uuid":1, "trajectory":1, \
                "start_time":1, "episodes":1})
        #res = msg_store.find(query)
        all_episodes = {}
        trajectory_times = []
        cnt=0
        for cnt, trajectory in enumerate(res):
            # Get trajectory times of all trajectories. Not filtered on displacement etc.
            trajectory_times.append(trajectory["start_time"])
            if trajectory["uuid"] not in best_uuids: continue

            all_episodes[trajectory["uuid"]] = Mongodb_to_list(trajectory["episodes"])
            cnt+=1
        print "Total Number of Episodes queried = %s." % cnt
        print "Number of Trajectories after filtering by displacement/poses = %s." % len(all_episodes)

        if len(all_episodes) < 12:
            print "Not enough episodes in region %s to learn model." % roi
            continue

        # **************************************************************#
        #            Activity Graphs/Code_book/Histograms               #
        # **************************************************************#
        rospy.loginfo('Generating Activity Graphs')

        params, tag = gh.AG_setup(input_data, date, str_roi)
        print "INFO: ", params, tag, activity_graph_dir
        print "NOTE: Currently using Object ID in the graphlets"
        """
        1. Use specific object ID.
        2. Use object type info.
        3. Encode all objects as "object".
        """
        gh.generate_graph_data(all_episodes, activity_graph_dir, params, tag, obj_type = 1)

        # **************************************************************#
        #           Generate Feature Space from Histograms              #
        # **************************************************************#
        rospy.loginfo('Generating Feature Space')
        feature_space = gh.generate_feature_space(activity_graph_dir, tag)
        # (code_book, graphlet_book, X_source_U, X_uuids) = feature_space
        # print "code_book length = ", len(feature_space[0])

        # **************************************************************#
        #                    Create a similarty space                   #
        # **************************************************************#
        # rospy.loginfo('Create Similarity Space')

        # similarity_space = get_similarity_space(feature_space)
        # dictionary_of_similarity = {}

        # for i in similarity_space:
        #     key = np.sum(i)
        #     if key in dictionary_of_similarity:
        #         dictionary_of_similarity[key] += 1
        #     else:
        #         dictionary_of_similarity[key] = 1

                ## print "similarty space matches =" #Note: Reducing +ve histogram counts to 1
                ## for key, cnt in dictionary_of_similarity.items():
                ## print key, cnt

        # **************************************************************#
        #                    Learn a Clustering model                   #
        # **************************************************************#
        rospy.loginfo('Learning on Feature Space')
        params, tag = gh.AG_setup(input_data, date, str_roi)
        smartThing = Learning(f_space=feature_space, roi=str_roi, vis=False)

        ##PCA Analysis of Feature Space:
        pca, variable_scores = smartThing.pca_investigate_variables()
        top = 0.1  # Percentage of graphlets to analyse (make this automatic?)
        smartThing.pca_graphlets(pca, variable_scores, top)

        rospy.loginfo('Good ol k-Means')
        smartThing.kmeans()  # Can pass k, or auto selects min(penalty)
        smartThing.kmeans_cluster_radius()

        # *******************************************************************#
        #                    Temporal Analysis                               #
        # *******************************************************************#
        rospy.loginfo('Learning Temporal Measures')
        # print "traj times = ", trajectory_times, "\n"
        smartThing.time_analysis(trajectory_times, plot=plotting)
        # Future: Save a dictionary of IDs, timestamps and cluster composition for further analysis
        #smartThing.methods["temporal_list_of_uuids"] = trajectory_times

        # Add the region knowledge to smartThing - Future: make modula.
        try:
            smartThing.methods["roi_knowledge"] = roi_knowledge[roi]
            smartThing.methods["roi_temp_list"] = roi_temp_list[roi]
        except KeyError:
            smartThing.methods["roi_knowledge"] = 0
            smartThing.methods["roi_temp_list"] = [0] * 24

        smartThing.save(mongodb=True, msg_store="spatial_qsr_models")
        #smartThing.save(learning_area)
        print "Learnt models for: "
        for key in smartThing.methods:
            print "    ", key

    print "COMPLETED LEARNING PHASE"
    return


class Offline_ActivityLearning(object):

    def __init__(self):
        print "starting activity learning"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/SafeZone'
        self.mean_window = 5


    def encode_qsrs(self):

        for cnt, file_ in enumerate(self.path):
            print "encoding file: ", file_
            self.get_events()
            cnt+=1

        return True

    def make_histograms(self):
        return True

    def learn(self):
        return True




if __name__ == "__main__":
    rospy.init_node("Offline Activity Learner")

    o = Offline_ActivityLearning()

    o.encode_qsrs()
    o.make_histograms()
    o.learn()
