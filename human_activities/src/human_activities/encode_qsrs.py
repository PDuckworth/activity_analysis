#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import cPickle as pickle
import numpy as np
import multiprocessing as mp

import utils as utils
import create_events as ce
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_qsr_trace import World_QSR_Trace
from qsrlib_utils.utils import merge_world_qsr_traces
from qsrlib_qstag.qstag import Activity_Graph
from qsrlib_qstag.utils import *


def worker_qsrs(chunk):
    (file_, path, soma_objects, config) = chunk

    e = utils.load_e(path, file_)

    dynamic_args = {}
    try:
        dynamic_args['argd'] = config['argd_args']
        dynamic_args['qtcbs'] = config['qtcbs_args']
        dynamic_args["qstag"] = {"params" : config['qstag_args']}
        dynamic_args['filters'] = {"median_filter": {"window": config['qsr_mean_window']}}  # This has been updated since ECAI paper.
    except KeyError:
        print "check argd, qtcbs, qstag parameters in config file"


    joint_types = {'head': 'head', 'torso': 'torso', 'left_hand': 'hand', 'right_hand': 'hand', 'left_knee': 'knee', 'right_knee': 'knee',
                   'left_shoulder': 'shoulder', 'right_shoulder': 'shoulder', 'head-torso': 'tpcc-plane'}
    object_types = joint_types.copy()
    add_objects = []
    for region, objects in ce.get_soma_objects().items():
        for o in objects:
            add_objects.append(o)
            try:
                generic_object = "_".join(o.split("_")[:-1])
                object_types[o] = generic_object
            except:
                print "didnt add:", object
    dynamic_args["qstag"]["object_types"] = object_types

    """1. CREATE QSRs FOR joints & Object """
    qsrlib = QSRlib()

    qsrs_for=[]
    for ob in objects:
        qsrs_for.append((str(ob), 'left_hand'))
        qsrs_for.append((str(ob), 'right_hand'))
        qsrs_for.append((str(ob), 'torso'))
    dynamic_args['argd']["qsrs_for"] = qsrs_for
    dynamic_args['qtcbs']["qsrs_for"] = qsrs_for

    req = QSRlib_Request_Message(config['which_qsr'], input_data=e.map_world, dynamic_args=dynamic_args)
    e.qsr_object_frame = qsrlib.request_qsrs(req_msg=req)

    # print ">", e.qsr_object_frame.qstag.graphlets.histogram
    # for i in e.qsr_object_frame.qstag.episodes:
    #     print i
    # sys.exit(1)
    """2. CREATE QSRs for joints - TPCC"""
    # print "TPCC: ",
    # # e.qsr_joint_frame = get_joint_frame_qsrs(file, e.camera_world, joint_types, dynamic_args)
    # qsrs_for = [('head', 'torso', ob) if ob not in ['head', 'torso'] and ob != 'head-torso' else () for ob in joint_types.keys()]
    # dynamic_args['tpcc'] = {"qsrs_for": qsrs_for}
    # dynamic_args["qstag"]["params"] = {"min_rows": 1, "max_rows": 2, "max_eps": 4}
    # qsrlib = QSRlib()
    # req = QSRlib_Request_Message(which_qsr="tpcc", input_data=e.camera_world, dynamic_args=dynamic_args)
    # e.qsr_joints_frame = qsrlib.request_qsrs(req_msg=req)
    # # pretty_print_world_qsr_trace("tpcc", e.qsr_joints_frame)
    # # print e.qsr_joints_frame.qstag.graphlets.histogram
    utils.save_event(e, "Learning/QSR_Worlds")

def call_qsrlib(in_path, dirs, parallel):

    for dir_ in dirs:
        list_of_events = []
        directory = os.path.join(in_path, dir_)
        for i in sorted(os.listdir(directory)):
            list_of_events.append((i, directory))

        if parallel:
            num_procs = mp.cpu_count()
            pool = mp.Pool(num_procs)
            chunk_size = int(np.ceil(len(os.listdir(directory))/float(num_procs)))
            pool.map(worker_qsrs, list_of_events, chunk_size)
            pool.close()
            pool.join()
        else:
            for cnt, i in enumerate(list_of_events):
                print "\n ", cnt, i
                worker_qsrs(i)


if __name__ == "__main__":
    """	Read events files,
    call QSRLib with parameters
    create QSR World Trace
    save event with QSR World Trace
    """

    ##DEFAULTS:
    path = '/home/' + getpass.getuser() + '/Datasets/Lucie_skeletons/Learning/Events'
    dates = [f for f in os.listdir(path)]
    call_qsrlib(path, dates, parallel=0)
