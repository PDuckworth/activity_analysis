#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import time
import cPickle as pickle
import numpy as np
import multiprocessing as mp
from create_events import *

from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
from qsrlib_io.world_qsr_trace import World_QSR_Trace
from qsrlib_utils.utils import merge_world_qsr_traces
from qsrlib_qstag.qstag import Activity_Graph
from qsrlib_qstag.utils import *


def get_map_frame_qsrs(file, world_trace, dynamic_args):
    """create QSRs between the person and the robot in map frame"""
    qsrs_for = [('robot', 'torso')]
    dynamic_args['qtcbs'] = {"qsrs_for": qsrs_for, "quantisation_factor": 0.0, "validate": False, "no_collapse": True}
    dynamic_args["qstag"] = {"params": {"min_rows": 1, "max_rows": 1, "max_eps": 3}}

    qsrlib = QSRlib()
    req = QSRlib_Request_Message(which_qsr="qtcbs", input_data=world_trace, dynamic_args=dynamic_args)
    qsr_map_frame = qsrlib.request_qsrs(req_msg=req)
    # print "    ", file, "episodes = "
    # for i in qsr_map_frame.qstag.episodes:
    #     print i
    return qsr_map_frame

def get_object_frame_qsrs(file, world_trace, objects, joint_types, dynamic_args):
    """create QSRs between the person's joints and the soma objects in map frame"""
    qsrs_for=[]
    for ob in objects:
        qsrs_for.append((str(ob), 'left_hand'))
        qsrs_for.append((str(ob), 'right_hand'))
        #qsrs_for.append((str(ob), 'torso'))

    dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.5, 'Near': 0.75,  'Medium': 1.5, 'Ignore': 10}}
    # dynamic_args['argd'] = {"qsrs_for": qsrs_for, "qsr_relations_and_values": {'Touch': 0.2, 'Ignore': 10}}
    dynamic_args['qtcbs'] = {"qsrs_for": qsrs_for, "quantisation_factor": 0.01, "validate": False, "no_collapse": True} # Quant factor is effected by filters to frame rate
    dynamic_args["qstag"] = {"object_types": joint_types, "params": {"min_rows": 1, "max_rows": 1, "max_eps": 2}}

    qsrlib = QSRlib()

    req = QSRlib_Request_Message(which_qsr=["argd", "qtcbs"], input_data=world_trace, dynamic_args=dynamic_args)
    # req = QSRlib_Request_Message(which_qsr="argd", input_data=world_trace, dynamic_args=dynamic_args)
    qsr_object_frame = qsrlib.request_qsrs(req_msg=req)

    for ep in qsr_object_frame.qstag.episodes:
        print ep

    # for cnt, h in  zip(qsr_object_frame.qstag.graphlets.histogram, qsr_object_frame.qstag.graphlets.code_book):
    #     print cnt, h#, qsr_object_frame.qstag.graphlets.graphlets[h]
    print ""
    return qsr_object_frame

def get_joint_frame_qsrs(file, world_trace, joint_types, dynamic_args):

    qsrs_for = [('head', 'torso', ob) if ob not in ['head', 'torso'] and ob != 'head-torso' else () for ob in joint_types.keys()]
    dynamic_args['tpcc'] = {"qsrs_for": qsrs_for}
    dynamic_args["qstag"] = {"object_types": joint_types, "params": {"min_rows": 1, "max_rows": 1, "max_eps": 3}}

    qsrlib = QSRlib()
    req = QSRlib_Request_Message(which_qsr="tpcc", input_data=world_trace, dynamic_args=dynamic_args)
    qsr_joints_frame = qsrlib.request_qsrs(req_msg=req)

    # for i in qsr_joints_frame.qstag.episodes:
    #     print i
    return qsr_joints_frame


def worker_qsrs(chunk):
    (file_, path, soma_objects, qsr_mean_window) = chunk
    e = load_e(path, file_)

    # joint_types = {'head' : 'head', 'neck' : 'neck', 'torso': 'torso','left_foot' : 'foot', 'right_foot' : 'foot',
    #  'left_shoulder' : 'shoulder', 'right_shoulder' : 'shoulder', 'left_hand' : 'hand', 'right_hand' : 'hand',
    #  'left_knee' : 'knee', 'right_knee': 'knee',  'right_elbow' : 'elbow', 'left_elbow' : 'elbow',  'right_hip' : 'hip', 'left_hip': 'hip'}
    # all_joints = ['head', 'neck', 'torso', 'left_foot', 'right_foot', 'left_shoulder', 'right_shoulder', 'left_hand', 'right_hand',
    # 'left_knee', 'right_knee',  'right_elbow', 'left_elbow',  'right_hip', 'left_hip']

    dynamic_args = {}
    dynamic_args['filters'] = {"median_filter": {"window": qsr_mean_window}}

    # # Robot - Person QTC Features
    # e.qsr_map_frame = get_map_frame_qsrs(file_, e.map_world, dynamic_args)

    #joint_types = {'head': 'head', 'torso': 'torso', 'left_hand': 'hand', 'right_hand': 'hand', 'left_knee': 'knee', 'right_knee': 'knee',
    #               'left_shoulder': 'shoulder', 'right_shoulder': 'shoulder', 'head-torso': 'tpcc-plane'}
    joint_types = {'left_hand': 'hand', 'right_hand': 'hand',  'head-torso': 'tpcc-plane'}

    joint_types_plus_objects = joint_types.copy()
    for object in soma_objects:
        generic_object = "_".join(object.split("_")[:-1])
        joint_types_plus_objects[object] = generic_object

    # # Key joints to Objects QSRs
    e.qsr_object_frame = get_object_frame_qsrs(file_, e.map_world, soma_objects, joint_types_plus_objects, dynamic_args)

    # # Person Joints TPCC Features
    #e.qsr_joint_frame = get_joint_frame_qsrs(file_, e.camera_world, joint_types, dynamic_args)
    save_event(e, "QSR_Worlds")


if __name__ == "__main__":
    """	Read events files,
    call QSRLib with parameters
    create QSR World Trace
    save event with QSR World Trace
    """

    ##DEFAULTS:
    path = '/home/' + getpass.getuser() + '/Datasets/Lucie_skeletons/Events'
    dirs = [f for f in os.listdir(path)]

    # dirs = [ '2016-04-05_ki', '2016-04-11_vi', '2016-04-08_vi', '2016-04-07_vi', '2016-04-05_me', '2016-04-06_ki']
    call_qsrlib(path, dirs)
