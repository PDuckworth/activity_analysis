#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import cPickle as pickle
import itertools
import multiprocessing as mp
import numpy as np
import getpass
import bisect
import math
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from qsrlib_io.world_trace import Object_State, World_Trace
# from plot_events import *
# from learn_qsrs import *

def save_event(e, loc=None):
    """Save the event into an Events folder"""
    p = e.dir.split('/')
    time = p[-2].split('_')[0]
    folder_date = p[-3]
    file_name = e.label + "_" + repr(e.start_frame) + "_" + repr(e.end_frame) + "_" + time
    path = '/'.join(p[:-4]) + '/' + loc + '/' + folder_date

    if not os.path.isdir(path):
        os.system('mkdir -p ' + path)
    f = open(os.path.join(path, file_name+".p"), "w")
    pickle.dump(e, f, 2)
    f.close()

def load_e(directory, event_file):
    """Loads an event file along with exception raise msgs"""

    try:
        file = directory + "/" + event_file
        with open(file, 'r') as f:
            e = pickle.load(f)
        return e

    except (EOFError, ValueError, TypeError), error:
        print "Load Error: ", error, directory, event_file
        return None


class event(object):
    """Event objt"""
    def __init__(self, uuid=None, label=None, dir=None, start=None, end=None, region=None, region_id=None):

        self.uuid = uuid
        self.label = label
        self.start_frame = start
        self.end_frame = end
        self.region = region
        self.region_id = region_id
        self.dir = dir

        self.sorted_timestamps = []
        self.sorted_ros_timestamps = []
        self.bad_timepoints = {}        #use for filtering
        self.skeleton_data = {}         #type: dict[timepoint][joint_id]= (x, y, z, x2d, y2d)
        self.map_frame_data = {}        #type: dict[timepoint][joint_id]= (x, y, z, x2d, y2d)
        self.robot_data = {}            #type: dict[timepoint][joint_id]= ((x, y, z), (roll, pitch, yaw))
        # self.world = World_Trace()

    def apply_mean_filter(self, window_length=3):
        """Once obtained the joint x,y,z coords.
        Apply a median filter over a temporal window to smooth the joint positions.
        Whilst doing this, create a world Trace object"""

        joints, ob_states = {}, {}
        world = World_Trace()
        window = {}
        filtered_cnt = 0

        for t in self.sorted_timestamps:
            joints[t] = {}

            for joint_id, (x,y,z) in self.skeleton_data[t].items():
                # print "jointID=", joint_id, (x,y,z)
                try:
                    window[joint_id].pop(0)
                except (IndexError, KeyError):
                    window[joint_id] = []

                window[joint_id].append((float(x), float(y), float(z)))
                avg_x, avg_y, avg_z = 0, 0, 0
                for l, point in enumerate(window[joint_id]):
                    avg_x += point[0]
                    avg_y += point[1]
                    avg_z += point[2]

                x,y,z = avg_x/float(l+1), avg_y/float(l+1), avg_z/float(l+1)
                joints[t][joint_id] = (x, y, z)

                #Create a QSRLib format list of Object States (for each joint id)
                if joint_id not in ob_states.keys():
                    ob_states[joint_id] = [Object_State(name=joint_id, timestamp=filtered_cnt, x=x, y=y, z=z)]
                else:
                    ob_states[joint_id].append(Object_State(name=joint_id, timestamp=filtered_cnt, x=x, y=y, z=z))
            filtered_cnt+=1

        # #Add all the joint IDs into the World Trace
        for joint_id, obj in ob_states.items():
            world.add_object_state_series(obj)

        self.filtered_skeleton_data = joints
        self.camera_world = world


    def get_world_frame_trace(self, world_objects):
        """Accepts a dictionary of world (soma like) objects.
        Adds the position of the object at each timepoint into the World Trace"""

        ob_states={}
        world = World_Trace()
        for t in self.sorted_timestamps:
            #Joints:
            for joint_id, (x, y, z) in self.map_frame_data[t].items():
                if joint_id not in ob_states.keys():
                    ob_states[joint_id] = [Object_State(name=joint_id, timestamp=t, x=x, y=y, z=z)]
                else:
                    ob_states[joint_id].append(Object_State(name=joint_id, timestamp=t, x=x, y=y, z=z))

            # SOMA objects
            for object, (x,y,z) in world_objects.items():
                if object not in ob_states.keys():
                    ob_states[object] = [Object_State(name=str(object), timestamp=t, x=x, y=y, z=z)]
                else:
                    ob_states[object].append(Object_State(name=str(object), timestamp=t, x=x, y=y, z=z))

            # Robot's position
            (x,y,z) = self.robot_data[t][0]
            if 'robot' not in ob_states.keys():
                ob_states['robot'] = [Object_State(name='robot', timestamp=t, x=x, y=y, z=z)]
            else:
                ob_states['robot'].append(Object_State(name='robot', timestamp=t, x=x, y=y, z=z))

        for object_state in ob_states.values():
            world.add_object_state_series(object_state)

        self.map_world = world


def get_immediate_dir(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir,name))]

def get_events(dir, mean_window):
    """From the skeleton detections, get the data into an event class """

    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5

    print "getting events from: %s" % dir
    dirs = get_immediate_dir(dir)
    meta_cnt = 0

    for d in sorted(dirs):
        d1 = dir+'/'+d+'/'
        d_sk = d1+'skeleton/'
        d_robot = d1+'robot/'
        uuid = d.split('_')[1]

        # Get the event labels
        labels = {}
        if os.path.isfile(os.path.join(d1, 'labels.txt')):
            f1 = open(os.path.join(d1, 'labels.txt'), 'r')
            for count, line in enumerate(f1):
                line_ = line.split('\n')[0].split(':')
                if line_[0] == 'label':
                    start, end = line_[2].split(',')
                    labels[line_[1]] = (start, end)

        # Get the robot's region location
        if os.path.isfile(os.path.join(d1, 'meta.txt')):
            f1 = open(os.path.join(d1, 'meta.txt'), 'r')
            for count, line in enumerate(f1):
                if count == 0: region_id = line.split('\n')[0].split(':')[1]
                elif count == 1: region = line.split('\n')[0].split(':')[1]
        else:
            region="unknown"
            region_id = 0

        try:
            something = region
        except UnboundLocalError:
            meta_cnt+=1
            region = "Kitchen"
            region_id = 1

        for event_label, (start_frame, end_frame) in labels.items():
            print "E:", event_label, (start_frame, end_frame)

            if event_label == "making_tea": continue

            e = event(uuid, event_label, d1, int(start_frame), int(end_frame), region, region_id)

            sk_files = [f for f in os.listdir(d_sk) if os.path.isfile(os.path.join(d_sk, f))]
            for file in sorted(sk_files):
                frame = int(file.split('.')[0].split('_')[1])

                if frame < int(start_frame) or frame > int(end_frame): continue
                if frame % 2 != 0: continue

                e.skeleton_data[frame] = {}
                e.sorted_timestamps.append(frame)

                f1 = open(d_sk+file,'r')
                for count,line in enumerate(f1):
                    if count == 0:
                        t = line.split(':')[1].split('\n')[0]
                        e.sorted_ros_timestamps.append(np.float64(t))

                    # read the joint name
                    elif (count-1)%10 == 0:
                        j = line.split('\n')[0]
                        e.skeleton_data[frame][j] = []
                    # read the x value
                    elif (count-1)%10 == 2:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.skeleton_data[frame][j].append(a)
                    # read the y value
                    elif (count-1)%10 == 3:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.skeleton_data[frame][j].append(a)
                    # read the z value
                    elif (count-1)%10 == 4:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.skeleton_data[frame][j].append(a)

            # apply the filter and create World_Trace
            e.apply_mean_filter(window_length=mean_window)

            # add the x2d and y2d (using filtered x,y,z data)
            for frame in e.sorted_timestamps:
                for joint, j in e.filtered_skeleton_data[frame].items():
                    (x,y,z) = j
                    x2d = int(x*fx/z*1 +cx);
                    y2d = int(y*fy/z*-1+cy);
                    new_j = (x, y, z, x2d, y2d)
                    e.filtered_skeleton_data[frame][joint] = new_j

            # read robot data
            r_files = [f for f in os.listdir(d_robot) if os.path.isfile(os.path.join(d_robot,f))]
            for file in sorted(r_files):
                frame = int(file.split('.')[0].split('_')[1])
                e.robot_data[frame] = [[],[]]
                f1 = open(d_robot+file,'r')
                for count,line in enumerate(f1):
                    # read the x value
                    if count == 1:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.robot_data[frame][0].append(a)
                    # read the y value
                    elif count == 2:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.robot_data[frame][0].append(a)
                    # read the z value
                    elif count == 3:
                        a = float(line.split('\n')[0].split(':')[1])
                        e.robot_data[frame][0].append(a)
                    # read roll pitch yaw
                    elif count == 5:
                        ax = float(line.split('\n')[0].split(':')[1])
                    elif count == 6:
                        ay = float(line.split('\n')[0].split(':')[1])
                    elif count == 7:
                        az = float(line.split('\n')[0].split(':')[1])
                    elif count == 8:
                        aw = float(line.split('\n')[0].split(':')[1])
                        # ax,ay,az,aw
                        roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
                        pitch = 10*math.pi / 180.   #we pointed the pan tilt 10 degrees
                        e.robot_data[frame][1] = [roll,pitch,yaw]

            # add the map frame data for skeleton
            for frame in e.sorted_timestamps:
                e.map_frame_data[frame] = {}
                xr, yr, zr = e.robot_data[frame][0]
                yawr = e.robot_data[frame][1][2]
                pr = e.robot_data[frame][1][1]

                # because the Nite tracker has z as depth, height as y and left/right as x
                # we translate this to the map frame with x, y and z as height.
                for joint, (y,z,x,x2d,y2d) in e.filtered_skeleton_data[frame].items():
                    # transformation from camera to map
                    # rot_x =
                    rot_y = np.matrix([[np.cos(pr), 0, np.sin(pr)], [0, 1, 0], [-np.sin(pr), 0, np.cos(pr)]])
                    rot_z = np.matrix([[np.cos(yawr), -np.sin(yawr), 0], [np.sin(yawr), np.cos(yawr), 0], [0, 0, 1]])
                    rot = rot_z*rot_y

                    # robot's position in map frame
                    pos_r = np.matrix([[xr], [yr], [zr+1.66]])

                    # person's position in camera frame
                    pos_p = np.matrix([[x], [-y], [z]])

                    # person's position in map frame
                    map_pos = rot*pos_p+pos_r
                    x_mf = map_pos[0][0]
                    y_mf = map_pos[1][0]
                    z_mf = map_pos[2][0]

                    j = (x_mf, y_mf, z_mf)
                    e.map_frame_data[frame][joint] = j

            # soma_objects = get_soma_objects(e.region)
            soma_objects = create_soma_objects(e.region)
            # sys.exit(1)
            e.get_world_frame_trace(soma_objects)

            if len(e.sorted_timestamps) > 5:
                save_event(e, "Events")
            else:
                print "dont save me"
                print "dont save me"

def create_soma_objects(region):
    objects = {}
    objects['Kitchen'] = {}
    objects['Long_room'] = {}
    objects['Robot_lab'] = {}
    objects['Staff_Room'] = {}

    # kitchen
    objects['Kitchen'] = {
    'Printer_console_11': (-8.957, -17.511, 1.1),                           # fixed
    'Printer_paper_tray_110': (-9.420, -18.413, 1.132),                     # fixed
    'Shelves_44': (-8.226, -15.223, 1.0),
    'Microwave_3': (-4.835, -15.812, 1.0),                                  # fixed
    'Kettle_32': (-2.511, -15.724, 1.41),                                   # fixed
    'Tea_Pot_47': (-3.855, -15.957, 1.0),                                   # fixed
    'Water_Cooler_33': (-4.703, -15.558, 1.132),                            # fixed
    'Waste_Bin_24': (-1.982, -16.681, 0.91),                                # fixed
    'Waste_Bin_27': (-1.7636072635650635, -17.074087142944336, 0.5),
    'Sink_28': (-2.754, -15.645, 1.046),                                    # fixed
    'Fridge_7': (-2.425, -16.304, 0.885),                                   # fixed
    'Paper_towel_111': (-1.845, -16.346, 1.213),                            # fixed
    'Double_doors_112': (-8.365, -18.440, 1.021)
    }
    return objects[region]

def get_soma_objects(region):

    path = '/home/scpd/Dropbox/Programming/Luice/Python'
    try:
        file = path + "/" + 'soma_objects.p'
        with open(file, 'r') as f:
            soma_objects = pickle.load(f)
    except (EOFError, ValueError, TypeError), error:
        print "Load Error: ", error
    return soma_objects[region]


def get_event(recording, path, window=5):

    print recording
    print path

    d1 = os.path.join(path, recording)
    d_sk = os.path.join(d1, 'skeleton/')
    d_robot = os.path.join(d1, 'robot/')
    uuid = recording.split('_')[1]
    date = recording.split('_')[0]

    print d1, d_sk, d_robot
    print uuid, date

    sys.exit(1)

    # Get the robot's region location
    if os.path.isfile(os.path.join(path, 'meta.txt')):
        meta = open(os.path.join(path, 'meta.txt'), 'r')
        for count, line in enumerate(meta):
            if count == 0: region_id = line.split('\n')[0].split(':')[1]
            elif count == 1: region = line.split('\n')[0].split(':')[1]
            elif count == 1: pan = line.split('\n')[0].split(':')[1]
            elif count == 1: tilt = line.split('\n')[0].split(':')[1]

    e = event(uuid, event_label, d1, int(start_frame), int(end_frame), region, region_id)



if __name__ == "__main__":

    ##DEFAULTS:
    # path = '/home/' + getpass.getuser() + '/Dropbox/Programming/Luice/Datasets/Lucie/'
    path = '/home/' + getpass.getuser() + '/SkeletonDataset/SafeZone'
    mean_window = 5

    for cnt, f in enumerate(path):
        print "activity from: ", f
        get_event(f, path, mean_window)

    print "created events in %s directory" % cnt
