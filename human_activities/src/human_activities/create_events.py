#!/usr/bin/env python
__author__ = 'p_duckworth'
import os, sys, csv
import pymongo
import itertools
import cPickle as pickle
import multiprocessing as mp
import numpy as np
import scipy as sp
import math
import bisect
import getpass
from scipy import signal
import matplotlib.pyplot as plt
import utils as utils
from tf.transformations import euler_from_quaternion
from qsrlib_io.world_trace import Object_State, World_Trace


class event(object):
    """Event object class"""
    def __init__(self, uuid, dir_, start, end, label, waypoint=None):

        self.uuid = uuid
        # self.start_frame, self.end_frame = self.get_last_frame()
        # self.waypoint = waypoint
        self.dir = dir_
        self.start_frame = start
        self.end_frame = end
        self.label = label
        self.sorted_timestamps = []
        # self.sorted_ros_timestamps = []
        self.bad_timepoints = {}        #use for filtering
        self.skeleton_data = {}         #type: dict[timepoint][joint_id]= (x, y, z, x2d, y2d)
        self.map_frame_data = {}        #type: dict[timepoint][joint_id]= (x, y, z, x2d, y2d)
        self.robot_data = {}            #type: dict[timepoint][joint_id]= ((x, y, z), (roll, pitch, yaw))
        # self.world = World_Trace()

    def apply_median_filter(self, window_size=11, vis=False):
        """Once obtained the joint x,y,z coords.
        Apply a median filter over a temporal window to smooth the joint positions.
        Whilst doing this, create a world Trace object"""

        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        data, f_data, ob_states = {}, {}, {}
        self.camera_world = World_Trace()
        self.filtered_skeleton_data = {}

        for t in self.sorted_timestamps:
            self.filtered_skeleton_data[t] = {}  # initialise with all timepoints
            for joint_id, (x,y,z) in self.skeleton_data[t].items():
                try:
                    data[joint_id]["x"].append(x)
                    data[joint_id]["y"].append(y)
                    data[joint_id]["z"].append(z)
                except KeyError:
                    data[joint_id] = {"x":[x], "y":[y], "z":[z]}

        for joint_id, joint_dic in data.items():
            f_data[joint_id] = {}
            for dim, values in joint_dic.items():
                filtered_values = sp.signal.medfilt(values, window_size) # filter
                f_data[joint_id][dim] = filtered_values

                if dim is "z" and 0 in filtered_values:
                    print ">> Z should never be 0. (distance to camera)."
                    return False
                    #vis = True

                if vis and "hand" in joint_id:
                    print joint_id
                    title1 = 'input %s position: %s' % (joint_id, dim)
                    title2 = 'filtered %s position: %s' % (joint_id, dim)
                    #plot_the_results(values, filtered_values, title1, title2)

            # Create a QSRLib format list of Object States (for each joint id)
            for cnt, t in enumerate(self.sorted_timestamps):
                x = f_data[joint_id]["x"][cnt]
                y = f_data[joint_id]["y"][cnt]
                z = f_data[joint_id]["z"][cnt]

                # add the x2d and y2d (using filtered x,y,z data)
                # print self.uuid, t, joint_id, (x,y,z)
                try:
                    x2d = int(x*fx/z*1 +cx);
                    y2d = int(y*fy/z*-1+cy);
                except ValueError:
                    print "ValueError for %s at frame: %s. t:%s" %(self.uuid, cnt, t)
                    x2d = 0
                    y2d = 0

                self.filtered_skeleton_data[t][joint_id] = (x, y, z, x2d, y2d)    # Kept for Legacy
                try:
                    ob_states[joint_id].append(Object_State(name=joint_id, timestamp=cnt, x=x, y=y, z=z))
                except KeyError:
                    ob_states[joint_id] = [Object_State(name=joint_id, timestamp=cnt, x=x, y=y, z=z)]

        # #Add all the joint IDs into the World Trace
        for joint_id, obj in ob_states.items():
            self.camera_world.add_object_state_series(obj)
        return True

    def apply_mean_filter_old(self, window_length=3):
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
        """Accepts a dictionary of world (soma) objects.
        Adds the position of the object at each timepoint into the World Trace
        Note 'frame' is the actual detection file number. 't' is a timestamp starting from 0.
        't' is used in the World Trace."""

        ob_states={}
        world = World_Trace()
        # print ">>", self.map_frame_data.keys()
        for t, frame in enumerate(self.sorted_timestamps):
            #Joints:
            for joint_id, (x, y, z) in self.map_frame_data[frame].items():
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
            (x,y,z) = self.robot_data[frame][0]
            if 'robot' not in ob_states.keys():
                ob_states['robot'] = [Object_State(name='robot', timestamp=t, x=x, y=y, z=z)]
            else:
                ob_states['robot'].append(Object_State(name='robot', timestamp=t, x=x, y=y, z=z))

        for object_state in ob_states.values():
            world.add_object_state_series(object_state)
        self.map_world = world


def get_labels(d1, d_sk):
    labels = {}
    if os.path.isfile(os.path.join(d1, 'labels.txt')):
        f1 = open(os.path.join(d1, 'labels.txt'), 'r')
        for count, line in enumerate(f1):
            line_ = line.split('\n')[0].split(':')
            activity_class = line_[1]
            if activity_class != 'making_tea':
                start, end = line_[2].split(',')
                labels[count] = (activity_class, int(start), int(end))
    else:
        number_of_frames = len([1 for f in os.listdir(d_sk) ])
        labels[0] = ("NA", 0, int(number_of_frames-1))
    return labels

def get_event(recording, path, soma_objects, config):
    """create event class from a recording"""

    """directories containing the data"""
    d1 = os.path.join(path, recording)
    if config["use_cpm"]:
        d_sk = os.path.join(d1, 'cpm_skeleton/')
    else:
        d_sk = os.path.join(d1, 'skeleton/')
    d_robot = os.path.join(d1, 'robot/')

    # If recording has been restricted with ROIs:
    region = ""
    if os.path.isfile(os.path.join(d1, 'meta.txt')):
        f1 = open(os.path.join(d1, 'meta.txt'), 'r')
        for count, line in enumerate(f1):
            if count == 0: region = line.split('\n')[0].split(':')[1].replace(" ", "")

    objects = {}
    try:
        objects = soma_objects[region]
    except KeyError:
        for r, objs in soma_objects.items():
            for ob, position in objs.items():
                objects[ob] = position

    try:
        """information stored in the filename - differently for ecai dataset"""
        date = recording.split('_')[0]
        time = recording.split('_')[1]
        uuid = recording.split('_')[2]
    except IndexError:
        date = ""
        time = recording.split('_')[0]
        uuid = recording.split('_')[1]
    print "date: %s. time: %s. uid: %s. " % (date, time, uuid) ,
    # print "objs: %s" % use_objects
    labels = get_labels(d1, d_sk)

    for iter, (label, st, end) in labels.items():
        # print iter
        """initialise event"""
        e = event(uuid, d1, st, end, label)
        e.objects = objects

        """get the skeleton data from each timepoint file"""
        sk_files = [f for f in os.listdir(d_sk) if os.path.isfile(os.path.join(d_sk, f))]

        """reduce the number of frames by a rate. Re-number from 1."""
        # frame = 1
        for file in sorted(sk_files):
            # original_frame = int(file.split('.')[0].split('_')[1])
            # if original_frame % config['reduce_frame_rate'] != 0: continue

            # new file format: "cpm_skl_00540.txt"
            if config["use_cpm"]:
                frame = int(file.split('.')[0].split('_')[-1])
            else:
                frame = int(file.split('.')[0].split('_')[1])

            if frame < int(st) or frame > int(end): continue
            e.sorted_timestamps.append(frame)
            f1 = open(d_sk+file,'r')
            # e.skeleton_data[frame] = {}
            e.skeleton_data[frame] = get_openni_values(f1)

            # frame+=1
            # if config['rec_inc_confidence_vals']:
            #     for count,line in enumerate(f1):
            #         if count == 0:
            #             t = line.split(':')[1].split('\n')[0]
            #             e.sorted_ros_timestamps.append(np.float64(t))
            #
            #         # read the joint name
            #         elif (count-1)%11 == 0:
            #             j = line.split('\n')[0]
            #             e.skeleton_data[frame][j] = []
            #         # read the x value
            #         elif (count-1)%11 == 2:
            #             a = float(line.split('\n')[0].split(':')[1])
            #             e.skeleton_data[frame][j].append(a)
            #         # read the y value
            #         elif (count-1)%11 == 3:
            #             a = float(line.split('\n')[0].split(':')[1])
            #             e.skeleton_data[frame][j].append(a)
            #         # read the z value
            #         elif (count-1)%11 == 4:
            #             a = float(line.split('\n')[0].split(':')[1])
            #             e.skeleton_data[frame][j].append(a)
            # else:
            #     for count,line in enumerate(f1):
            #         if count == 0:
            #             t = line.split(':')[1].split('\n')[0]
            #             e.sorted_ros_timestamps.append(np.float64(t))
            #
            #         # read the joint name
            #         elif (count-1)%10 == 0:
            #             j = line.split('\n')[0]
            #             e.skeleton_data[frame][j] = []
            #         # read the x value
            #         elif (count-1)%10 == 2:
            #             a = float(line.split('\n')[0].split(':')[1])
            #             e.skeleton_data[frame][j].append(a)
            #         # read the y value
            #         elif (count-1)%10 == 3:
            #             a = float(line.split('\n')[0].split(':')[1])
            #             e.skeleton_data[frame][j].append(a)
            #         # read the z value
            #         elif (count-1)%10 == 4:
            #             a = float(line.split('\n')[0].split(':')[1])
            #             e.skeleton_data[frame][j].append(a)

        # for frame, data in  e.skeleton_data.items():
            # print frame, data['head']
        # sys.exit(1)

        """ apply a skeleton data filter and create a QSRLib.World_Trace object"""
        # e.apply_mean_filter(window_length=config['joints_mean_window'])
        if not e.apply_median_filter(config['joints_mean_window']): return

        """ read robot odom data"""

        r_files = [f for f in os.listdir(d_robot) if os.path.isfile(os.path.join(d_robot, f))]
        # frame = 1
        for file in sorted(r_files):
            frame = int(file.split('.')[0].split('_')[1])
            if frame not in e.skeleton_data.keys(): continue

            # if original_frame % config['reduce_frame_rate'] != 0: continue
            e.robot_data[frame] = [[],[]]
            f1 = open(d_robot+file,'r')
            for count, line in enumerate(f1):
                # print count, line, config['ptu_vals_stored']

                if count == 1:# read the x value
                    a = float(line.split('\n')[0].split(':')[1])
                    e.robot_data[frame][0].append(a)
                elif count == 2:# read the y value
                    a = float(line.split('\n')[0].split(':')[1])
                    e.robot_data[frame][0].append(a)
                elif count == 3:# read the z value
                    a = float(line.split('\n')[0].split(':')[1])
                    e.robot_data[frame][0].append(a)
                elif count == 5:# read roll pitch yaw
                    ax = float(line.split('\n')[0].split(':')[1])
                elif count == 6:
                    ay = float(line.split('\n')[0].split(':')[1])
                elif count == 7:
                    az = float(line.split('\n')[0].split(':')[1])
                elif count == 8:
                    aw = float(line.split('\n')[0].split(':')[1])

                    if not config['ptu_vals_stored']:
                        roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
                        pitch = 10*math.pi / 180.   #we pointed the pan tilt 10 degrees
                        e.robot_data[frame][1] = [roll,pitch,yaw]

                elif count > 8 and config['ptu_vals_stored']:
                    if count == 10:
                        pan = float(line.split('\n')[0].split(':')[1])
                    elif count == 11:
                        tilt = float(line.split('\n')[0].split(':')[1])
                        roll, pitch, yaw = euler_from_quaternion([ax, ay, az, aw])    #odom
                        yaw += pan #*math.pi / 180.                   # this adds the pan of the ptu state when recording took place.
                        pitch += tilt #*math.pi / 180.                # this adds the tilt of the ptu state when recording took place.
                        e.robot_data[frame][1] = [roll,pitch,yaw]
            # frame+=1

        # add the map frame data for the skeleton detection
        for frame in e.sorted_timestamps:
            """Note frame does not start from 0. It is the actual file frame number"""
            e.map_frame_data[frame] = {}
            xr, yr, zr = e.robot_data[frame][0]
            yawr = e.robot_data[frame][1][2]
            pr = e.robot_data[frame][1][1]

            #  because the Nite tracker has z as depth, height as y and left/right as x
            #  we translate this to the map frame with x, y and z as height.
            for joint, (y,z,x,x2d,y2d) in e.filtered_skeleton_data[frame].items():
                # transformation from camera to map
                rot_y = np.matrix([[np.cos(pr), 0, np.sin(pr)], [0, 1, 0], [-np.sin(pr), 0, np.cos(pr)]])
                rot_z = np.matrix([[np.cos(yawr), -np.sin(yawr), 0], [np.sin(yawr), np.cos(yawr), 0], [0, 0, 1]])
                rot = rot_z*rot_y

                pos_r = np.matrix([[xr], [yr], [zr+1.66]]) # robot's position in map frame
                pos_p = np.matrix([[x], [-y], [-z]]) # person's position in camera frame

                map_pos = rot*pos_p+pos_r # person's position in map frame
                x_mf = map_pos[0,0]
                y_mf = map_pos[1,0]
                z_mf = map_pos[2,0]

                j = (x_mf, y_mf, z_mf)
                e.map_frame_data[frame][joint] = j

        # print ">",e.sorted_timestamps
        # for i in e.sorted_timestamps:
        #     print i, e.map_frame_data[i]['head'], e.map_frame_data[i]['left_hand']#, e.map_frame_data[i]['right_hand'] #e.skeleton_data[i]['right_hand'], e.map_frame_data[i]['right_hand']   , yaw, pitch
        # sys.exit(1)
        if len(e.sorted_timestamps) >= 5:
            e.get_world_frame_trace(e.objects)
            utils.save_event(e, "Learning/Events")
            return True
        else:
            print "  >dont save me - recording too short."
            return False

def get_openni_values(f1):
    openni_values = {}
    for count, line in enumerate(f1):
        if count == 0: continue
        line = line.split(',')
        joint_name = line[0]
        openni_values[joint_name] = {}
        x2d = float(line[1])
        y2d = float(line[2])
        x = float(line[3])
        y = float(line[4])
        z = float(line[5])
        # c = float(line[6])
        openni_values[joint_name] = [x,y,z]
        # openni_values[joint_name]['x2d']=x2d
        # openni_values[joint_name]['y2d']=y2d
        # openni_values[joint_name]['x']=x
        # openni_values[joint_name]['y']=y
        # openni_values[joint_name]['z']=z
    return openni_values


# def get_soma_objects():

    # objects = {}
    #objects['Kitchen'] = {}
    #objects['Long_room'] = {}
    # objects['Robot_lab'] = {}
    #objects['Staff_Room'] = {}

    # kitchen
    #objects['Kitchen'] = {
    #'Printer_console_11': (-8.957, -17.511, 1.1),                           # fixed
    #'Printer_paper_tray_110': (-9.420, -18.413, 1.132),                     # fixed
    #'Microwave_3': (-4.835, -15.812, 1.0),                                  # fixed
    #'Kettle_32': (-2.511, -15.724, 1.41),                                   # fixed
    #'Tea_Pot_47': (-3.855, -15.957, 1.0),                                   # fixed
    #'Water_Cooler_33': (-4.703, -15.558, 1.132),                            # fixed
    #'Waste_Bin_24': (-1.982, -16.681, 0.91),                                # fixed
    #'Waste_Bin_27': (-1.7636072635650635, -17.074087142944336, 0.5),
    #'Sink_28': (-2.754, -15.645, 1.046),                                    # fixed
    #'Fridge_7': (-2.425, -16.304, 0.885),                                   # fixed
    #'Paper_towel_111': (-1.845, -16.346, 1.213),                            # fixed
    #'Double_doors_112': (-8.365, -18.440, 1.021),
    #}
    # objects['Robot_lab'] = {
    # 'Majd_desk_1': (-7.3, -33.5, 1.2),
    # 'Baxter_desk_1': (-4.4, -31.8, 1.2),
    # 'Poster_Board_1': (-4.3, -34.0, 1.2)
    # }
    # return objects

# def get_soma_objects():
#     #todo: read from soma2 mongo store.
#
#     """TSC OBJECTS"""
#     objects = {}
#     objects['Kitchen'] = {}
#     objects['Reception'] = {}
#     objects['Hopotality'] = {}
#     objects['Corporate'] = {}
#     objects['Support'] = {}
#
#     objects['Kitchen'] = {
#     #'Microwave_1':  (-53.894511011092348, -5.6271549435167918, 1.2075203138621333),
#     'Microwave_2':  (-52.29, -5.6271549435167918, 1.2075203138621333),
#     'Sink_2':  (-55.902430164089097, -5.3220418631789883, 0.95348616325025226),
#     'Fruit_bowl_3':  (-55.081272358597374, -8.5550720977828973, 1.2597648941515749),
#     #'Fruit_bowl_11':  (-8.957, -17.511, 1.1),
#     'Dishwasher_4':  (-55.313495480985964, -5.822285141172836, 0.87860846828010275),
#     'Coffee_Machine_5': (-50.017233832554183, -5.4918825204775921, 1.3139597647929069)
#     }
#
#     objects['Reception'] = {
#     'Coffee_Machine_6': (-5.5159040452346737, 28.564135219405774, 1.3149322505645362),
#     #'Fridge_7': (-50.35, -5.24, 1.51)
#     }
#
#     objects['Hospitality'] = {
#     'Printer_8':  (-1.6876578896088092, -5.9433505603441326, 1.1084470787101761),
#     'Sink_9':  (2.9, -3.0, 1.1),
#     #'Coffee_Machine_10': (-50.35, -5.24, 1.51)
#     }
#
#     objects['Corporate'] = {
#     'Printer_11':  (-23.734682053245283, -14.096880839756942, 1.106873440473277),
#     }
#
#     objects['Support'] = {
#     #'Printer_1 2':  (-8.957, -17.511, 1.1),
#     }
#
#     return objects


if __name__ == "__main__":

    ##DEFAULTS:
    # path = '/home/' + getpass.getuser() + '/Dropbox/Programming/Luice/Datasets/Lucie/'
    path = '/home/' + getpass.getuser() + '/SkeletonDataset'
    config = {
        'reduce_frame_rate' : 3,
        'joints_mean_window' : 5}

    for cnt, f in enumerate(path):
        print "activity from: ", f
        get_event(f, path, config)

    print "created events in %s directory" % cnt
