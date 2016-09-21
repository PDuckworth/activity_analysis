#!/usr/bin/env python

import roslib
roslib.load_manifest('tf')
import rospy
import tf
import sys
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from strands_navigation_msgs.msg import TopologicalMap
from skeleton_tracker.msg import joint_message, skeleton_tracker_state, skeleton_message, skeleton_complete
from mongodb_store.message_store import MessageStoreProxy

class SkeletonManager(object):
    """To deal with Skeleton messages once they are published as incremental msgs by OpenNI2."""

    def __init__(self):
        self.baseFrame = '/head_xtion_depth_optical_frame'
        self.joints = ['head', 'neck', 'torso', 'left_shoulder', 'left_elbow', 'left_hand',
                'left_hip', 'left_knee', 'left_foot', 'right_shoulder', 'right_elbow',
                'right_hand', 'right_hip', 'right_knee', 'right_foot']

        self.data = {} #current tf frame data for 15 joints
        self.accumulate_data = {} # accumulates multiple tf frames
        self.users = {} # keeps the tracker state message, timepoint and UUID
        self.map_info = "don't know"  # topological map name
        self.current_node = "don't care"  # topological node waypoint
        self.robot_pose = Pose()   # pose of the robot

        # logging to mongo:
        self._with_logging = rospy.get_param("~log_skeleton", "false")
        self._message_store = rospy.get_param("~message_store", "people_skeleton")

        # initialise data to recieve tf data  ## Moved to the action
        self._initialise_data()
        self.robot_pose_flag = 0

        # listeners:
        self.tf_listener = tf.TransformListener()

        rospy.Subscriber("skeleton_data/new_incremental", skeleton_message, self.incremental_callback)

        #self.uuid_listener = rospy.Subscriber("/people", user_ID, self.uuid_callback)
        rospy.Subscriber("skeleton_data/state", skeleton_tracker_state, self.tracker_state_callback)
        rospy.Subscriber("/current_node", String, callback=self.node_callback, queue_size=1)
        rospy.Subscriber("/robot_pose", Pose, callback=self.robot_callback, queue_size=1)
        self.topo_listerner = rospy.Subscriber("/topological_map", TopologicalMap, self.map_callback, queue_size = 10)

        # publishers:
        self.publish_incr = rospy.Publisher('skeleton_data/incremental', skeleton_message, queue_size = 10)
        self.publish_comp = rospy.Publisher('skeleton_data/complete', skeleton_complete, queue_size = 10)
        self.rate = rospy.Rate(15.0)

        # only publish the skeleton data when the person is far enough away (distance threshold)
        # maximum number of frames for one detection (not used when restricting the logger)
        self.frame_thresh = 5000
        self.dist_thresh = 1.0
        self.dist_flag = 1

        # initialise mongodb client
        if self._with_logging:
            rospy.loginfo("Connecting to mongodb...%s" % self._message_store)
            self._store_client = MessageStoreProxy(collection=self._message_store)
        print ">>initialised publisher"

    def _initialise_data(self):
        #to cope with upto 10 people in the scene
        for subj in xrange(1,11):
            self.data[subj] = {}
            self.data[subj]['flag'] = 0
            self.users[subj] = {"message": "No message"}

            for i in self.joints:
                self.data[subj][i] = dict()
                #self.data[subj][i]['value'] = [0,0,0]
                #self.data[subj][i]['value'] = [0,0,0]
                self.data[subj][i]['t_old'] = 0


    def _get_tf_data(self):
        for subj in xrange(1,11):
            joints_found = True
            for i in self.joints:
                if self.tf_listener.frameExists(self.baseFrame) and joints_found:
                    try:
                        tp = self.tf_listener.getLatestCommonTime(self.baseFrame, self.baseFrame+"/user_%d/%s" % (subj, i))
                        # print self.baseFrame+"/user_%d/%s" % (subj, i), tp
                        if tp != self.data[subj][i]['t_old']:
                            self.data[subj][i]['t_old'] = tp
                            self.data[subj]['flag'] = 1
                            (self.data[subj][i]['value'], self.data[subj][i]['rot']) = \
                                self.tf_listener.lookupTransform(self.baseFrame, self.baseFrame+"/user_%d/%s" % (subj, i), rospy.Time(0))

                    except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                        joints_found = False
                        self.data[subj]['flag'] = 0  #don't publish part of this Users skeleton
                        continue

            # stop tracking this user after this much frames
            if "frame" in self.users[subj]:
                if self.users[subj]["frame"] >= self.frame_thresh:
                    self.users[subj]["message"] = "Out of Scene"
                    self.data[subj]['flag'] = 0

            #If the tracker_state is 'Out of Scene' publish the accumulated skeleton
            if self.users[subj]["message"] == "Out of Scene" and subj in self.accumulate_data:
                self._publish_complete_data(subj, self.users[subj]["uuid"]) #logged_uuid from skeleton logger
                self.data[subj]['flag'] = 0

        #For all subjects, publish the incremental skeleton and accumulate into self.data also.
        list_of_subs = [subj for subj in self.data if self.data[subj]['flag'] == 1]

        # incr_msg = skeleton_message()    # if no subjects detected:
        for subj in list_of_subs:
            if self.users[subj]["message"] != "New":
                continue            # this catches cases where a User leaves the scene but they still have /tf data
            self.dist_flag = 1      # initiate the distance threshold flag to 1

            # print ">", subj,self.users[subj]["message"],self.users[subj]["frame"]
            incr_msg = skeleton_message()
            incr_msg.userID = subj
            incr_msg.uuid = self.users[subj]["uuid"]
            for i in self.joints:
                joint = joint_message()
                joint.name = i
                # joint.time = self.data[subj][i]['t_old']

                position = Point(x = self.data[subj][i]['value'][0], \
                           y = self.data[subj][i]['value'][1], z = self.data[subj][i]['value'][2])
                rot = Quaternion(x = self.data[subj][i]['rot'][0], y = self.data[subj][i]['rot'][1],
                           z = self.data[subj][i]['rot'][2], w = self.data[subj][i]['rot'][3])
                if self.data[subj][i]['value'][2] <= self.dist_thresh:
                    self.dist_flag = 0
                joint.pose.position = position
                joint.pose.orientation = rot
                incr_msg.joints.append(joint)
            self.data[subj]['flag'] = 0

            if self.dist_flag:
                #publish the instant frame message on /incremental topic
                self.publish_incr.publish(incr_msg)

                print self.new_incremental, incr_msg

                sys.exit(1)
                
                #update a frame
                self.users[subj]["frame"] += 1

                #accumulate the messages
                if self.users[subj]["message"] == "New":
                    self._accumulate_data(subj, incr_msg)
                elif self.users[subj]["message"] == "No message":
                    print "Just published this user. They are not back yet, get over it."
                else:
                    raise RuntimeError("this should never have occured; why is message not `New` or `Out of Scene' ??? ")

        #publish this for the action server if no subj in scene
        # if len(list_of_subs) == 0:
        #     self.publish_incr.publish(incr_msg)


    def _accumulate_data(self, subj, current_msg):
        # accumulate the multiple skeleton messages until user goes out of scene
        if current_msg.userID in self.accumulate_data:
            self.accumulate_data[current_msg.userID].append(current_msg)
        else:
            self.accumulate_data[current_msg.userID] = [current_msg]


    def _publish_complete_data(self, subj, uuid):
        # print "pub compl", subj, uuid
        # when user goes "out of scene" publish their accumulated data
        st = self.accumulate_data[subj][0].joints[0].time
        en = self.accumulate_data[subj][-1].joints[-1].time

        msg = skeleton_complete(uuid = uuid, \
                                skeleton_data = self.accumulate_data[subj], \
                                number_of_detections = len(self.accumulate_data[subj]), \
                                map_name = self.map_info, current_topo_node = self.current_node, \
                                start_time = st, end_time = en, robot_pose = self.robot_pose )

        self.users[subj]["message"] = "No message"
        self.publish_comp.publish(msg)
        rospy.loginfo("User #%s: published %s msgs as %s" % (subj, len(self.accumulate_data[subj]), msg.uuid))

        # remove the user from the users dictionary and the accumulated data dict.
        del self.accumulate_data[subj]
        del self.users[subj]["uuid"]

        if self._with_logging:
            query = {"uuid" : msg.uuid}
            #self._store_client.insert(traj_msg, meta)
            self._store_client.update(message=msg, message_query=query, upsert=True)


    def tracker_state_callback(self, msg):
        if self.robot_pose_flag != 1: return
        # get the tracker state message and UUID of tracker user
        if msg.message == 'New':
            self.users[msg.userID]["uuid"] = msg.uuid
            self.users[msg.userID]["frame"] = 0
        self.users[msg.userID]["message"] = msg.message
        self.users[msg.userID]["timepoint"] = msg.timepoint


    def incremental_callback(self, msg):
        self.new_incremental = msg

    def robot_callback(self, msg):
        if self.robot_pose_flag != 1: return
        self.robot_pose = msg

    def node_callback(self, msg):
        if self.robot_pose_flag != 1: return
        self.current_node = msg.data

    def map_callback(self, msg):
        # get the topological map name
        self.map_info = msg.map
        self.topo_listerner.unregister()

    def publish_skeleton(self):
        self._get_tf_data()


if __name__ == '__main__':
    rospy.init_node('skeleton_publisher', anonymous=True)

    sk_manager = SkeletonManager()
    sk_manager.publish_skeleton()
