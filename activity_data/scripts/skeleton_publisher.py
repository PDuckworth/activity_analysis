#!/usr/bin/env python
import roslib
import rospy
import os
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge
import getpass, datetime
import argparse
import sensor_msgs.msg
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
import topological_navigation.msg
from strands_navigation_msgs.msg import TopologicalMap
from skeleton_tracker.msg import skeleton_tracker_state, skeleton_message, robot_message
from activity_data.msg import  skeleton_complete
from mongodb_store.message_store import MessageStoreProxy
from tf.transformations import euler_from_quaternion

class SkeletonManager(object):
    """To deal with Skeleton messages once they are published as incremental msgs by OpenNI2."""

    def __init__(self, record_rgb=False):

        self.record_rgb = record_rgb # to deal with the anonymous setting at tsc
        self.accumulate_data = {} # accumulates multiple skeleton msg
        self.accumulate_robot = {} # accumulates multiple skeleton msg
        self.sk_mapping = {} # does something in for the image logging

        self.map_info = "don't know"  # topological map name
        self.current_node = "don't care"  # topological node waypoint
        self.robot_pose = Pose()   # pose of the robot
        self.ptu_pan = self.ptu_tilt = 0.0

        # directory to store the data
        self.date = str(datetime.datetime.now().date())

        self.dir1 = '/home/' + getpass.getuser() + '/SkeletonDataset/no_consent/' + self.date+'/'
        if not os.path.exists(self.dir1):
            print '  -create folder:',self.dir1
            os.makedirs(self.dir1)

        # flags to make sure we received every thing
        self._flag_robot = 0
        self._flag_node = 0
        self._flag_rgb = 0
        #self._flag_rgb_sk = 0
        self._flag_depth = 0

        # open cv stuff
        self.cv_bridge = CvBridge()

        # Get rosparams
        self._with_logging = rospy.get_param("~log_skeleton", "false")
        self._message_store = rospy.get_param("~message_store", "people_skeleton")
        self._database = rospy.get_param("~database", "message_store")
        self.camera = "head_xtion"

        # listeners
        rospy.Subscriber("skeleton_data/incremental", skeleton_message, self.incremental_callback)
        rospy.Subscriber('/'+self.camera+'/rgb/image_color', sensor_msgs.msg.Image, callback=self.rgb_callback, queue_size=10)
        # rospy.Subscriber('/'+self.camera+'/rgb/sk_tracks', sensor_msgs.msg.Image, callback=self.rgb_sk_callback, queue_size=10)
        rospy.Subscriber('/'+self.camera+'/depth/image' , sensor_msgs.msg.Image, self.depth_callback, queue_size=10)
        rospy.Subscriber("/robot_pose", Pose, callback=self.robot_callback, queue_size=10)
        rospy.Subscriber("/current_node", String, callback=self.node_callback, queue_size=1)
        rospy.Subscriber("/ptu/state", sensor_msgs.msg.JointState, callback=self.ptu_callback, queue_size=1)
        self.topo_listerner = rospy.Subscriber("/topological_map", TopologicalMap, self.map_callback, queue_size = 10)
        rospy.Subscriber("skeleton_data/state", skeleton_tracker_state, self.state_callback)

        # publishers:
        # self.publish_incr = rospy.Publisher('skeleton_data/incremental', skeleton_message, queue_size = 10)
        self.publish_comp = rospy.Publisher('skeleton_data/complete', skeleton_complete, queue_size = 10)
        self.rate = rospy.Rate(15.0)

        # only publish the skeleton data when the person is far enough away (distance threshold)
        # maximum number of frames for one detection
        self.max_num_frames = 3000
        self.dist_thresh = 0
        self.dist_flag = 1

        # mongo store
        if self._with_logging:
            rospy.loginfo("Connecting to mongodb...%s" % self._message_store)
            self._store_client = MessageStoreProxy(collection=self._message_store, database=self._database)

    def convert_to_world_frame(pose, robot_msg):
        """Convert a single camera frame coordinate into a map frame coordinate"""
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        y,z,x = pose.x, pose.y, pose.z

        (xr, yr, zr) = robot_msg.robot_pose.position
        (ax, ay, az, aw) = robot_msg.robot_pose.orientation
        roll, pr, yawr = euler_from_quaternion([ax, ay, az, aw])

        yawr += robot_msg.PTU_pan
        pr += robot_msg.PTU_tilt

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

        print ">>" , x_mf, y_mf, z_mf
        return Point(x_mf, y_mf, z_mf)

    def _publish_complete_data(self, subj, uuid, vis=False):
        """when user goes "out of scene" publish their accumulated data"""
        # print ">> publishing these: ", uuid, len(self.accumulate_data[uuid])

        st = self.accumulate_data[uuid][0].time
        en = self.accumulate_data[uuid][-1].time

        first_pose = self.accumulate_data[uuid][0].joints[0].pose.positon
        robot_msg =  self.accumulate_robot[uuid][0]
        first_map_point = convert_to_world_frame(first_pose, robot_msg)

        vis=True
        if vis:
            print ">>>"
            print "storing: ", uuid, type(uuid)
            print "date: ", self.date, type(self.date)
            print "number of detectons: ", len(self.accumulate_data[uuid]), type(len(self.accumulate_data[uuid]))
            print "map info: ", self.map_info, type(self.map_info)
            print "current node: ", self.current_node, type(self.current_node)
            print "start/end rostime:", st, type(st), en, type(en)

        msg = skeleton_complete(uuid = uuid, date = self.date, \
                                time = self.sk_mapping[uuid]['time'], \
                                skeleton_data = self.accumulate_data[uuid], \
                                number_of_detections = len(self.accumulate_data[uuid]), \
                                map_name = self.map_info, current_topo_node = self.current_node, \
                                start_time = st, end_time = en, robot_data = self.accumulate_robot[uuid], \
                                human_map_point = first_map_point)

        self.publish_comp.publish(msg)
        rospy.loginfo("User #%s: published %s msgs as %s" % (subj, len(self.accumulate_data[uuid]), msg.uuid))

        if self._with_logging:
            query = {"uuid" : msg.uuid}
            #self._store_client.insert(traj_msg, meta)
            self._store_client.update(message=msg, message_query=query, upsert=True)

        # remove the user from the users dictionary and the accumulated data dict.
        del self.accumulate_data[uuid]
        del self.sk_mapping[uuid]

    def _dump_images(self, msg):
        """For each incremental skeleton - dump an rgb/depth image to file"""

        self.inc_sk = msg
        if str(datetime.datetime.now().date()) != self.date:
            print 'new day!'
            self.date = str(datetime.datetime.now().date())
            self.dir1 = '/home/' + getpass.getuser() + '/SkeletonDataset/no_consent/'+self.date+'/'
            print 'checking if folder exists:',self.dir1
            if not os.path.exists(self.dir1):
                print '  -create folder:',self.dir1
                os.makedirs(self.dir1)

        if self.sk_mapping[msg.uuid]["state"] is 'Tracking' and self.sk_mapping[msg.uuid]["frame"] is 1:

            self.sk_mapping[msg.uuid]['time'] = str(datetime.datetime.now().time()).split('.')[0]
            t = self.sk_mapping[msg.uuid]['time']+'_'
            print '  -new skeleton detected with id:', msg.uuid
            new_dir = self.dir1+self.date+'_'+t+msg.uuid #+'_'+waypoint
            # print "new", new_dir

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                os.makedirs(new_dir+'/depth')
                os.makedirs(new_dir+'/robot')
                os.makedirs(new_dir+'/skeleton')
                if self.record_rgb:
                    os.makedirs(new_dir+'/rgb')
                    #os.makedirs(new_dir+'/rgb_sk')

                # create the empty bag file (closed in /skeleton_action)
                # self.bag_file = rosbag.Bag(new_dir+'/detection.bag', 'w')


        t = self.sk_mapping[self.inc_sk.uuid]['time']+'_'
        new_dir = self.dir1+self.date+'_'+t+self.inc_sk.uuid #+'_'+waypoint
        if os.path.exists(new_dir):
            # setup saving dir and frame
            d = new_dir+'/'
            f = self.sk_mapping[self.inc_sk.uuid]['frame']
            if f < 10:          f_str = '0000'+str(f)
            elif f < 100:          f_str = '000'+str(f)
            elif f < 1000:          f_str = '00'+str(f)
            elif f < 10000:          f_str = '0'+str(f)
            elif f < 100000:          f_str = str(f)

            # save images
            if self.record_rgb:
                cv2.imwrite(d+'rgb/rgb_'+f_str+'.jpg', self.rgb)
                #cv2.imwrite(d+'rgb_sk/sk_'+f_str+'.jpg', self.rgb_sk)
            cv2.imwrite(d+'depth/depth_'+f_str+'.jpg', self.xtion_img_d_rgb)

            # try:
            #     self.bag_file.write('rgb', self.rgb_msg)
            #     self.bag_file.write('depth', self.depth_msg)
            #     self.bag_file.write('rgb_sk', self.rgb_sk_msg)
            # except:
            #     rospy.logwarn("Can not write rgb, depth, and rgb_sk to a bag file.")

            x=float(self.robot_pose.position.x)
            y=float(self.robot_pose.position.y)
            z=float(self.robot_pose.position.z)
            xo=float(self.robot_pose.orientation.x)
            yo=float(self.robot_pose.orientation.y)
            zo=float(self.robot_pose.orientation.z)
            wo=float(self.robot_pose.orientation.w)
            p = Point(x, y, z)
            q = Quaternion(xo, yo, zo, wo)
            robot = Pose(p,q)

            pan = float(self.ptu_pan)
            tilt = float(self.ptu_tilt)

            # try:
            #     self.bag_file.write('robot', robot)
            # except:
            #    rospy.logwarn("cannot write the robot position to bag. Has the bag been closed already?")

            # save robot data  in text file
            f1 = open(d+'robot/robot_'+f_str+'.txt','w')
            f1.write('position\n')
            f1.write('x:'+str(x)+'\n')
            f1.write('y:'+str(y)+'\n')
            f1.write('z:'+str(z)+'\n')
            f1.write('orientation\n')
            f1.write('x:'+str(xo)+'\n')
            f1.write('y:'+str(yo)+'\n')
            f1.write('z:'+str(zo)+'\n')
            f1.write('w:'+str(wo)+'\n')
            f1.write('ptu_state\n')
            f1.write('ptu_pan:'+str(pan)+'\n')
            f1.write('ptu_tilt:'+str(tilt)+'\n')
            f1.close()

            # save skeleton data in bag file
            #x=float(self.robot_pose.position.x)
            #y=float(self.robot_pose.position.y)
            #z=float(self.robot_pose.position.z)
            #xo=float(self.robot_pose.orientation.x)
            #yo=float(self.robot_pose.orientation.y)
            #zo=float(self.robot_pose.orientation.z)
            #wo=float(self.robot_pose.orientation.w)
            #p = Point(x, y, z)
            #q = Quaternion(xo, yo, zo, wo)
            #skel = Pose(p,q)
            #bag.write('skeleton', skel)

            # save skeleton data in text file
            f1 = open(d+'skeleton/skl_'+f_str+'.txt','w')
            f1.write('time:'+str(self.inc_sk.time.secs)+'.'+str(self.inc_sk.time.nsecs)+'\n')
            for i in self.inc_sk.joints:
                f1.write(i.name+'\n')
                f1.write('position\n')
                f1.write('x:'+str(i.pose.position.x)+'\n')
                f1.write('y:'+str(i.pose.position.y)+'\n')
                f1.write('z:'+str(i.pose.position.z)+'\n')
                f1.write('orientation\n')
                f1.write('x:'+str(i.pose.orientation.x)+'\n')
                f1.write('y:'+str(i.pose.orientation.y)+'\n')
                f1.write('z:'+str(i.pose.orientation.z)+'\n')
                f1.write('w:'+str(i.pose.orientation.w)+'\n')
                f1.write('confidence:'+str(i.confidence)+'\n')
            f1.close()

            # update frame number
            if self.inc_sk.uuid in self.sk_mapping:
                self.sk_mapping[self.inc_sk.uuid]['frame'] += 1

            #publish the gaze request of person on every detection:
            # if self.inc_sk.joints[0].name == 'head':
            #     head = Header(frame_id='head_xtion_depth_optical_frame')
            #     look_at_pose = PoseStamped(header = head, pose=self.inc_sk.joints[0].pose)
            #     self.publish_consent_pose.publish(look_at_pose)
            # #self.gazeClient.send_goal(self.gazegoal)

    def incremental_callback(self, msg):
        """accumulate the multiple skeleton messages until user goes out of scene"""
        if self._flag_robot and self._flag_rgb and self._flag_depth:
            if msg.uuid in self.sk_mapping:
                if self.sk_mapping[msg.uuid]["state"] is 'Tracking':
                    if len(self.accumulate_data[msg.uuid]) < self.max_num_frames:
                        self.accumulate_data[msg.uuid].append(msg)
                        robot_msg = robot_message(robot_pose = self.robot_pose, PTU_pan = self.ptu_pan, PTU_tilt = self.ptu_tilt)
                        self.accumulate_robot[msg.uuid].append(robot_msg)
                        self._dump_images(msg)
                        # print msg.userID, msg.uuid, len(self.accumulate_data[msg.uuid])

    def new_user_detected(self, msg):
        self.sk_mapping[msg.uuid] = {"state":'Tracking', "frame":1}
        self.accumulate_data[msg.uuid] = []
        self.accumulate_robot[msg.uuid] = []

    def state_callback(self, msg):
        """Reads the state messages from the openNi tracker"""
        # print msg.uuid, msg.userID, msg.message
        if msg.message == "Tracking":
            self.new_user_detected(msg)
        elif msg.message == "Out of Scene" and msg.uuid in self.sk_mapping:
            self.sk_mapping[msg.uuid]["state"] = "Out of Scene"
        elif msg.message == "Visible" and msg.uuid in self.sk_mapping:
            self.sk_mapping[msg.uuid]["state"] = "Tracking"
        elif msg.message == "Stopped tracking" and msg.uuid in self.accumulate_data:
            if len(self.accumulate_data[msg.uuid]) != 0:
                self._publish_complete_data(msg.userID, msg.uuid)   #only publish if data captured

    def robot_callback(self, msg):
        self.robot_pose = msg
        if self._flag_robot == 0:
            print ' >robot pose received'
            self._flag_robot = 1

    def ptu_callback(self, msg):
        self.ptu_pan, self.ptu_tilt = msg.position

    def node_callback(self, msg):
        self.current_node = msg.data
        if self._flag_node == 0:
            print ' >current node received'
            self._flag_node = 1

    def map_callback(self, msg):
        # get the topological map name
        self.map_info = msg.map
        self.topo_listerner.unregister()

    def rgb_callback(self, msg):
        # self.rgb_msg = msg
        rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self._flag_rgb is 0:
            print ' >rgb image received'
            self._flag_rgb = 1

    #def rgb_sk_callback(self, msg):
    #    # self.rgb_sk_msg = msg
    #    rgb_sk = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #    self.rgb_sk = cv2.cvtColor(rgb_sk, cv2.COLOR_RGB2BGR)
    #    if self._flag_rgb_sk is 0:
    #        print ' >rgb skel image received'
    #        self._flag_rgb_sk = 1

    def depth_callback(self, msg):
        # self.depth_msg = msg
        depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_array = np.array(depth_image, dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        self.xtion_img_d_rgb = depth_array*255
        if self._flag_depth is 0:
            print ' >depth image received'
            self._flag_depth = 1

if __name__ == '__main__':
    rospy.init_node('skeleton_publisher', anonymous=True)

    record_rgb = rospy.get_param("~record_rgb", True)
    print "recording RGB images: %s" % record_rgb

    sk_manager = SkeletonManager(record_rgb)
    rospy.spin()
