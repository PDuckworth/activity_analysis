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
from mongodb_store.message_store import MessageStoreProxy

class SkeletonManagerConsent(object):
    """To deal with Skeleton messages once they are published as incremental msgs by OpenNI2."""

    def __init__(self):

        self.reinisialise()
        self.reset_data()

        # open cv stuff
        self.cv_bridge = CvBridge()
        self.camera = "head_xtion"
        self.max_num_frames = 1000

        # listeners
        rospy.Subscriber("skeleton_data/incremental", skeleton_message, self.incremental_callback)
        rospy.Subscriber('/'+self.camera+'/rgb/image_color', sensor_msgs.msg.Image, callback=self.rgb_callback, queue_size=10)
        #rospy.Subscriber('/'+self.camera+'/rgb/sk_tracks', sensor_msgs.msg.Image, callback=self.rgb_sk_callback, queue_size=10)
        rospy.Subscriber('/'+self.camera+'/depth/image' , sensor_msgs.msg.Image, self.depth_callback, queue_size=10)
        rospy.Subscriber("/robot_pose", Pose, callback=self.robot_callback, queue_size=10)
        rospy.Subscriber("/ptu/state", sensor_msgs.msg.JointState, callback=self.ptu_callback, queue_size=1)
        rospy.Subscriber("skeleton_data/state", skeleton_tracker_state, self.state_callback)


    def reinisialise(self):
        # directory to store the data
        self.date = str(datetime.datetime.now().date())
        self.dir1 = '/home/' + getpass.getuser() + '/SkeletonDataset/consent/' + self.date+'/'
        if not os.path.exists(self.dir1):
            print '  -create folder:',self.dir1
            os.makedirs(self.dir1)

        # flag to stop listeners
        self.listen_to = 1

        # flags to make sure we recived every thing
        self.requested_consent_flag = 0
        self._flag_robot = 0
        self._flag_rgb = 0
        #self._flag_rgb_sk = 0
        self._flag_depth = 0

    def reset_data(self):
        self.accumulate_data = {}  # accumulates multiple skeleton msgs
        self.accumulate_robot = {} # accumulates multiple robot msgs
        self.accumulate_robot = {}
        self.accumulate_rgb_images = {}
        #self.accumulate_rgb_sk_images = {}
        self.accumulate_depth_images = {}
        self.sk_mapping = {}       # holds state of users

        self.robot_pose = Pose()   # init pose of the robot
        self.ptu_pan = self.ptu_tilt = 0.0    # init pan tilt of the head xtion
        self.listen_to = 0


    def incremental_callback(self, msg):
        """accumulate all data until consent is requested and accepted"""
        if self._flag_robot and self._flag_rgb and self._flag_depth and self.requested_consent_flag is 0:
            if msg.uuid in self.sk_mapping:
                if self.sk_mapping[msg.uuid]["state"] is 'Tracking' and len(self.accumulate_data[msg.uuid]) <= self.max_num_frames:
                        self.accumulate_data[msg.uuid].append(msg)
                        robot_msg = robot_message(robot_pose = self.robot_pose, PTU_pan = self.ptu_pan, PTU_tilt = self.ptu_tilt)
                        self.accumulate_robot[msg.uuid].append(robot_msg)
                        self.accumulate_rgb_images[msg.uuid].append(self.rgb)
                        #self.accumulate_rgb_sk_images[msg.uuid].append(self.rgb_sk)
                        self.accumulate_depth_images[msg.uuid].append(self.xtion_img_d_rgb)


    def new_user_detected(self, msg):
        self.sk_mapping[msg.uuid] = {"state":'Tracking', "time":str(datetime.datetime.now().time()).split('.')[0]+'_'}
        self.accumulate_data[msg.uuid] = []
        self.accumulate_robot[msg.uuid] = []
        self.accumulate_rgb_images[msg.uuid] = []
        #self.accumulate_rgb_sk_images[msg.uuid] = []
        self.accumulate_depth_images[msg.uuid] = []


    def state_callback(self, msg):
        """Reads the state messages from the openNi tracker"""
        if msg.message == "Tracking":
            self.new_user_detected(msg)
        elif msg.message == "Out of Scene" and msg.uuid in self.sk_mapping:
            self.sk_mapping[msg.uuid]["state"] = "Out of Scene"
        elif msg.message == "Visible" and msg.uuid in self.sk_mapping:
            self.sk_mapping[msg.uuid]["state"] = "Tracking"
        elif msg.message == "Stopped tracking" and msg.uuid in self.accumulate_data and self.requested_consent_flag == 0:
            if len(self.accumulate_data[msg.uuid]) != 0:
                self._remove_stored_data(msg.userID, msg.uuid)

    def _remove_stored_data(self, subj, uuid):
        """when a uuid is stopped being tracked - delete their held data"""
        del self.sk_mapping[uuid]
        del self.accumulate_data[uuid]
        del self.accumulate_robot[uuid]
        del self.accumulate_rgb_images[uuid]
        #del self.accumulate_rgb_sk_images[uuid]
        del self.accumulate_depth_images[uuid]


    def consent_given_dump(self, uuid):
        """For a consented skeleton - dump all the rgb/rgb_sk/depth images to file"""

        # print ">>", self.sk_mapping[uuid]
        # print len(self.sk_mapping[uuid]), len(self.accumulate_data[uuid]), len(self.accumulate_robot[uuid]), len(self.accumulate_rgb_images[uuid]), len(self.accumulate_rgb_sk_images[uuid]), len(self.accumulate_depth_images[uuid])
        print "dumping data for %s%s" % (self.sk_mapping[uuid]['time'], uuid)

        """Loop over all the held data and save to disc"""
        for f, incr_msg in enumerate(self.accumulate_data[uuid]):
            if str(datetime.datetime.now().date()) != self.date:  		    #This will only happen if the action is called over night.
                print 'new day!'
                self.date = str(datetime.datetime.now().date())
                self.dir1 = '/home/' + getpass.getuser() + '/SkeletonDataset/no_consent/'+self.date+'/'
                print 'checking if folder exists:',self.dir1
                if not os.path.exists(self.dir1):
                    print '  -create folder:',self.dir1
                    os.makedirs(self.dir1)

            t = self.sk_mapping[uuid]['time']
            new_dir = self.dir1+self.date+'_'+t+uuid #+'_'+waypoint

            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                os.makedirs(new_dir+'/depth')
                os.makedirs(new_dir+'/robot')
                os.makedirs(new_dir+'/skeleton')
                os.makedirs(new_dir+'/rgb')
                os.makedirs(new_dir+'/rgb_sk')

            if os.path.exists(new_dir):
                # setup saving dir and frame
                d = new_dir+'/'

                f_ = f+1
                if f < 10:          f_str = '0000'+str(f_)
                elif f < 100:          f_str = '000'+str(f_)
                elif f < 1000:          f_str = '00'+str(f_)
                elif f < 10000:          f_str = '0'+str(f_)
                elif f < 100000:          f_str = str(f_)

                # save images
                cv2.imwrite(d+'rgb/rgb_'+f_str+'.jpg', self.accumulate_rgb_images[uuid][f])
                #cv2.imwrite(d+'rgb_sk/sk_'+f_str+'.jpg', self.accumulate_rgb_sk_images[uuid][f])
                cv2.imwrite(d+'depth/depth_'+f_str+'.jpg', self.accumulate_depth_images[uuid][f])


                # get robot data
                rob_incr_msg = self.accumulate_robot[uuid][f]
                x=float(rob_incr_msg.robot_pose.position.x)
                y=float(rob_incr_msg.robot_pose.position.y)
                z=float(rob_incr_msg.robot_pose.position.z)
                xo=float(rob_incr_msg.robot_pose.orientation.x)
                yo=float(rob_incr_msg.robot_pose.orientation.y)
                zo=float(rob_incr_msg.robot_pose.orientation.z)
                wo=float(rob_incr_msg.robot_pose.orientation.w)
                p = Point(x, y, z)
                q = Quaternion(xo, yo, zo, wo)
                robot = Pose(p,q)
                pan = float(rob_incr_msg.PTU_pan)
                tilt = float(rob_incr_msg.PTU_tilt)

                # write robot data in text file
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

                # save skeleton data in text file
                f1 = open(d+'skeleton/skl_'+f_str+'.txt','w')
                f1.write('time:'+str(incr_msg.time.secs)+'.'+str(incr_msg.time.nsecs)+'\n')
                for i in incr_msg.joints:
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


    def robot_callback(self, msg):
        if self.listen_to == 1:
            self.robot_pose = msg
            if self._flag_robot == 0:
                print ' >robot pose received'
                self._flag_robot = 1

    def ptu_callback(self, msg):
        if self.listen_to == 1:
            self.ptu_pan, self.ptu_tilt = msg.position

    def node_callback(self, msg):
        if self.listen_to == 1:
            self.current_node = msg.data
            if self._flag_node == 0:
                print ' >current node received'
                self._flag_node = 1

    def map_callback(self, msg):
        # get the topological map name
        self.map_info = msg.map
        self.topo_listerner.unregister()

    def rgb_callback(self, msg):
        if self.listen_to == 1:
            self.rgb_msg = msg   # to serve to the webserver - for consent
            rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self._flag_rgb is 0:
                print ' >rgb image received'
                self._flag_rgb = 1

    #def rgb_sk_callback(self, msg):
    #    if self.listen_to == 1:
    #        self.rgb_sk_msg = msg   # to serve to the webserver - for consent
    #        rgb_sk = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #        self.rgb_sk = cv2.cvtColor(rgb_sk, cv2.COLOR_RGB2BGR)
    #        if self._flag_rgb_sk is 0:
    #            print ' >rgb skel image recived'
    #            self._flag_rgb_sk = 1

    def depth_callback(self, msg):
        if self.listen_to == 1:
            # self.depth_msg = msg
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            self.xtion_img_d_rgb = depth_array*255
            if self._flag_depth is 0:
                print ' >depth image recived'
                self._flag_depth = 1

if __name__ == '__main__':
    rospy.init_node('skeleton_publisher', anonymous=True)

    record_rgb = rospy.get_param("~record_rgb", True)
    print "recording RGB images: %s" % record_rgb

    sk_manager = SkeletonManager(record_rgb)
    rospy.spin()
