#! /usr/bin/env python
import roslib
import tf
import sys, os
import rospy
import yaml
import actionlib
import rosbag
import getpass, datetime
import shutil
import math
from random import randint
from tf.transformations import quaternion_from_euler

from std_msgs.msg import String, Header, Int32
from geometry_msgs.msg import PoseStamped, Pose, Point32, Polygon, PoseArray
from std_srvs.srv import Empty, EmptyResponse
from soma2_msgs.msg import SOMA2Object
from skeleton_tracker.msg import skeleton_message
from skeleton_publisher_with_consent import SkeletonManagerConsent
from record_skeletons_action.msg import skeletonAction, skeletonActionResult
from consent_tsc.msg import ManageConsentAction, ConsentResult, ManageConsentGoal
from strands_navigation_msgs.msg import MonitoredNavigationAction, MonitoredNavigationGoal

from scitos_ptu.msg import *
import strands_gazing.msg
import topological_navigation.msg
from mary_tts.msg import maryttsAction, maryttsGoal
from mongodb_store.message_store import MessageStoreProxy
from nav_goals_generator.srv import NavGoals, NavGoalsRequest, NavGoalsResponse

from viper_ros.navigation import GoTo

class skeleton_server(object):
    def __init__(self, name="record_skeletons", num_of_frames=1000):
        """Action Server to observe a location for a duration of time.
           Record humans being detected
           If > a number of frames, request consent to save images.
        """

        rospy.loginfo("Skeleton Recording - starting an action server")
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, skeletonAction, \
                                                    execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self.load_config()
        self.number_of_frames_before_consent_needed = num_of_frames

        self.sk_publisher = SkeletonManagerConsent()

        # nav client
        self.nav_client = actionlib.SimpleActionClient('monitored_navigation', MonitoredNavigationAction)
        rospy.loginfo("Wait for monitored navigation server")
        self.nav_client.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Done")

        # PTU client
        self.ptu_client = actionlib.SimpleActionClient('SetPTUState', PtuGotoAction)
        rospy.loginfo("Wait for PTU action server")
        self.ptu_client.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Done")

        # mongo store
        self.msg_store = MessageStoreProxy(database='message_store', collection='consent_images')
        self.soma_id_store = MessageStoreProxy(database='message_store', collection='soma_activity_ids_list')
        self.soma_store = MessageStoreProxy(database="soma2data", collection="soma2")

        # gazing action server
        self.gaze_client()
        self.publish_consent_pose = rospy.Publisher('skeleton_data/consent_pose', PoseStamped, queue_size = 10, latch=True)

        # topo nav move
        # self.nav_client()

        # speak
        self.speak()

        # auto select viewpoint for recording
        self.view_dist_thresh_low = 1.9
        self.view_dist_thresh_high = 2
        self.possible_views = []

        # visualizing the view point goal in RVIZ
        self.pub_viewpose = rospy.Publisher('activity_view_goal', PoseStamped, queue_size=10)
        self.pub_all_views = rospy.Publisher('activity_possible_view_goals', PoseArray, queue_size=10)

        print ">>initialised recording action"


    def execute_cb(self, goal):
        print "send `start recording` page..."
        self.signal_start_of_recording()
        self.sk_publisher.reinisialise()
        self.sk_publisher.max_num_frames =  self.number_of_frames_before_consent_needed

        duration = goal.duration
        start = rospy.Time.now()
        end = rospy.Time.now()
        consent_msg = "nothing"
        print "GOAL:", goal

        self.get_soma_objects()
        self.create_possible_views()
        self.select_a_viewpoint()

        # self.send_robot_to_viewpoint(goal.waypoint, start, end+duration)

        self.set_ptu_state()
        consented_uuid = ""
        request_consent = 0

        while (end - start).secs < duration.secs and request_consent == 0:
            if self._as.is_preempt_requested():
                 break

            for cnt, (uuid, incr_msgs) in enumerate(self.sk_publisher.accumulate_data.items()):
                #print ">>", len(self.sk_publisher.accumulate_data[uuid]), uuid

                #publish the location of a person as a gaze request
                if cnt == 0 and len(incr_msgs) > 0:
                    if incr_msgs[-1].joints[0].name == 'head':
                        head = Header(frame_id='head_xtion_depth_optical_frame')
                        look_at_pose = PoseStamped(header = head, pose=incr_msgs[-1].joints[0].pose)
                        self.publish_consent_pose.publish(look_at_pose)
                        # self.gazeClient.send_goal(self.gazegoal)

                if len(incr_msgs) >= self.number_of_frames_before_consent_needed:
                    request_consent = 1
                    consented_uuid = uuid
                    self.sk_publisher.requested_consent_flag = 1  # stops the publisher storing data
                    self.upload_images_to_mongo(uuid)       # uploads latest images to mongo

                    print "breaking loop for: %s" % consented_uuid
                    self.reset_ptu()
                    self.speaker.send_goal(maryttsGoal(text=self.speech))

                    new_duration = duration.secs - (end - start).secs
                    consent_msg = self.consent_client(new_duration)
                    print "consent returned: %s: %s" % (consent_msg, consented_uuid)
                    # break

            end = rospy.Time.now()

        # after the action reset ptu and stop publisher
        print "exited loop - %s" %consent_msg
        self.reset_ptu()

        # if no skeleton was recorded for the threshold
        if request_consent is 0:
            self.return_to_main_webpage()

        if self._as.is_preempt_requested():
            print "The action is being preempted, cancelling everything. \n"
            self.return_to_main_webpage()
            return self._as.set_preempted()
        self._as.set_succeeded(skeletonActionResult())

        if consent_msg is "everything":
	        self.sk_publisher.consent_given_dump(consented_uuid)

        self.sk_publisher.reset_data()
        print "finished action\n"

    def send_robot_to_viewpoint(self, waypoint, starttime=None, endstime=None):
        """
        Given a viewpoint - send the robot there, and fix the PTU angle
        """

        """ USE VIPER GoTo"""
        goal = GoToGoal(
            waypoint = waypoint, mode = "human", starttime = starttime, robot_pose = self.selected_pose, ptu_state = None
        )


     # 	rospy.loginfo("GOTO: x: %s y: %s", self.selected_pose.position.x, self.selected_pose.position.y)
     #  	goal = MonitoredNavigationGoal()
    	# goal.action_server = 'move_base'
     #  	goal.target_pose.header.frame_id = 'map'
     #  	goal.target_pose.header.stamp = rospy.get_rostime() #rospy.Time.now()
        #
     #  	goal.target_pose.pose = self.selected_pose
     #  	self.nav_client.send_goal(goal)
     #  	self.nav_client.wait_for_result()
        # res = self.nav_client.get_result()
        # rospy.loginfo("Nav Client Result: %s", str(res))
        #
        # if res.outcome != 'succeeded':
        #     vinfo = ViewPointInfo()
        #
        #     vinfo.waypoint = waypoint
        #     vinfo.map_name  = rospy.get_param('/topological_map_name', "no_map_name")
        #     vinfo.mode = "human"
        #     vinfo.starttime = starttime
        #     vinfo.robot_pose = self.selected_pose
        #     vinfo.ptu_state = ptu_state
        #     vinfo.nav_failure = True
        #     vinfo.success = False
        #     vinfo.soma_objs = []
        #     self.msg_store.insert(vinfo)
        # return 'aborted'

    def select_a_viewpoint(self):
        """
        Given a set of random viewpoints in a roi, select one.
            1. needs to be further away than the lower threshold
            2. must be in the same roi as WP
        """

        view_goals = PoseArray()
        view_goals.header.frame_id = "/map"
        poses, yaws, dists = [], [], []

        obj = self.selected_object_pose
        for cnt, p in enumerate(self.possible_views.goals.poses):
            x_dist = p.position.x - obj.position.x
            y_dist = p.position.y - obj.position.y

            # print "xDist %s, yDist %s" %(x_dist, y_dist)
            dist = abs(math.hypot(x_dist, y_dist))

            if dist > self.view_dist_thresh_low:

                """CHECK THAT IT's IN THE ROI BEFPRE SELECTING:"""
                poses.append(p)
                yaw = math.atan2(y_dist, x_dist)
                yaws.append(yaw)
                dists.append( (x_dist, y_dist) )

        r = randint(0,len(poses))-1
        print "ViewPoint: (%s,%s). Yaw: %s rads. Dist: %s" % (poses[r].position.x, poses[r].position.y, yaws[r], dists[r])

        self.selected_pose = poses[r]
        view_goals.poses = poses
        self.pub_all_views.publish(view_goals)

        """What happens if a pose is not selected: go to waypoint?"""

        roll = pitch = 0
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaws[r])

        self.selected_pose.orientation.x = quaternion[0]
        self.selected_pose.orientation.y = quaternion[1]
        self.selected_pose.orientation.z = quaternion[2]
        self.selected_pose.orientation.w = quaternion[3]

        # Publish ViewPose for visualisation
        s = PoseStamped()
        s.header.frame_id = "/map"
        s.pose = self.selected_pose
        self.pub_viewpose.publish(s)


    def create_possible_views(self, centre=None, N=10, r=None):
        """Given a soma object - create candidate viewpoints around object.
        Keep a list of successful views - to use again.
        N = number of verticies for ROI
        r = radius of polygon
        """

        if centre == None:
            centre_x = self.selected_object_pose.position.x
            centre_y = self.selected_object_pose.position.y
        if r == None: r = self.view_dist_thresh_high

        pi = math.pi
        roi = Polygon()
        roi.points = []
        for n in xrange(N):
            p = Point32()
            p.x = r * math.cos(2*pi*n/float(N)) + centre_x
            p.y = r * math.sin(2*pi*n/float(N)) + centre_y
            p.z = 0
            roi.points.append(p)
        self.nav_goals_client(roi)


    def get_soma_objects(self):
        """srv call to mongo and get the list of object IDs to use and locations"""

        ids = self.soma_id_store.query(Int32._type)
        ids = [id_[0].data for id_ in ids]
        print "SOMA IDs to observe >> ", ids

        objs = self.soma_store.query(SOMA2Object._type, message_query={"id":{"$exists":"true"}, "$where":"this.id in %s" %ids})
        dummy_objects = [(-52.29, -5.62, 1.20), (-50.01, -5.49, 1.31), (-1.68, -5.94, 1.10)]

        all_dummy_objects = {
        'Printer_console_11': (-8.957, -17.511, 1.1),                           # fixed
        'Printer_paper_tray_110': (-9.420, -18.413, 1.132),                     # fixed
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
        dummy_objects = all_dummy_objects.values()
        # print "SOM objs in region: %s " % dummy_objects
        r = randint(0,len(dummy_objects))-1

        for cnt, (x,y,z) in enumerate(dummy_objects):
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            objs.append(pose)

        self.selected_object_pose = objs[r]
        print "target: (%s, %s) " % (objs[r].position.x, objs[r].position.y)
        return

    def consent_client(self, duration):
        rospy.loginfo("Creating consent client")
        ret = None
        start = rospy.Time.now()
        end = rospy.Time.now()
        try:
            consent_client = actionlib.SimpleActionClient('manage_consent', ManageConsentAction)
            if consent_client.wait_for_server(timeout=rospy.Duration(10)):
                goal = ManageConsentGoal()
                consent_client.send_goal(goal)

                # check whether you've been preempted, shutdown etc. while waiting for consent
                while not self._as.is_preempt_requested() and (end - start).secs < duration:

                    if consent_client.wait_for_result(timeout = rospy.Duration(1)):
                        result = consent_client.get_result()
                        int_result = result.result.result

                        if int_result == ConsentResult.DEPTH_AND_RGB:
                            # print 'depth+rgb'
                            ret = "everything"
                        else:
                            ret = "nothing"
                        #elif int_result == ConsentResult.DEPTH_ONLY:
                        #    # print 'just depth'
                        #    ret = "depthskel"
                        break
                    end = rospy.Time.now()

                if (end - start).secs >= duration:
                    print "timed out"

                if self._as.is_preempt_requested():
                    consent_client.cancel_all_goals()
            else:
                rospy.logwarn('No manage consent server')
        except Exception, e:
            rospy.logwarn('Exception when trying to manage consent: %s' % e)
        return ret


    def upload_images_to_mongo(self, uuid):

        rgb = self.sk_publisher.rgb_msg
        #rgb_sk = self.sk_publisher.rgb_sk_msg
        # depth = self.sk_publisher.depth_msg  # not created anymore

        # Skeleton on rgb background
        #query = {"_meta.image_type": "rgb_sk_image"}
        #self.msg_store.update(message=rgb_sk, meta={'image_type':"rgb_sk_image"}, message_query=query, upsert=True)

        query = {"_meta.image_type": "rgb_image"}
        self.msg_store.update(message=rgb, meta={'image_type':"rgb_image"}, message_query=query, upsert=True)

        # Skeleton on depth background
        # depth = self.sk_publisher.accumulate_rgb_images[uuid][-1]
        # query = {"_meta.image_type": "depth_image"}
        # depth_img_to_mongo = self.msg_store.update(message=self.depth_msg, meta={'image_type':"depth_image"}, message_query=query, upsert=True)



    def signal_start_of_recording(self):
        rospy.wait_for_service('signal_recording_started', timeout=10)
        signal_recording_started = rospy.ServiceProxy('signal_recording_started', Empty)
        # tell the webserver to say we've started recording
        signal_recording_started()

    def return_to_main_webpage(self):
        rospy.wait_for_service('return_to_main_webpage', timeout=10)
        main_webpage_return = rospy.ServiceProxy('return_to_main_webpage', Empty)
        # tell the webserver to go back to the main page - if no consent was requested
        main_webpage_return()

    def load_config(self):
        self.filepath = os.path.join(roslib.packages.get_pkg_dir("record_skeletons_action"), "config")
        try:
            self.config = yaml.load(open(os.path.join(self.filepath, 'config.ini'), 'r'))
            print "config file loaded", self.config.keys()
        except:
            print "no config file found"

    def reset_ptu(self):
        ptu_goal = PtuGotoGoal();
        ptu_goal.pan = 0
        ptu_goal.tilt = 0
        ptu_goal.pan_vel = 30
        ptu_goal.tilt_vel = 30
        self.ptu_client.send_goal(ptu_goal)
        self.ptu_client.wait_for_result()

    def set_ptu_state(self, waypoint=None):
        ptu_goal = PtuGotoGoal();

        try:
            ptu_goal.pan = self.config[waypoint]['pan']
            ptu_goal.tilt = self.config[waypoint]['tilt']
            ptu_goal.pan_vel = self.config[waypoint]['pvel']
            ptu_goal.tilt_vel = self.config[waypoint]['tvel']
        except:
            ptu_goal.pan = 0 # 175
            ptu_goal.tilt = 10
            ptu_goal.pan_vel = 20
            ptu_goal.tilt_vel = 20

        self.ptu_client.send_goal(ptu_goal)
        self.ptu_client.wait_for_result()
        # self.reset_ptu()


    def go_back_to_where_I_came_from(self):
        if self.nav_goal_waypoint is not None and self.nav_goal_waypoint != self.config[self.nav_goal_waypoint]['target']:
            try:
                # self.navgoal.target = self.config[self.nav_goal_waypoint]['target']
                self.navgoal.target = self.nav_goal_waypoint
            except:
                print "nav goal not set - staying at %s" % self.navgoal.target
            self.navClient.send_goal(self.navgoal)
            self.navClient.wait_for_result()

    def nav_goals_client(self, roi):
        rospy.wait_for_service('/nav_goals')
        proxy = rospy.ServiceProxy('/nav_goals', NavGoals)
        req = NavGoalsRequest(n=1000, inflation_radius=0.5, roi=roi)
        self.possible_views = proxy(req)  # returned list of poses

    def consent_ret_callback(self, msg):
        print "-> consent ret callback", self.request_sent_flag, msg
        if self.request_sent_flag != 1: return
        self.consent_ret=msg
        self.speaker.send_goal(maryttsGoal(text="Thank you"))

    def gaze_client(self):
        rospy.loginfo("Creating gaze client")
        _as = actionlib.SimpleActionClient('gaze_at_pose', strands_gazing.msg.GazeAtPoseAction)
        _as.wait_for_server()
        gazegoal = strands_gazing.msg.GazeAtPoseGoal()
        gazegoal.topic_name = '/skeleton_data/consent_pose'
        gazegoal.runtime_sec = 60
        _as.send_goal(gazegoal)

    def speak(self):
        self.speaker = actionlib.SimpleActionClient('/speak', maryttsAction)
        got_server = self.speaker.wait_for_server(rospy.Duration(1))
        while not got_server:
            rospy.loginfo("Data Consent is waiting for marytts action...")
            got_server = self.speaker.wait_for_server(rospy.Duration(1))
            if rospy.is_shutdown():
                return
        self.speech = "Please  may  I  get  your  consent  to  store  video   I  just  recorded."



if __name__ == "__main__":
    rospy.init_node('skeleton_action_server')

    num_of_frames = 600
    skeleton_server("record_skeletons", num_of_frames)
    rospy.spin()
