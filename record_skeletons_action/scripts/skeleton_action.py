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
import random
import numpy as np

from tf.transformations import quaternion_from_euler
from std_msgs.msg import String, Header, Int32
from geometry_msgs.msg import PoseStamped, Pose, Point32, Polygon, PoseArray
from sensor_msgs.msg import JointState

from std_srvs.srv import Empty, EmptyResponse
from soma2_msgs.msg import SOMA2Object, SOMA2ROIObject
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

from viper_ros.msg import ViewInfo
from shapely.geometry import Polygon, Point

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

        self.reduce_frame_rate_by = rospy.get_param("~frame_rate_reduce", 3)
        self.number_of_frames_before_consent_needed = rospy.get_param("~consent_num_frames", num_of_frames)
        rospy.loginfo("Required num of frames: %s" % self.number_of_frames_before_consent_needed)


        # skeleton publisher class (logs data given a detection)
        self.sk_publisher = SkeletonManagerConsent()
        self.soma_map = rospy.get_param("~soma_map", "collect_data_map_cleaned")
        self.soma_config = rospy.get_param("~soma_config", "test")

        # robot pose
        self.listen_to_robot_pose = 1
        rospy.Subscriber("/robot_pose", Pose, callback=self.robot_callback, queue_size=10)

        # nav client
        self.nav_client = actionlib.SimpleActionClient('monitored_navigation', MonitoredNavigationAction)
        rospy.loginfo("Wait for monitored navigation server")
        self.nav_client.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Done")

        # PTU client
        self.ptu_height = 1.70 # assume camera is 1.7m
        self.ptu_client = actionlib.SimpleActionClient('SetPTUState', PtuGotoAction)
        rospy.loginfo("Wait for PTU action server")
        self.ptu_client.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Done")

        # mongo store
        self.msg_store = MessageStoreProxy(database='message_store', collection='consent_images')
        self.soma_id_store = MessageStoreProxy(database='message_store', collection='soma_activity_ids_list')
        self.soma_store = MessageStoreProxy(database="soma2data", collection="soma2")
        self.views_msg_store = MessageStoreProxy(collection='activity_view_stats')
        self.soma_roi_store = MessageStoreProxy(database='soma2data', collection='soma2_roi')

        # gazing action server
        self.gaze_client()
        self.publish_consent_pose = rospy.Publisher('skeleton_data/consent_pose', PoseStamped, queue_size = 10, latch=True)

        # speak
        self.speak()

        # auto select viewpoint for recording
        self.view_dist_thresh_low = 1.5
        self.view_dist_thresh_high = 2.5
        self.possible_nav_goals = []

        # visualizing the view point goal in RVIZ
        self.pub_viewpose = rospy.Publisher('activity_view_goal', PoseStamped, queue_size=10)
        self.pub_all_views = rospy.Publisher('activity_possible_view_goals', PoseArray, queue_size=10)
        print ">>initialised recording action"

    def execute_cb(self, goal):
        self.listen_to_robot_pose = 1

        duration = goal.duration
        start = rospy.Time.now()
        end = rospy.Time.now()
        consent_msg = "nothing"
        rospy.loginfo("Observe ROI: %s", goal.roi_id)

        observe_polygon = self.get_roi_to_observe(goal.roi_id, goal.roi_config)
        if observe_polygon == None:
            return self._as.set_preempted()

        self.get_soma_objects(observe_polygon)
        self.create_possible_navgoals()

        self.generate_viewpoints()
        nav_fail = self.goto()

        consented_uuid = ""
        request_consent = 0

        rospy.loginfo("init recording page and skeleton pub")
        self.signal_start_of_recording()
        self.sk_publisher.reinisialise()
        self.sk_publisher.max_num_frames =  self.number_of_frames_before_consent_needed

        while (end - start).secs < duration.secs and request_consent == 0:
            if self._as.is_preempt_requested():
                 break

            # print "1 ", self.sk_publisher.accumulate_data.keys()
            for cnt, (uuid, incr_msgs) in enumerate(self.sk_publisher.accumulate_data.items()):
                # print ">>", len(self.sk_publisher.accumulate_data[uuid]), uuid

                # publish the location of a person as a gaze request
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

                    rospy.loginfo("breaking loop for: %s" % consented_uuid)
                    self.reset_ptu()
                    self.speaker.send_goal(maryttsGoal(text=self.speech))

                    new_duration = duration.secs - (end - start).secs
                    consent_msg = self.consent_client(new_duration)
                    rospy.loginfo("consent returned: %s: %s" % (consent_msg, consented_uuid))
                    # break

            end = rospy.Time.now()

        # after the action reset ptu and stop publisher
        rospy.loginfo("exited loop. consent=%s" %consent_msg)

        # LOG THE STATS TO MONGO
        res = (request_consent, consent_msg)
        observed_roi = goal.roi_config + ":" + goal.roi_id
        self.log_view_info(res, nav_fail, observed_roi, start, end)

        # reset everything:
        self.reset_everything()

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

    def log_view_info(self, res, nav_fail, roi, starttime=None, endstime=None):
        """Given the action is over, log the viewpoint and bool output of the recording (successfull?)"""

        vinfo = ViewInfo()
        vinfo.waypoint = roi
        vinfo.map_name  = rospy.get_param('/topological_map_name', "no_map_name")
        vinfo.mode = "activity_rec"
        vinfo.starttime = int(starttime.to_sec())
        vinfo.robot_pose = self.selected_robot_pose

        vinfo.ptu_state = JointState(name=['pan', 'tilt'], position=[self.ptu_values[0], self.ptu_values[1]],
            velocity= [self.ptu_values[2], self.ptu_values[3]])

        vinfo.nav_failure = nav_fail

        (request_consent, consent_msg) = res
        vinfo.success = False
        if consent_msg == "everything": vinfo.success = True

        vinfo.soma_objs = [self.selected_object, repr(self.selected_object_pose)]
        rospy.loginfo("logged view stats: nav:%s, record:%s, consent:%s." % (vinfo.nav_failure, bool(request_consent), vinfo.success))
        self.views_msg_store.insert(vinfo)


    def learn_best_viewpoint(self):
        """Query the database and retrieve the good views:
        1. where nav did fail
        2. which recorded and received consenst.
        """
        query = {"mode":"activity_rec", "nav_failure":False, "success":True}
        ret = self.views_msg_store.query(ViewInfo._type)
        for view,meta in ret:
            print ">>", view.soma_objs[0], view.soma_objs[1].position



    def goto(self):
        """
        Given a viewpoint - send the robot there, and fix the PTU angle
        returns: Nav_Failure stats.
        """
        inds = range(len(self.possible_poses))
        random.shuffle(inds)

        # add one more possible pose - for the waypoint pose
        inds.append(len(self.possible_poses))
        self.possible_poses.append(self.robot_pose)

        for cnt, ind in enumerate(inds):
            """For all possible viewpoints, try to go to one - if fails, loop."""

            if self._as.is_preempt_requested():
                return

            # Publish ViewPose for visualisation
            s = PoseStamped()
            s.header.frame_id = "/map"
            s.pose = self.possible_poses[ind]
            self.pub_viewpose.publish(s)

            rospy.loginfo("NAV GoTo: x: %s y: %s.", self.possible_poses[ind].position.x, self.possible_poses[ind].position.y)
            goal = MonitoredNavigationGoal()
            goal.action_server = 'move_base'
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.get_rostime() #rospy.Time.now()

            goal.target_pose.pose = self.possible_poses[ind]
            self.selected_robot_pose = goal.target_pose.pose
            self.nav_client.send_goal(goal)
            self.nav_client.wait_for_result()
            res = self.nav_client.get_result()

            if res.outcome != 'succeeded':
                rospy.loginfo("nav goal fail: %s" % str(res.outcome))
                continue
            else:
                rospy.loginfo("Reached nav goal: %s" % str(res.outcome))

                obj = self.selected_object_pose
                dist_z = abs(self.ptu_height - obj.position.z)
                p = self.possible_poses[ind]
                dist = abs(math.hypot((p.position.x - obj.position.x), (p.position.y - obj.position.y)))
                ptu_tilt = math.degrees(math.atan2(dist_z, dist))
                rospy.loginfo("ptu: 175, ptu tilt: %s" % ptu_tilt)
                self.set_ptu_state(pan=175, tilt=ptu_tilt)
                return False

        """IF NO VIEWS work (even the waypoint?)- try looking on mongo for one that has previously worked?"""
        return True


    def generate_viewpoints(self):
        """
        Given a set of random viewpoints in a roi, filter those too close, and define a yaw for each.
            1. needs to be further away than the lower threshold
            2. must be in the same roi as WP
            3. create yaw of robot view
            N/A. as a backup, calculate the viewpoint from the waypoint
        """
        view_goals = PoseArray()
        view_goals.header.frame_id = "/map"
        poses, yaws, dists = [], [], []

        obj = self.selected_object_pose
        for cnt, p in enumerate(self.possible_nav_goals.goals.poses):
            x_dist = p.position.x - obj.position.x
            y_dist = p.position.y - obj.position.y

            # print "xDist %s, yDist %s" %(x_dist, y_dist)
            dist = abs(math.hypot(x_dist, y_dist))
            if dist > self.view_dist_thresh_low:
                if self.robot_polygon.contains(Point([p.position.x, p.position.y])):
                    # yaw = math.atan2(y_dist, x_dist)
                    p = self.add_quarternion(p, x_dist, y_dist)
                    poses.append(p)
                    # yaws.append(yaw)
                    # dists.append( (x_dist, y_dist) )

        view_goals.poses = poses
        self.pub_all_views.publish(view_goals)
        self.possible_poses = poses
        # self.possible_yaws = yaws

        # add the viewpoint from the waypoint - as a back up if all others fail
        x_dist = self.robot_pose.position.x - obj.position.x
        y_dist = self.robot_pose.position.y - obj.position.y
        dist = abs(math.hypot(x_dist, y_dist))
        self.robot_pose = self.add_quarternion(self.robot_pose, x_dist, y_dist)

    def add_quarternion(self, pose, x_dist, y_dist):
        """Calculate the robots orientation given a point
        """
        roll = pitch = 0
        yaw = math.atan2(y_dist, x_dist)

        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        return pose

    def create_possible_navgoals(self, centre=None, N=10, r=None):
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

    def get_robot_roi(self):
        """Find the roi of the robot at it's waypoint - in case no target ROI is passed
        Make sure the randomly generated viewpoints are all in this roi also - i.e. this room
        """
        self.robot_polygon = None
        for (roi, meta) in self.soma_roi_store.query(SOMA2ROIObject._type):
            if roi.map_name != self.soma_map: continue
            if roi.config != self.soma_config: continue
            if roi.geotype != "Polygon": continue
            polygon = Polygon([ (p.position.x, p.position.y) for p in roi.posearray.poses])

            if polygon.contains(Point([self.robot_pose.position.x, self.robot_pose.position.y])):
                rospy.loginfo("Robot in ROI: %s" %roi.type)
                return polygon
        rospy.logwarn("This waypoint is not defined in a ROI")
        return None

    def get_roi_to_observe(self, roi_id, roi_config):
        """Get the roi to observe.
           Select objects to observe based upon this region - i.e. the recommended interesting roi
        """
        self.robot_polygon = self.get_robot_roi()
        observe_polygon = None

        if roi_id != "":
            for (roi, meta) in self.soma_roi_store.query(SOMA2ROIObject._type):
                if roi.map_name != self.soma_map: continue
                if roi.config != roi_config: continue
                if roi.roi_id != roi_id: continue
                if roi.geotype != "Polygon": continue
                observe_polygon = polygon
                rospy.loginfo("Observe ROI: %s" %roi.type)
            if observe_polygon ==None:
                rospy.logwarn("ROI given to observe not found")
        else:
            rospy.logwarn("No ROI given to observe - use robot ROI")
            observe_polygon = self.robot_polygon
        return observe_polygon

    def get_soma_objects(self, target_polygon):
        """srv call to mongo and get the list of object IDs to use and locations"""

        ids = self.soma_id_store.query(Int32._type)
        ids = [id_[0].data for id_ in ids]
        print "SOMA IDs to observe >> ", ids
        objs = self.soma_store.query(SOMA2Object._type, {"id":{"$exists":"true"}, "$where":"this.id in %s" %ids})
        # dummy_objects = [(-52.29, -5.62, 1.20), (-50.01, -5.49, 1.31), (-1.68, -5.94, 1.10)]

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
        'Double_doors_112': (-8.365, -18.440, 1.021),
        'robot_lab_Majd_desk': (-7.3, -33.5, 1.2),
        'robot_lab_Baxter_desk':(-4.4, -31.8, 1.2),
        'robot_lab_Poster':(-4.3, 34.0, 1.2)
        }
        # reduce all the objects to those in the same region as the robot

        objects_in_roi = []
        try:
            for (obj_name, (x,y,z)) in all_dummy_objects.items():
                if target_polygon.contains(Point([x, y])):
                    pose = Pose()
                    pose.position.x = x
                    pose.position.y = y
                    pose.position.z = z
                    objects_in_roi.append((obj_name, pose))

            r = random.randint(0,len(objects_in_roi))-1
            (self.selected_object, self.selected_object_pose) = objects_in_roi[r]
            rospy.loginfo("selected object to view: %s. nav_target: (%s, %s)" % (self.selected_object, objects_in_roi[r][1].position.x, objects_in_roi[r][1].position.y))
            self.selected_object_id = r
        except AttributeError:
            rospy.loginfo("Robot not in a ROI - cannot select a view point of an object in that roi")

            selected_object_pose
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
        """Send one RGB image to mongodb to be used by the webserver,
        to ask for consent on the main PCs screen.
        """
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
        return


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

    def reset_ptu(self):
        ptu_goal = PtuGotoGoal();
        ptu_goal.pan = 0
        ptu_goal.tilt = 0
        ptu_goal.pan_vel = 30
        ptu_goal.tilt_vel = 30
        self.ptu_client.send_goal(ptu_goal)
        self.ptu_client.wait_for_result()

    def reset_everything(self):
        self.reset_ptu()
        self.possible_poses = []
        self.selected_pose = None
        self.possible_nav_goals = []

    def set_ptu_state(self, pan=175, tilt=10):
        ptu_goal = PtuGotoGoal();

        ptu_goal.pan = pan
        ptu_goal.tilt = tilt
        ptu_goal.pan_vel = 30
        ptu_goal.tilt_vel = 30

        self.ptu_values = (ptu_goal.pan, ptu_goal.tilt, ptu_goal.pan_vel, ptu_goal.tilt_vel)
        self.ptu_client.send_goal(ptu_goal)
        self.ptu_client.wait_for_result()

    # def go_back_to_where_I_came_from(self):
        # if self.nav_goal_waypoint is not None and self.nav_goal_waypoint != self.config[self.nav_goal_waypoint]['target']:
            # try:
                # # self.navgoal.target = self.config[self.nav_goal_waypoint]['target']
                # self.navgoal.target = self.nav_goal_waypoint
            # except:
                # print "nav goal not set - staying at %s" % self.navgoal.target
            # self.navClient.send_goal(self.navgoal)
            # self.navClient.wait_for_result()

    def nav_goals_client(self, roi):
        rospy.loginfo("waiting for nav goals client")
        rospy.wait_for_service('/nav_goals')
        proxy = rospy.ServiceProxy('/nav_goals', NavGoals)
        req = NavGoalsRequest(n=1000, inflation_radius=0.5, roi=roi)
        self.possible_nav_goals = proxy(req)  # returned list of poses

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

    def robot_callback(self, msg):
        if self.listen_to_robot_pose == 1:
            self.robot_pose = msg
        self.listen_to_robot_pose = 0

if __name__ == "__main__":
    rospy.init_node('skeleton_action_server')

    skeleton_server("record_skeletons")
    rospy.spin()
