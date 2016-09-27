#! /usr/bin/env python
import roslib
import sys, os
import rospy
import yaml
import actionlib
import rosbag
import getpass, datetime
import shutil

from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped, Pose
from std_srvs.srv import Empty, EmptyResponse
from skeleton_tracker.msg import skeleton_message
from skeleton_publisher_with_consent import SkeletonManagerConsent
from record_skeletons_action.msg import skeletonAction, skeletonActionResult
from consent_tsc.msg import ManageConsentAction, ConsentResult, ManageConsentGoal

from scitos_ptu.msg import *
import strands_gazing.msg
from mary_tts.msg import maryttsAction, maryttsGoal
from mongodb_store.message_store import MessageStoreProxy

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
        self.sk_publisher = SkeletonManagerConsent()

        # PTU state - based upon current_node callback
        self.ptu_action_client = actionlib.SimpleActionClient('/SetPTUState', PtuGotoAction)
        self.ptu_action_client.wait_for_server()

        # mongo store
        self.msg_store = MessageStoreProxy(database='message_store', collection='consent_images')
        
        self.number_of_frames_before_consent_needed = num_of_frames
        # gazing action server
        #self.gaze_client()
        self.publish_consent_pose = rospy.Publisher('skeleton_data/consent_pose', PoseStamped, queue_size = 10, latch=True)
                
        # topo nav move
        # self.nav_client()

        # speak
        self.speak()
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

        self.set_ptu_state(goal.waypoint)
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
                #        #self.gazeClient.send_goal(self.gazegoal)

                if len(incr_msgs) >= self.number_of_frames_before_consent_needed:
                    request_consent = 1
                    consented_uuid = uuid
                    self.sk_publisher.requested_consent_flag = 1  # stops the publisher storing data
                    self.load_images_to_view_on_mongo(uuid)       # uploads latest images to mongo

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


    def load_images_to_view_on_mongo(self, uuid):
    
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
        self.ptu_action_client.send_goal(ptu_goal)
        self.ptu_action_client.wait_for_result()

    def set_ptu_state(self, waypoint):
        ptu_goal = PtuGotoGoal();
        try:
            ptu_goal.pan = self.config[waypoint]['pan']
            ptu_goal.tilt = self.config[waypoint]['tilt']
            ptu_goal.pan_vel = self.config[waypoint]['pvel']
            ptu_goal.tilt_vel = self.config[waypoint]['tvel']
            self.ptu_action_client.send_goal(ptu_goal)
            self.ptu_action_client.wait_for_result()
        except KeyError:
            self.reset_ptu()

    def go_back_to_where_I_came_from(self):
        if self.nav_goal_waypoint is not None and self.nav_goal_waypoint != self.config[self.nav_goal_waypoint]['target']:
            try:
                # self.navgoal.target = self.config[self.nav_goal_waypoint]['target']
                self.navgoal.target = self.nav_goal_waypoint
            except:
                print "nav goal not set - staying at %s" % self.navgoal.target
            self.navClient.send_goal(self.navgoal)
            self.navClient.wait_for_result()

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

    def nav_client(self):
        rospy.loginfo("Creating nav client")
        self.navClient = actionlib.SimpleActionClient('topological_navigation', topological_navigation.msg.GotoNodeAction)
        self.navClient.wait_for_server(timeout=rospy.Duration(1))
        self.navgoal = topological_navigation.msg.GotoNodeGoal()

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
    
