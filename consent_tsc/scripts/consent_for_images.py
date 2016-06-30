#!/usr/bin/env python
import sys, os
import rospy, roslib
import cv2
import numpy as np
import strands_webserver.client_utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from mongodb_store.message_store import MessageStoreProxy
import actionlib
from consent_tsc.msg import ManageConsentAction, ConsentResult
from consent_tsc.srv import UserConsent
from std_srvs.srv import Empty, EmptyResponse

class ManageConsentWebpage(object):

    def __init__(self):
       
        self.display_no = rospy.get_param("~display", 0)
        self.filepath = os.path.join(roslib.packages.get_pkg_dir("consent_tsc"), "webserver")
        # how long to wait before I give up
        self.timeout = 360

        # used to track recording state for debugging
        self.recording = False
        rospy.Service('signal_recording_started', Empty, self.started_recording_callback)


        self.consent_as = actionlib.SimpleActionServer('manage_consent', ManageConsentAction, self.execute_cb, False)
        self.consent_as.start()


        # http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
        # strands_webserver.client_utils.set_http_root(http_root)
        # rospy.sleep(1)
        # strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')
        # #subscribers
        # rospy.Subscriber("/skeleton_data/consent_req", String, callback=self.consent_req_callback, queue_size=1)
        # rospy.Subscriber("/skeleton_data/consent_ret", String, callback=self.consent_ret_callback, queue_size=1)
        # rospy.Subscriber("/skeleton_data/recording_started", String, callback= self.started_recording_callback, queue_size=1)

    def setup(self, database='message_store', collection='consent_images'):
        self.msg_store = MessageStoreProxy(collection=collection, database=database)
        self.bridge = CvBridge()
        self.user_consent_srv = rospy.Service('user_consent_provided', UserConsent, self.user_consent_received)
        self.user_consent = None

    def teardown(self):
        self.user_consent_srv.shutdown()
        self.user_consent = None
        http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
        strands_webserver.client_utils.set_http_root(http_root)        
        strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')


    def user_consent_received(self, req):
        print str(req)
        rospy.loginfo('user clicked something')
        self.user_consent = ConsentResult(result=req)

    def execute_cb(self, goal):
        if not self.recording:
            rospy.logwarn('Consent management server was called when not recording. Strange things may happen')
        
        self.setup()
        try:
            self.get_imgs()
        except Exception, e:
            rospy.logwarn('Exception while generating images: %s' % e)    
        

        strands_webserver.client_utils.set_http_root(self.filepath)        
        strands_webserver.client_utils.display_relative_page(self.display_no, 'consent.html')

        end_at = rospy.get_rostime() + rospy.Duration(self.timeout)

        while rospy.get_rostime() < end_at and self.user_consent is None and not self.consent_as.is_preempt_requested() and not rospy.is_shutdown():
            rospy.sleep(0.2)

        if self.consent_as.is_preempt_requested():
            self.consent_as.set_preempted()
        elif self.user_consent is not None:
            self.consent_as.set_succeeded(self.user_consent)
        else:
            self.consent_as.set_aborted()

        self.teardown()




    def started_recording_callback(self, req):
        print "serve recording webpage"
        strands_webserver.client_utils.set_http_root(self.filepath)        
        strands_webserver.client_utils.display_relative_page(self.display_no, 'recording.html')
        self.recording = True
        return EmptyResponse()
        
        # else:
        #     print "back to original webpage"
        #     http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
        #     strands_webserver.client_utils.set_http_root(http_root)
        #     rospy.sleep(1)
        #     strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')

    # def consent_req_callback(self, msg):
    #     print "req cb", msg
    #     self.consent_req=msg
    #     self.get_imgs()
    # strands_webserver.client_utils.set_http_root(self.filepath)
        # rospy.sleep(1)
        # strands_webserver.client_utils.display_relative_page(self.display_no, 'recording.html')
        #self.serve_webpage()

    # def consent_ret_callback(self, msg):
    #     self.consent_ret=msg
    #     print "returned:", msg, self.consent_ret
    #     rospy.sleep(5)
    #     #re-serve the index main webpage
    #     if msg.data != "init":
    #         http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
    #         strands_webserver.client_utils.set_http_root(http_root)
    #         rospy.sleep(1)
    #         strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')


    def serve_webpage(self):
        if not os.path.isfile(self.filepath + '/recording.html'):
            return
        strands_webserver.client_utils.set_http_root(self.filepath)
        rospy.sleep(1)
        strands_webserver.client_utils.display_relative_page(self.display_no, 'recording.html')
        self.manage_timeout()
        self.consent_ret=None

    def create_image(self, query, filename, depth=False):
        msg, meta = self.msg_store.query(Image._type, meta_query=query, single=True)
        if not msg:
            raise Exception('No matching message_store entry')
        if depth:
            #depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            cv2.imwrite(self.filepath + '/images/' + filename +'.jpeg', depth_array*255)

        else:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.filepath + '/images/' + filename +'.jpeg', cv_image)


    def get_imgs(self):
        print 'Creating new images for consent'
        self.create_image({"image_type": "rgb_sk_image"}, 'image1')
        self.create_image({"image_type": "depth_image"}, 'image2', depth=True)
        self.create_image({"image_type": "white_sk_image"}, 'image3')
        return


if __name__ == "__main__":
    """ listen to consent topic
        grab images from mongo-put them on disk
        call serve webserver with a page to obtain consent
    """
    rospy.init_node('consent_for_images')

    print "consent node for storing images..."

    cm = ManageConsentWebpage()
    rospy.spin()
