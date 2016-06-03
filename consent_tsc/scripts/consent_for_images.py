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

class manageConsentWebpage(object):

    def __init__(self, database='message_store', collection='consent_images'):
        self.consent_req = None
        self.consent_ret = None

        self.timeout = 100
        self.filepath = os.path.join(roslib.packages.get_pkg_dir("consent_tsc"), "webserver")
        self.msg_store = MessageStoreProxy(collection=collection, database=database)
        self.bridge = CvBridge()
        self.display_no = rospy.get_param("~display", 0)

        http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
        strands_webserver.client_utils.set_http_root(http_root)
        rospy.sleep(1)
        strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')
        #subscribers
        rospy.Subscriber("/skeleton_data/consent_req", String, callback=self.consent_req_callback, queue_size=1)
        rospy.Subscriber("/skeleton_data/consent_ret", String, callback=self.consent_ret_callback, queue_size=1)
        rospy.Subscriber("/skeleton_data/recording_started", String, callback= self.started_recording_callback, queue_size=1)

    def started_recording_callback(self, msg):
	if msg.data != "finished":
            print "serve recording webpage"
            strands_webserver.client_utils.set_http_root(self.filepath)
            rospy.sleep(1)
            strands_webserver.client_utils.display_relative_page(self.display_no, 'recording.html')
        else:
            print "back to original webpage"
            http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
            strands_webserver.client_utils.set_http_root(http_root)
            rospy.sleep(1)
            strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')

    def consent_req_callback(self, msg):
        print "req cb", msg
        self.consent_req=msg
        self.get_imgs()
	# strands_webserver.client_utils.set_http_root(self.filepath)
        # rospy.sleep(1)
        # strands_webserver.client_utils.display_relative_page(self.display_no, 'recording.html')
        #self.serve_webpage()

    def consent_ret_callback(self, msg):
        self.consent_ret=msg
        print "returned:", msg, self.consent_ret
        rospy.sleep(5)
        #re-serve the index main webpage
        if msg.data != "init":
	    http_root = os.path.join(roslib.packages.get_pkg_dir('tsc_robot_ui'), 'pages')
	    strands_webserver.client_utils.set_http_root(http_root)
	    rospy.sleep(1)
            strands_webserver.client_utils.display_relative_page(self.display_no, 'index.html')


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

    cm = manageConsentWebpage()
    rospy.spin()
