#!/usr/bin/env python
import rospy
import actionlib
from std_srvs.srv import Empty, EmptyResponse

from consent_tsc.msg import ManageConsentAction, ConsentResult, ManageConsentGoal



if __name__ == "__main__":
    rospy.init_node('consent_test_client')

    try:
        rospy.wait_for_service('signal_recording_started', timeout=10)
        signal_recording_started = rospy.ServiceProxy('signal_recording_started', Empty)
        # tell the webserver to say we've started recording
        signal_recording_started()

        rospy.sleep(3)

        consent_client = actionlib.SimpleActionClient('manage_consent', ManageConsentAction)
        if consent_client.wait_for_server(timeout=rospy.Duration(10)):
            goal = ManageConsentGoal()
            consent_client.send_goal(goal)

            # here you should check whether you've been preempted, shutdown etc. while waiting for consent
            while True:
                if consent_client.wait_for_result(timeout = rospy.Duration(1)):
                    result = consent_client.get_result()
                    int_result = result.result.result
                    
                    if int_result == ConsentResult.DEPTH_AND_RGB:
                        print 'depth+rgb'
                    elif int_result == ConsentResult.DEPTH_ONLY:
                        print 'just depth'
                    else:
                        print 'no consent'
                    break
        else:
            rospy.logwarn('No manage consent server')
    except Exception, e:
        rospy.logwarn('Exception when trying to manage consent: %s' % e)
