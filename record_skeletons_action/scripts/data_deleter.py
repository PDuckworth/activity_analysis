#!/usr/bin/env python
import roslib
import rospy
import sys, os
import getpass, datetime
import shutil
from std_msgs.msg import String
from mongodb_store.message_store import MessageStoreProxy
import sensor_msgs.msg
from skeleton_tracker.srv import *

def remover_of_images(req):
    time = req.time
    uuid = req.uuid
    consent = req.consent

    date =  str(datetime.datetime.now().date())
    dataset = '/home/' + getpass.getuser() +'/SkeletonDataset/pre_consent/'
    dataset_path = os.path.join(dataset, date)

    dataset_consented_path = os.path.join('/home', getpass.getuser(), 'SkeletonDataset/SafeZone')
    if not os.path.exists(dataset_consented_path):
        os.makedirs(dataset_consented_path)
    print "uuid: %s, consent: %s" % (uuid, consent)

    # find the specific recording to keep (either most images or most recent)
    try:
        for d in os.listdir(dataset_path):
            if uuid in d:
                specific_recording = d

        print "this one:", specific_recording
        location = os.path.join(dataset_path, specific_recording)

        # if consent == "nothing":
        #     shutil.rmtree(os.path.join(location, 'rgb'))
        #     shutil.rmtree(os.path.join(location, 'rgb_skel'))
        #     shutil.rmtree(os.path.join(location, 'depth'))
        #     shutil.rmtree(os.path.join(location, 'skel'))
        #     shutil.rmtree(os.path.join(location, 'robot'))

        if consent == "depthskel":
            print "--remove rgb"
            shutil.rmtree(os.path.join(location, 'rgb'))
            shutil.rmtree(os.path.join(location, 'rgb_sk'))
            # os.remove(os.path.join(location, 'detection.bag'))

        elif consent == "skel":
            print "--remove rgb and depth"
            shutil.rmtree(os.path.join(location, 'rgb'))
            shutil.rmtree(os.path.join(location, 'rgb_sk'))
            shutil.rmtree(os.path.join(location, 'depth'))
            # os.remove(os.path.join(location, 'detection.bag'))

        if "nothing" not in consent:
            print "moving files..."
            new_location = os.path.join(dataset_consented_path, specific_recording)
            os.rename(location, new_location)

    except:
        rospy.logerr("File(s) or directory(ies) can not be found!!!")
    # remove everything in the dataset that is not "Safe"
    rospy.sleep(5.0)
    shutil.rmtree(dataset_path)

    return DeleteImagesResponse(True)


def execute():
    rospy.init_node('skeleton_image_logger', anonymous=True)
                       #service_name      #service_prototype  #handler_function
    s = rospy.Service('/delete_images_service', DeleteImages,  remover_of_images)
    rospy.spin()



if __name__ == '__main__':
    rospy.init_node('skeleton_image_logger', anonymous=True)
    execute()

