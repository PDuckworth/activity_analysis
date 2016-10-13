#! /usr/bin/env python
import rospy
import roslib
import sys, os
from std_msgs.msg import String, Header, Int32
from mongodb_store.message_store import MessageStoreProxy
# from soma2_msgs.msg import SOMA2Object



if __name__ == "__main__":
    rospy.init_node('query_soma')

    # connecting
    soma_store = MessageStoreProxy(database="message_store", collection="soma_activity_ids_list")

    if len(sys.argv) > 1:
        for cnt, arg in enumerate(sys.argv):
            if cnt ==0: continue
            # print arg

            print 'Add an object ID to a msg store list: %s ' %arg
            new_obj_id = Int32(int(arg))

            # putting something in
            soma_store.insert_named("object id %s" %arg, new_obj_id)

            # # getting it back out
            # id,meta = soma_store.query_named("object id %s" %arg, Int32._type)
            # print scene, meta

    else:
        print "Requires a list of SOMA2 object IDs to add to a database"
