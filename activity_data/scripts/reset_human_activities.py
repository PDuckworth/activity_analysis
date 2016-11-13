#! /usr/bin/env python
import os, sys
import rospy
from mongodb_store.message_store import MessageStoreProxy
from activity_data.msg import HumanActivities

if __name__ == "__main__":
    rospy.init_node('insert_msgs_to_mongo')
    msg_store = MessageStoreProxy(database='message_store', collection='activity_learning')

    for (ret, meta) in msg_store.query(type = HumanActivities._type):
        print ret.uuid
        ret.qsrs = False
        ret.activity = False
        ret.topics = []
        ret.hier_topics = []
        ret.cpm = False
        msg_store.update(message_query={"uuid":ret.uuid}, message=ret)
