#!/usr/bin/env python

import sys
import rospy
import numpy as np
import cPickle as pickle

from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose
from activity_data.msg import *
from skeleton_tracker.msg import *

from qsrlib_io.world_trace import Object_State, World_Trace
from qsrlib_ros.qsrlib_ros_client import QSRlib_ROS_Client
from qsrlib.qsrlib import QSRlib, QSRlib_Request_Message
import qsrlib_qstag.utils as utils
import qsrlib_utils.median_filter as filters
from mongodb_store.message_store import MessageStoreProxy

class dataReader(object):

	def __init__(self):

		self.skeleton_data = {}  #keeps the published complete skeleton in a dictionary. key = uuid
		self.rate = rospy.Rate(15.0)

		## listeners:
		rospy.Subscriber("skeleton_data/complete", skeleton_complete, self.skeleton_callback)

		## Logging params:
		self._with_mongodb = rospy.get_param("~with_mongodb", "true")
		self._log_world = rospy.get_param("~log_world", "false")
		self._log_qsrs = rospy.get_param("~log_qsrs", "false")
		self._message_store = rospy.get_param("~message_store_prefix", "people_skeleton")

		if self._log_world:
			msg_store = self._message_store + "_world_state"
			rospy.loginfo("Connecting to mongodb...%s" % msg_store)
			self._store_client_world = MessageStoreProxy(collection=msg_store)


		## QSR INFO:
		self._which_qsr = rospy.get_param("~which_qsr", "qtcbs")
		#self.which_qsr = ["qtcbs", "argd"]

		quantisation_factor = 0.01
		validate = False
		no_collapse = True
		argd_params = {"Touch": 0.5, "Near": 2, "Far": 3}
		qstag_params = {"min_rows" : 1, "max_rows" : 1, "max_eps" : 5}

		self.dynamic_args = {
						"qtcbs": {"quantisation_factor": quantisation_factor,
								  "validate": validate,
								  "no_collapse": no_collapse},
						"argd": {"qsr_relations_and_values" : argd_params},
						"qstag": {"params" : qstag_params},
						#"filters": {"median_filter" : {"window": 2}}
						}

		self.cln = QSRlib_ROS_Client()


	def _create_qsrs(self):
		while not rospy.is_shutdown():

			for uuid, msg_data in self.skeleton_data.items():
				if msg_data["flag"] != 1: continue
				print ">> recieving worlds for:", uuid
				qsrlib_response_message = self._call_qsrLib(uuid, msg_data)
				print "??", qsrlib_response_message

				del self.skeleton_data[uuid]

			self.rate.sleep()



	def skeleton_callback(self, msg):
		self.skeleton_data[msg.uuid] = {
					"flag": 1,
					"start_time": msg.start_time,
					"end_time": msg.end_time,
					"world": self.convert_skeleton_to_world(msg.skeleton_data)}


	def convert_skeleton_to_world(self, data, use_every=1):
		world = World_Trace()

		joint_states = {'head' : [], 'neck' : [],  'torso' : [], 'left_shoulder' : [],'left_elbow' : [],
				'left_hand' : [],'left_hip' : [], 'left_knee' : [],'left_foot' : [],'right_shoulder' : [],
				'right_elbow' : [],'right_hand' : [],'right_hip' : [],'right_knee' : [],'right_foot' : []}

		for tp, msg in enumerate(data):
			for i in msg.joints:
				o = Object_State(name=i.name, timestamp=tp, x=i.pose.position.x,\
				                y = i.pose.position.y, z=i.pose.position.z)
				joint_states[i.name].append(o)

		for joint_data in joint_states.values():
			world.add_object_state_series(joint_data)
		return world


	def _call_qsrLib(self, uuid, msg_data):

		qsrlib_request_message = QSRlib_Request_Message(which_qsr=self._which_qsr, input_data=msg_data["world"], \
		                        dynamic_args=self.dynamic_args)
		req = self.cln.make_ros_request_message(qsrlib_request_message)

		if self._log_world:
			print "logging the world?"

			msg = QSRlibMongo(uuid = uuid, data = pickle.dumps(msg_data["world"]), \
						start_time = msg_data["start_time"],  end_time = msg_data["end_time"]   )

			print "msg worked", type(msg)
			query = {"uuid" : uuid}
			self._store_client_world.update(message=msg, message_query=query, upsert=True)

		res = self.cln.request_qsrs(req)
		qsrlib_response_message = pickle.loads(res.data)
		return qsrlib_response_message



class objectData(object):
	def __init__(self):
		self.objects = []
		self.classified_types = []
		rospy.Subscriber("/objects", object_message_type, self.callback)

	def callback(self, msg):
		if len(msg.objects) > 0:
			self.objects = msg.objects
			self.classified_types = msg.classified_types


class Importer(object):
	def __init__(self):
		meta = {}
		self.msg = skeleton_msg(header=h, uuid=uuid, skeleton=skeleton)

		self._client = pymongo.MongoClient(rospy.get_param("mongodb_host"), rospy.get_param("mongodb_port"))



		p_id = i._store_client.update(message=msg, meta=meta, upsert=True)


if __name__ == "__main__":
	rospy.init_node('activity_data')

	print "getting data for qualitative activity analysis..."

	dr = dataReader()
	dr._create_qsrs()

