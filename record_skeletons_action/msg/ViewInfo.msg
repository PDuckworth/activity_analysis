string roi  ### roi observing

string roi_config   ### roi config to observe

string map_name  ### map name

string mode 	#### object/human

int32 starttime   ### starttime of search

int32 timestamp 	### timestamp (when the view was executed)

geometry_msgs/Pose robot_pose 	### robot pose

sensor_msgs/JointState ptu_state ### ptu configuration

bool nav_success # True if navigation goto is successful

bool rec_success # True if recorded a person for the threshold duration

bool consent_success # True if that person gave consent_success

string soma_objs ### soma_obj and ID used to observe

geometry_msgs/Pose soma_obj_pose ### soma_obj pose used to observe
