string waypoint  ### waypoint id 

string map_name  ### map name

string mode 	#### object/human

int32 starttime   ### starttime of search

int32 timestamp 	### timestamp (when the view was executed)

geometry_msgs/Pose robot_pose 	### robot pose

sensor_msgs/JointState ptu_state ### ptu configuration

bool nav_failure # True if navigation fails

bool success # True if at least one object or person was perceived 

string[] soma_objs ### list of soma_objs seen from the view

