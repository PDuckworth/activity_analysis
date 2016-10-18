# activity analysis package

activity analysis package. To manage and represent skeleton and object data about humans in indoor environments.


activity_data
==============

skeleton_publisher.py continuously logs data from the Openni2 Skeleton Tracker package [here](https://github.com/OMARI1988/skeleton_tracker).

It will log in `~/SkeltonDataset/no_consent/`: RGB and Depth images, along with estimated human pose sequence, the robots position, the date, time and a UUID, for each detected human.

Note: There is a flag in main to log anonymous data only (i.e. no RGB data is saved to disc).

```
rosrun activity_data skeleton_publisher.py
```



record_skeleton_action
==============

skeleton_action.py is an action server which records a location (given a goal), to obtain a human detection. Once a detected human has more than a threshold of recorded poses, the action will try to obtain consent in order to store their RGBD data to disc.

It logs an RGB image to mongo, and calls the consent server.

To run:

 ```
rosrun record_skeleton_action skeleton_action.py
rosrun actionlib axclient /record_skeletons
 ```

Requires [shapely](https://pypi.python.org/pypi/Shapely) and [nav_goals_generator](https://github.com/strands-project/strands_navigation/tree/indigo-devel/nav_goals_generator):

```
sudo apt-get install python-shapely
roslaunch nav_goals_generator nav_goals_generator.launch
```

consent_tsc
==============

consent_for_images.py is an action server which deals with obtaining consent from a recorded individual. It serves the latest detected images to the webserver to display and displays yes/no style buttons on screen. It returns the value of this consent.
Required when the webserver and the recording action server are running on different machines.

```
rosrun consent_tsc consent_for_images.py
```

human_activities
==============

Learning_action.py is an action which uses an unsupervised, qualitative framework to learn common motion patterns from the collection of detected humans.

It first obtains all detected human pose sequences from mongo/or from file, and abstracts the pose information usign Qualitative Spatial Representations, as per [QSRLib](https://github.com/strands-project/strands_qsr_lib).

It then performs unsupervised clustering as per recent literature, [here](http://eprints.whiterose.ac.uk/103049/).

To run:

 ```
rosrun human_activities Learning_action.py
rosrun actionlib axclient /LearnHumanActivities
 ```

Configuration File:

`\activity_analysis\human_activities\config\config.ini`


Requires: (LDA package)[https://pypi.python.org/pypi/lda]:

`pip install lda`
