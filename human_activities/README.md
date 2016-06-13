# Relational Learner
A ROS package that uses qualitative spatio-temporal relations to encode RGBD observations.
It then learns a semantic consepts within this representation, analogous to learning human motion patterns, and activity classes in an unsupervised domain.


Prerequisites
-------------

- roscore
- mongodb_store
- skeleton_tracker
- openni2_launch
- soma
- qsrlib


Getting started
-------------------------------
1. Run skeleton_tracker:
    ```
    $ roslaunch skeleton_tracker tracker.launch ...
    ```


2. Run openni2_launch
    ```
    $ roslaunch openni2_launch openni2.launch ...
    ```

3. Run Soma (for visualisation)
    ```
    $ rosrun soma
    ```


4. Run the Human Activities launch, which includes the Learning Action, QSRLib and the online recogniser Action :
    ```
    $ rosrun human_activities human_activities.launch
    ```
