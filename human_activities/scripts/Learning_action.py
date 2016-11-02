#! /usr/bin/env python

import os, sys
import rospy
import actionlib
import getpass, datetime
from std_msgs.msg import String, Header
from mongodb_store.message_store import MessageStoreProxy
from learn_activities import Offline_ActivityLearning
from human_activities.msg import LearningActivitiesAction, LearningActivitiesResult, LastKnownLearningPoint, QSRProgress

class Learning_server(object):
    def __init__(self, name= "LearnHumanActivities"):

        # Start server
        rospy.loginfo("Learning Human Activites action server")
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, LearningActivitiesAction,
            execute_cb=self.execute, auto_start=False)
        self._as.start()

        self.recordings = "no_consent"
        # self.recordings = "ecai_Recorded_Data"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
        self.ol = Offline_ActivityLearning(self.path, self.recordings)

        self.ol.soma_map = rospy.get_param("~soma_map", "collect_data_map_cleaned")
        self.ol.soma_config = rospy.get_param("~soma_config", "test")
        self.ol.roi_config = rospy.get_param("~roi_config", "test")
        self.ol.soma_roi_store = MessageStoreProxy(database='soma2data', collection='soma2_roi')
        self.msg_store = MessageStoreProxy(database='message_store', collection='activity_learning_stats')

        self.qsr_progress_date = ""
        self.qsr_progress_uuid = ""
        self.last_learn_date = ""
        self.last_learn_success  = bool()

    def cond(self):
        if self._as.is_preempt_requested() or (rospy.Time.now() - self.start).secs > self.duration.secs:
            return True
        return False

    def execute(self, goal):
        print "\nLearning Goal: %s seconds." % goal.duration.secs
        self.duration = goal.duration
        self.start = rospy.Time.now()
        self.end = rospy.Time.now()

        while (self.end - self.start).secs < self.duration.secs:
            self.get_dates_to_process()
            self.get_dates_to_learn()

            self.ol.get_soma_rois()      #get SOMA ROI Info
            self.ol.get_soma_objects()   #get SOMA Objects Info

            for date in self.not_processed_dates:
                if self.cond(): break
                print "\nprocessing date: %s " % date

                for uuid in self.get_uuids_to_process(date):
                    if self.cond(): break

                    #remove any uuids you've already processed on this date
                    if date == self.qsr_progress_date:
                        if uuid <= self.qsr_progress_uuid: continue
                            # print "already processed"

                    self.ol.get_events(date, uuid)      #convert skeleton into world frame coords
                    self.ol.encode_qsrs_sequentially(date, uuid) #encode the observation into QSRs
                    self.update_qsr_progress(date, uuid)

                if date == self.last_learn_date:
                    if self.last_learn_success: continue

                if not self.cond():
                    self.ol.make_temp_histograms_online(date, self.last_learn_date)  # create histograms with local code book

                if not self.cond():
                    self.ol.make_term_doc_matrix(date)  # create histograms with gloabl code book

                if not self.cond():
                    self.ol.online_lda_activities(date, self.last_learn_date)  # run the new feature space into oLDA
                    self.update_last_learning(date, True)
                    rospy.loginfo("completed learning for %s" % date)

            self.end = rospy.Time.now()

            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted(LearningActivitiesResult())

            elif (rospy.Time.now() - self.start).secs > self.duration.secs:
                rospy.loginfo('%s: Timed out' % self._action_name)
                self._as.set_preempted(LearningActivitiesResult())

            else:
                rospy.loginfo('%s: Completed' % self._action_name)
                self._as.set_succeeded(LearningActivitiesResult())
            return
        return

    def get_uuids_to_process(self, folder):
        return [uuid for uuid in sorted(os.listdir(os.path.join(self.path, self.recordings, folder)), reverse=False)]

    def get_dates_to_process(self):
        """ Find the sequence of date folders (on disc) which have not been processed into QSRs.
        ret: self.not_processed_dates - List of date folders to use
        """
        for (ret, meta) in self.msg_store.query(QSRProgress._type):
            if ret.type != "QSRProgress": continue
            if ret.last_date_used >= self.qsr_progress_date:
                self.qsr_progress_date = ret.last_date_used
                self.qsr_progress_uuid = ret.uuid

        print "qsr progress date: ", self.qsr_progress_date, self.qsr_progress_uuid
        self.not_processed_dates = []
        for date in sorted(os.listdir(os.path.join(self.path, self.recordings)), reverse=False):
            if date >= self.qsr_progress_date:
                self.not_processed_dates.append(date)
        print "not processed dates:", self.not_processed_dates

    def update_qsr_progress(self, date, uuid):
        query={"type":"QSRProgress"}
        date_ran = str(datetime.datetime.now().date())
        msg = QSRProgress(type="QSRProgress", date_ran=date_ran, last_date_used=date, uuid=uuid)
        self.msg_store.update(message=msg, message_query=query, upsert=True)


    def update_last_learning(self, date, success):
        #%todo: add success rate?
        self.last_learn_date = date
        date_ran = str(datetime.datetime.now().date())
        msg = LastKnownLearningPoint(type="oLDA", date_ran=date_ran, last_date_used=self.last_learn_date, success=success)
        query= {"type":"oLDA", "date_ran":date_ran}
        self.msg_store.update(message=msg, message_query=query, upsert=True)

    def get_dates_to_learn(self):
        """ Find the sequence of date folders which need to be learned.
        """
        for (ret, meta) in self.msg_store.query(LastKnownLearningPoint._type):
            if ret.type != "oLDA": continue
            if ret.last_date_used > self.last_learn_date:
                self.last_learn_date = ret.last_date_used
                self.last_learn_success = ret.success
        print "Last learned date: ", self.last_learn_date

if __name__ == "__main__":
    rospy.init_node('learning_human_activities_server')

    Learning_server(name = "LearnHumanActivities")
    rospy.spin()
