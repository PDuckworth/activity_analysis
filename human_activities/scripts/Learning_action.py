#! /usr/bin/env python

import os, sys
import rospy
import actionlib
import numpy as np
import getpass, datetime
from std_msgs.msg import String, Header
from mongodb_store.message_store import MessageStoreProxy
from learn_activities import Offline_ActivityLearning
from activity_data.msg import HumanActivities
from human_activities.msg import LearningActivitiesAction, LearningActivitiesResult, LastKnownLearningPoint, QSRProgress
from human_activities.msg import FloatList, oLDA

class Learning_server(object):
    def __init__(self, name= "learn_human_activities"):

        # Start server
        rospy.loginfo("%s action server" % name)
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, LearningActivitiesAction,
            execute_cb=self.execute, auto_start=False)
        self._as.start()

        self.recordings = "no_consent"
        # self.recordings = "ecai_Recorded_Data"
        self.path = '/home/' + getpass.getuser() + '/SkeletonDataset/'
        self.ol = Offline_ActivityLearning(self.path, self.recordings)
        self.batch_size= self.ol.config['events']['batch_size']
        self.learn_store = MessageStoreProxy(database='message_store', collection='activity_learning')
        self.msg_store = MessageStoreProxy(database='message_store', collection='activity_learning_stats')
        #self.cpm_flag = rospy.get_param("~use_cpm", False)

        self.last_learn_date = ""
        self.last_learn_success  = bool()

    def cond(self):
        if self._as.is_preempt_requested() or (rospy.Time.now() - self.start).secs > self.duration.secs:
            return True
        return False

    def query_msg_store(self):
        """Queries the database and gets some data to learn on."""
        query = {"cpm":False, "qsrs":False, "activity":False, "temporal":False}
        if self.ol.config["events"]["use_cpm"] == True:
            query["cpm"] = True

        #query = {"cpm":True}
        result = self.learn_store.query(type=HumanActivities._type, message_query=query, limit=self.batch_size)
        uuids = []
        for (ret, meta) in result:
            uuids.append(ret)
        print "query = %s. Ret: %s:%s" % (query, len(result), len(uuids))
        return uuids

    def execute(self, goal):

        print "\nLearning Goal: %s seconds." % goal.duration.secs
        print "batch size max: %s" %  self.ol.config['events']['batch_size']

        self.duration = goal.duration
        self.start = rospy.Time.now()
        self.end = rospy.Time.now()

        self.ol.get_soma_rois()      #get SOMA ROI Info
        self.ol.get_soma_objects()   #get SOMA Objects Info
        #self.get_last_learn_date()
        self.get_last_oLDA_msg()

        learnt_anything = False
        while not self.cond():
            uuids_to_process = self.query_msg_store()
            dates = set([r.date for r in uuids_to_process])

            #to get the folder names vs uuid
            uuids_dict = {}
            for date in dates:
                for file_name in sorted(os.listdir(os.path.join(self.path, self.recordings, date)), reverse=False):
                    k = file_name.split("_")[-1]
                    uuids_dict[k] = file_name

            batch = []
            for ret in uuids_to_process:
                if self.cond(): break

                try:
                    file_name = uuids_dict[ret.uuid]
                except KeyError:
                    print "--no folder for: %s" % ret.uuid
                    continue
                if self.ol.get_events(ret.date, file_name):                #convert skeleton into world frame coords
                    self.ol.encode_qsrs_sequentially(ret.date, file_name)  #encode the observation into QSRs
                    #ret.qsrs = True
                    #self.learn_store.update(message=ret, message_query={"uuid":ret.uuid}, upsert=True)
                    batch.append(file_name)
                    learn_date = ret.date
                self.end = rospy.Time.now()
            if self.cond(): break

            #restrict the learning to only use this batch of uuids
            if len(batch) == 0: break
            self.ol.batch = batch
            # print "batch uuids: ", batch
            print "learning date:", learn_date

            if self.cond(): break
            # self.ol.make_temp_histograms_online(learn_date, self.last_learn_date)  # create histograms with local code book
            new_olda_ret = self.ol.make_temp_histograms_online(learn_date, self.last_olda_ret)  # create histograms with local code book

            if self.cond(): break
            gamma_uuids = self.ol.make_term_doc_matrix(learn_date)  # create histograms with gloabl code book

            if self.cond(): break
            new_olda_ret, gamma = self.ol.online_lda_activities(learn_date, new_olda_ret)  # run the new feature space into oLDA
            #self.update_last_learning(learn_date, True)
            self.update_learned_uuid_topics(uuids_to_process, gamma_uuids, gamma)
            self.last_olda_ret = new_olda_ret
            rospy.loginfo("completed learning loop: %s" % learn_date)
            learnt_anything = True
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

        if learnt_anything:
            #print type(self.last_olda_ret.code_book)
            self.ol.olda_msg_store.insert(message=self.last_olda_ret)
            print "oLDA output pushed to msg store"
        return

    def update_learned_uuid_topics(self, uuids_to_process, uuids, gamma):
        for ret in uuids_to_process:
            #print ret.uuid, ret.topics #uuids.index(ret.uuid)
            try:
                ind = uuids.index(ret.uuid)
                ret.topics = gamma[ind]
                ret.activity = True
            except ValueError:
                pass
                #print "--not learned on"
            ret.qsrs = True
            self.learn_store.update(message=ret, message_query={"uuid":ret.uuid}, upsert=True)

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
        print "dates to process:", self.not_processed_dates

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

    def get_last_learn_date(self):
        """ Find the last time the learning was run - i.e. where the oLDA is to update
        """
        for (ret, meta) in self.msg_store.query(LastKnownLearningPoint._type):
            if ret.type != "oLDA": continue
            if ret.last_date_used > self.last_learn_date:
                self.last_learn_date = ret.last_date_used
                self.last_learn_success = ret.success
        print "Last learned date: ", self.last_learn_date

    def get_last_oLDA_msg(self):
        """ Find the last learning msg on mongo
        """
        self.last_olda_ret = oLDA()
        last_date, last_time = "", ""
        for (ret, meta) in self.ol.olda_msg_store.query(oLDA._type):
            if ret.date > last_date and ret.time > last_time:
                last_date = ret.date
                last_time = ret.time
                self.last_olda_ret = ret
        print "Last learned date: ", last_date
        #return last_date


if __name__ == "__main__":
    rospy.init_node('learning_human_activities_server')

    Learning_server()
    rospy.spin()
