#!/usr/bin/env python
__author__ = 'paul.duckworth'
import os, sys
import cv2
import numpy as np
from os import listdir, path
from os.path import isfile, join, isdir
import pdb

class ImageCreator():

    def __init__(self, directory):

        self.video_iter = 0
        self.all_videos = [v for v in sorted(listdir(directory))  if isdir(join(directory, v))]
        #pdb.set_trace()
        #self.video = self.all_videos[self.video_iter]

        self.video_len = 0
        self.directory = directory
        #self.dir = join(self.directory, self.video)
        self.skeleton_data = {}
        self.joints = [
            'head',
            'neck',
            'torso',
            'left_shoulder',
            'left_elbow',
            'left_hand',
            'left_hip',
            'left_knee',
            'left_foot',
            'right_shoulder',
            'right_elbow',
            'right_hand',
            'right_hip',
            'right_knee',
            'right_foot']


    def fix_cpm_sk(self):
        """Creates the rgb image with overlaid skeleton tracks.
        Outputs each frame to /rgb_sk folder.
        It uses the rgb images released in the DOI, and so images may be blurred for privicy reasons.
        """
        for self.video_iter, self.video in enumerate(self.all_videos): 
            self.dir = join(self.directory, self.video)
            print self.dir

            self.video_len = len([f for f in listdir(join(self.dir, "cpm_skeleton")) if isfile(join(self.dir, "cpm_skeleton", f))])
            for val in range(1, self.video_len+1):
                if int(val)<10:         val_str = '0000'+str(val)
                elif int(val)<100:      val_str = '000'+str(val)
                elif int(val)<1000:     val_str = '00'+str(val)
                elif int(val)<10000:    val_str = '0'+str(val)
                elif int(val)<100000:   val_str = str(val)
                self.fix_2d_sk(val_str)

    def fix_2d_sk(self,val_str):
        try:
            f1 = open(join(self.dir,'cpm_skeleton','cpm_skl_'+val_str+'.txt'),'r')
            self.skeleton_data = {}
            for count, line in enumerate(f1):
                if count == 0:
                    time = line
                    continue
                line = line.split(',')
                joint_name = line[0]

                self.skeleton_data[joint_name] = {}
                x2d = int(line[1])
                y2d = int(line[2])
                x = float(line[3])
                y = float(line[4])
                z = float(line[5])
                # c = float(line[6])
                self.skeleton_data[joint_name] = [x2d,y2d,x,y,z]
            f1.close()

            f1 = open(join(self.dir,'cpm_skeleton','cpm_skl_'+val_str+'.txt'),'w')
            f1.write(time)
            for joint, [x2d,y2d,x,y,z] in self.skeleton_data.items():
                new_x2d = int(y2d *  (480/float(368)))
                new_y2d = int(x2d *  (480/float(368)))
                new_line = joint+","+str(new_x2d)+","+str(new_y2d)+","+str(x)+","+str(y)+","+str(z)
                f1.write(new_line)
                f1.write('\n')
            f1.close()
        except IOError:
            # print "no cpm file:", val_str
            True


directory = "/home/strands/SkeletonDataset/no_consent"
print "directory = %s" % directory

# date_files = [f for f in listdir(directory)]
#dates = [f for f in ['2016-12-02', '2016-12-05', '2016-12-06', '2016-12-07', '2016-12-08', '2016-12-09', '2016-12-12', '2016-12-13', '2016-12-14', '2016-12-14', '2016-12-16',
#'2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22', '2017-01-03', '2017-01-06', '2017-01-10', '2017-01-11', '2017-01-12', 
dates = [f for f in ['2017-01-13', '2017-01-16'] if isdir(join(directory, f))]

for each_date in sorted(dates):
    d = join(directory, each_date)
    ic = ImageCreator(d)
    ic.fix_cpm_sk()
