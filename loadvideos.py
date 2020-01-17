#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:07:05 2019
Small cv tool for annotating videos. During reading you can make superevents,
which you can annotate later in more detail.
@author: jakob
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pylab as plt
import math
import os
import file_tools

def vid_to_img(file_path, save_path, size=(128,128)):
    """
    Take superevents in image stream and repeat short sequences around the
    superevent for precise annotation
    Keys:
        q = quit
        SPACE = set superevent
    Params
        file_path = file_path to video
        save_path = path were to save the images, if "" save as npy
        size = out size of the images
    """
    cap = cv2.VideoCapture(file_path)
    org_fps = cap.get(cv2.CAP_PROP_FPS)
    tmp = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Original fps: ", org_fps)
    print("Total number of frames: ", tmp)
    count = 0
    fl = []
    intake = []
    erri = 0

    while(cap.isOpened()):
        frame_id = cap.get(1)
        ret, frame = cap.read()

        if (ret is False):
            print("cv error, frame: ", frame_id)
            if erri > 100:
                break
            erri += 1
        if count == tmp:
            break
#        if (frame_id % math.floor(fps) == 0):
        if frame is not None:
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow('gray', frame)
            frame = frame[:,:,::-1]
            resized = cv2.resize(frame,(224,224),cv2.INTER_AREA)
            fl.append(resized.copy())

            if save_path is not "":
                cv2.imwrite(save_path+str(int(frame_id))+'.jpg', resized[:,:,::-1])
            key = cv2.waitKey(10)
            if key == 32:
                print("added intake", count)
                intake.append(count)
            if key == ord('q'):
                break
            print('\r\r\r', frame_id, ret, end='')
        count += 1

    cap.release()
    return fl, intake

def labels(args):
    pass

def repeat(images, labels, idx):
    """
    Take superevents in image stream and repeat short sequences around the
    superevent for precise annotation
    Keys:
        q = next superevent
        a = add endtime
        s = add starttime
        r = reset labels
        SPACE = pause/restart:
            n = next frame
            b = previous frame
            1 = null
            2 = fetch
            3 = eat
            4 = drnk
            5 = return
    Params
        images = list of images
        labels = list of labels
        idx = list of superevents
    """
    annotations = np.zeros(len(images))
    i=0
    count = 0
    for i in idx:
        start_count = int(i - 50)
        end_count = int(i + 50)
        count = start_count
        anno = 0
        while(1):
            key = cv2.waitKey(50)
            if key == ord('q'):
                break

            if key == ord('a'):
                print("added endtime")
                end_count += 20

            if key == ord('s'):
                print("added starttime")
                start_count -= 20

            if key == ord('r'):
                print("reset to null")
                annotations[start_count:end_count] = 0

            if key == 32:
                while(1):
                    # pause mode

                    k = cv2.waitKey()
                    if k == 32:
                        # resume
                        break

                    if k == ord('n'):
                        # one image further
                        count += 1
                        annotations[count] = anno
                        print("frame ", count, " is " + labels[int(annotations[count])])
                        if count > end_count:
                            count = start_count
                        cv2.imshow('', cv2.resize(images[count],(3*224,3*224),cv2.INTER_AREA))

                    if k == ord('b'):
                        # one image back
                        count -= 1
                        annotations[count] = anno
                        print("frame ", count, " is " + labels[int(annotations[count])])
                        if count < start_count:
                            count = end_count
                        cv2.imshow('', cv2.resize(images[count],(3*224,3*224),cv2.INTER_AREA))

                    # labeling
                    if k == ord('1'):
                        annotations[count] = 0
                        anno = 0
                        print("frame ", count, " is " + labels[0])

                    if k == ord('2'):
                        annotations[count] = 1
                        anno = 1
                        print("frame ", count, " is " + labels[1])

                    if k == ord('3'):
                        annotations[count] = 2
                        anno = 2
                        print("frame ", count, " is " + labels[2])

                    if k == ord('4'):
                        annotations[count] = 3
                        anno = 3
                        print("frame ", count, " is " + labels[3])

                    if k == ord('5'):
                        annotations[count] = 4
                        anno = 4
                        print("frame ", count, " is " + labels[4])

            cv2.imshow('', cv2.resize(images[count],(3*224,3*224),cv2.INTER_AREA))
            print("frame ", count, " is " + labels[int(annotations[count])])
            count +=1
            if count > end_count:
                count = start_count
    cv2.destroyAllWindows()
    return annotations

#class Data():
#    def __init__(self, data, annotations, name):
#        self.data = np.array(data)
#        self.labels = np.array(annotations)
#        self.samples = self.data.shape[0]
#        self.width = self.data.shape[1]
#        self.height = self.data.shape[2]
#        self.channels = self.data.shape[3]
#        self.name = name
#        self.label_names = ['null', 'fetch', 'eat', 'drink', 'return']


if __name__ == '__main__':
#    file = ''
    file = ''
    loads, saves, names = file_tools.list_dir(file)
    count = 0
#    for i in loads:
    save = saves[count]
    fl, intake = vid_to_img(loads[count], "", 1)
    annotations = repeat(fl, ['null', 'fetch', 'eat', 'drink', 'return'], intake)
    np.save(save + 'data.npy', fl)
    np.save(saves[count] + '/anno.npy', annotations)
#    count += 1
