#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:56:22 2019

@author: jakob
"""
import os
import numpy as np

data_path = "/home/jakob/FAUprog/intake_gesture_detection/data"

def list_dir(file_path):
    tmp = []
    names = []
    saves = []
     # !!! no image check, walk dir
    for root, dirs, files in os.walk(file_path):
#        print(dirs)
        for file in files:
            names.append(os.path.splitext(file)[0])
            saves.append(root)
            f = os.path.join(root, file)
            tmp.append(f)
    return tmp, saves, names

def list_tfr(file_path):
    names = []
     # !!! no image check, walk dir
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if check_ext(file, '.tfrecords'):
                names.append(os.path.join(root, file))
    return names

def n_samples(file_path, ext='.mkv'):
    n = 0
     # !!! no image check, walk dir
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if check_ext(file, ext):
                n += 1
    return n

def check_ext(path, ext):
    '''
    Check if path ends with specific extension string, list or tuple
    # Arguments
        path: Path with filename as string
        ext:  Extension name as string, list or tuple
    # Returns
        Boolean true or false
    '''
    extensions = get_ext(path)

    if isinstance(ext, (list, tuple)):
        for name in ext:
            if not name.startswith('.'):
                    name = '.' + name
                    print(name)
            for i in extensions:
                if name == i:
                    return True

    elif isinstance(ext, str):
        if not ext.startswith('.'):
            ext = '.' + ext
        for i in extensions:
                if ext == i:
                    return True
    return False

def get_ext(path):
    '''
    Get the extensions delimited with a dot
    # Arguments
        path: Path with filename as string
    # Returns
        ext_list: List with extensions seperated with a prepended point
    '''
    ext_list = []
    a = True
    while a is True:
        path, ext = os.path.splitext(path)
        if ext is not '':
            ext_list.append(ext)
        else:
            a = False

    return ext_list