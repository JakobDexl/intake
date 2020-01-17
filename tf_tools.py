#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:58:33 2019
inspired by https://github.com/prouast/deep-intake-detection/
Contains the data pipline with tfrecords
@author: jakob dexl
"""
import os
import tensorflow as tf
import numpy as np
try:
    import cv2
except:
    print("no cv2")
import file_tools

##############
# datasets
##############

def load_test(batch_size, seq_len, data_dir, reject=True):
    
    filenames = file_tools.list_tfr(data_dir)

    dataset = tf.data.TFRecordDataset(filenames)
    # map the parser
    dataset = dataset.map(map_func=input_parser)

    # sample the data densly with a sliding window
    dataset = sliding_records(dataset, seq_len, 1, 1, only_keep_last_label=False)
    if reject:
        dataset = rejection_sampling(dataset, seq_len)
    # set batchsize
    dataset = dataset.batch(batch_size)

    # standardize
    dataset = standardize_batch(dataset)
#    dataset = dataset.map(tf.keras.applications.vgg16.preprocess_input)
    
    # preload
    dataset = dataset.prefetch(1)

    return dataset


def load_train(batch_size, seq_len, shuffle=True):
    data_dir = '/home/jdexl/files/intake/train/'
#    data_dir = '/home/jakob/FAUprog/intake_gesture_detection/data/train/'
    filenames = file_tools.list_tfr(data_dir)
#    files = tf.data.Dataset.list_files(filenames)
#    # Shuffle files if needed
#    if is_training:
#        files = files.shuffle(NUM_SHARDS)

    dataset = tf.data.TFRecordDataset(filenames)
    # handling between files
#    dataset = dataset.interleave(dataset)

    # map the parser
    dataset = dataset.map(map_func=input_parser)

    # sample the data densly with a sliding window
    dataset = sliding_records(dataset, seq_len, 1, 1, only_keep_last_label=False)
    dataset = rejection_sampling(dataset, seq_len, initial=[0.92,0.02,0.02,0.02,0.02])

    # shuffle the dataset and repeat it
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # just return the last label of the swquence
#    dataset = gather_label(dataset, seq_len - 1)

    # set batchsize
    dataset = dataset.batch(batch_size)

    # standardize
    dataset = standardize_batch(dataset)
#    dataset = dataset.map(tf.keras.applications.vgg16.preprocess_input)


    # preload
    dataset = dataset.prefetch(1)

    return dataset


def load_val(batch_size, seq_len, shuffle=True):
#    data_dir = '/home/jakob/FAUprog/intake_gesture_detection/data/val/'
    data_dir = '/home/jdexl/files/intake/val/'
    filenames = file_tools.list_tfr(data_dir)
    dataset = tf.data.TFRecordDataset(filenames)

    # handling between files
#    dataset = files.interleave(dataset, cycle_length=NUM_SHARDS)

    # map the parser
    dataset = dataset.map(map_func=input_parser)
#    dataset = rejection_sampling(dataset, seq_len)

    # sample the data densly with a sliding window
    dataset = sliding_records(dataset, seq_len, 1, 1, only_keep_last_label=False)
    dataset = rejection_sampling(dataset, seq_len, initial=[0.82,0.05,0.03,0.05,0.05])

    # shuffle the dataset and repeat it
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000).repeat()

    # just return the last label of the swquence
#    dataset = gather_label(dataset, seq_len - 1)

    # set batchsize
    dataset = dataset.batch(batch_size)

    # standardize
#    dataset = dataset.map(tf.keras.applications.vgg16.preprocess_input)
    dataset = standardize_batch(dataset)

    # preload
    dataset = dataset.prefetch(1)

#    it = iter(dataset)
#    iter = dataset.make_one_shot_iterator()#initializable_iterator()
#    el = iter.get_next()
#    count = 0
#    with tf.Session() as sess:
#        sess.run(iter.initializer)
#        out = sess.run(el)
##        while(1):
##            try:
##                out = sess.run(el)
##                print('\r\r\r', count, end='')
##                count += 1
##            except:
##                print("stopped")
##                break
    return dataset #out

"""
Tf features
"""


def _int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""
Tf file saving and reading
"""

def save_tfrecords(data, label, vidname, save_dir, off=0, compress=False):
    '''
    Write data to tfrecord file
    # Arguments
        data:
        label:
        vid:
        savedir:
        off:
        compress:
    # Returns
        None
    '''
    filename = os.path.join(save_dir, vidname+'.tfrecords')
    writer = tf.io.TFRecordWriter(filename)

    for i in range(data.shape[0]):

        features = {}
        features['Label'] = _int64(int(label[i]))  #  _bytes(np.array(label[i]).tostring())

        features['Frames'] = _int64(data.shape[0])
        features['Height'] = _int64(data.shape[2])
        features['Width'] = _int64(data.shape[1])
        features['Channels'] = _int64(data.shape[3])
        features['ID'] = _int64(off + i)
        if compress:
            features['Compressed'] = _int64(1)
            features['Image'] = _bytes(cv2.imencode('.jpg', data[i,:,:,::-1])[1].tostring())

        else:
            features['Compressed'] = _int64(0)
            features['Image'] = _bytes(np.array(data[i]).tostring())

#        features['Name'] = _bytes(vidname)

        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    print(i)
    writer.close()


def input_parser(serialized_example):
    """Map serialized example to image data and label."""

    features = {}
    features['Label'] = tf.io.FixedLenFeature([], dtype=tf.int64)  # dtype=tf.string)
    features['Image'] = tf.io.FixedLenFeature([], dtype=tf.string)
    features['Frames'] = tf.io.FixedLenFeature([], dtype=tf.int64)
    features['Height'] = tf.io.FixedLenFeature([], dtype=tf.int64)
    features['Width'] = tf.io.FixedLenFeature([], dtype=tf.int64)
    features['Channels'] = tf.io.FixedLenFeature([], dtype=tf.int64)
    features['ID'] = tf.io.FixedLenFeature([], dtype=tf.int64)
    features['Compressed'] = tf.io.FixedLenFeature([], dtype=tf.int64)
    contents = tf.io.parse_single_example(serialized_example, features)

    w = contents['Width']
    h = contents['Height']
    c = contents['Channels']
    compressed = contents['Compressed']

    image_data = tf.cond(compressed > 0,
                         lambda: tf.io.decode_image(contents['Image'], 3,
                                                    tf.uint8),
                         lambda: tf.io.decode_raw(contents['Image'], tf.uint8))

    image_data = tf.reshape(image_data, [w, h, c])
#    label = tf.decode_raw(contents['Label'], tf.uint8)
#    label = tf.one_hot(tf.decode_raw(contents['Label'], tf.uint8), 5)
    label = tf.one_hot(contents['Label'], 5)
#    label = contents['Label']
    return image_data, label


"""
Tf dataset map functions
"""

def sliding_records(dataset, window_size, window_stride=1, window_shift=1,
                    only_keep_last_label=True):
    '''
    Maps a sliding window function to the tf dataset
    # Arguments
        dataset:
        window_size:
        window_stride:
        window_shift:
    # Returns
        Mapped dataset
    '''

    # TODO which label to keep?
    if only_keep_last_label:
        mapping = lambda i, l: tf.data.Dataset.zip((i.batch(window_size),l))
    else:
        mapping = lambda i, l: tf.data.Dataset.zip((i.batch(window_size),
                                                   l.batch(window_size)))

    dataset = dataset.window(size=window_size, shift=window_shift,
                             stride=window_stride,
                             drop_remainder=True).flat_map(mapping)
    return dataset


def rejection_sampling(dataset, seq_len, target=[0.5, 0.125, 0.125, 0.125, 0.125], initial=None):
    '''
    Rejection sampling adjusts the different label distributions by dropping 
    samples
    # Arguments
       
    # Returns
        Mapped dataset
    '''
    # return just the last label 
    def class_func(features, label):
        return tf.math.argmax(label[-1])

    resampler = tf.data.experimental.rejection_resample(class_func,
                                                        target_dist=target,
                                                        initial_dist=initial)
    dataset = dataset.apply(resampler)

    dataset = dataset.map(lambda extra_label,
                          features_and_label: features_and_label)

    return dataset


def gather_label(dataset, idx):
    mapping = lambda i, l: (i,tf.gather(l, idx))
    dataset = dataset.map(map_func=mapping)
    return dataset


def standardize_batch(dataset):
    mapping = lambda b, l: (_standardization(tf.cast(b, tf.float32)), l)
    dataset = dataset.map(map_func=mapping)
    return dataset


def _standardization(images):
    """
    Linearly scales image data to have zero mean and unit variance.
    Taken from Raoust Paper implementation
    """
    num_pixels = tf.reduce_prod(tf.shape(images))
    images_mean = tf.reduce_mean(images)
    variance = tf.reduce_mean(tf.square(images)) - tf.square(images_mean)
    variance = tf.nn.relu(variance)
    stddev = tf.sqrt(variance)
    # Apply a minimum normalization that protects us against uniform images.
    min_stddev = tf.math.rsqrt(tf.cast(num_pixels, dtype=tf.float32))
    pixel_value_scale = tf.maximum(stddev, min_stddev)
    pixel_value_offset = images_mean
    images = tf.subtract(images, pixel_value_offset)
    images = tf.divide(images, pixel_value_scale)
    return images


def flip(dataset):
    pass

def _color(x):
    """Color augmentation

    Arguments
        dataset: Image

    Returns
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def color(dataset):
    return dataset.map(_color, num_parallel_calls=4)

#def zoom(dataset):
#    pass
#def crop(dataset):
#    pass


###################
#    Experimental
###################
"""
Tf dataset helper
"""


def get_n(dataset):
    # not recommended, takes a lot of time
    it = iter(dataset.repeat(0))
    count = 0
    while 1:
        try:
            ne = next(it)
            count += 1
        except:
            exit
    print(count)
