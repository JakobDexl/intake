#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:51:47 2020
Contains all metrics like f1 scores for each class and advanced loss functions.
@author: jakob
"""
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

###############################################################################
#    loss
###############################################################################
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    From: https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed

# from roust paper
def _compute_balanced_sample_weight(labels):
    """
    From https://github.com/prouast/deep-intake-detection/blob/231a4028e076881487194583f839803cea9f14b6/model/oreba_main.py
    Calculate the balanced sample weight for imbalanced data.
    """ 
    f_labels = tf.reshape(labels,[-1]) if labels.get_shape().ndims == 2 else labels
    y, idx, count = tf.unique_with_counts(f_labels)
    total_count = tf.size(f_labels)
    label_count = tf.size(y)
    calc_weight = lambda x: tf.divide(tf.divide(total_count, x),
        tf.cast(label_count, tf.float64))
    class_weights = tf.map_fn(fn=calc_weight, elems=count, dtype=tf.float64)
    sample_weights = tf.gather(class_weights, idx)
    sample_weights = tf.reshape(sample_weights, tf.shape(labels))
    return tf.cast(sample_weights, tf.float32)

def loss(ypred, ytrue, is_training=True):
    # Training with multiple labels per sequence
    if is_training:
        sample_weights = _compute_balanced_sample_weight(tf.math.argmax(ytrue, -1))
    else:
        sample_weights = tf.ones_like(ytrue, dtype=tf.float32)
    
    print(sample_weights)
    # Calculate and scale cross entropy
    scaled_loss = tfa.seq2seq.sequence_loss(
        logits=ypred,
        targets=tf.cast(ypred, tf.int32),
        weights=sample_weights)
    return tf.identity(scaled_loss, name='seq2seq_loss')


###############################################################################
#    metrics
###############################################################################
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision1 = precision(y_true, y_pred)
    recall1 = recall(y_true, y_pred)
    return 2*((precision1*recall1)/(precision1+recall1+K.epsilon()))

def f_none(y_true, y_pred):
    y1 = y_true[:,-1,0]
    y2 = y_pred[:,-1,0]
    return f1(y1, y2)

def f_fetch(y_true, y_pred):
    y1 = y_true[:,-1,1]
    y2 = y_pred[:,-1,1]
    return f1(y1, y2)

def f_eat(y_true, y_pred):
    y1 = y_true[:,-1,2]
    y2 = y_pred[:,-1,2]
    return f1(y1, y2)

def f_drink(y_true, y_pred):
    y1 = y_true[:,-1,3]
    y2 = y_pred[:,-1,3]
    return f1(y1, y2)

def f_ret(y_true, y_pred):
    y1 = y_true[:,-1,-1]
    y2 = y_pred[:,-1,-1]
    return f1(y1, y2)

###############################################################################
#    callbacks
###############################################################################
    
# history callback
class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))