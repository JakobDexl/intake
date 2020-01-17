#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:59:50 2020
Script for evaluating.
@author: jakob
"""

import tensorflow as tf
import numpy as np
#import tensorflow.keras.backend as K
import tf_tools
import pandas as pd
from models import cnn_lstm
# error on the cluster
#import seaborn as sns
#try:
#    import matplotlib.pyplot as plt
#    print("got plt")
#except:
#    print("no visualization")
#
#try:
#    import seaborn as sns
#    print("got sns")
#except:
#    print("no visualization")
#tf.compat.v1.disable_eager_execution()
#tf.config.set_soft_device_placement(True)

# params
seq_len = 16
num_classes = 5
size = 224
batch_size = 8
epochs = 10
classes = ['none', 'fetch', 'eat', 'drink', 'return']

model = cnn_lstm(seq_len, num_classes,size, freeze=0, weights=False)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# load weights
model.load_weights('model2.h5')
#model = tf.keras.models.load_model('model2.h5.h5')

# get data
data = tf_tools.load_val(batch_size, seq_len, '/home/jdexl/files/intake/train/1/')
#    tf.get_variable_scope().reuse_variables()
# 400 steps müssten drin sein bei train und val
#
#model.evaluate(data, steps=50)
#data = tf_tools.load_val(batch_size, seq_len, False)#, '/home/jdexl/files/intake/train/1/')

# loop trough batches and evaluate model. Is done because we need the true 
# labels
for step, (x_batch_train, y_batch_train) in enumerate(data):
    if step == 0:
        logits = model(x_batch_train)[:,-1]
        true = y_batch_train[:,-1]
    
    else:
        pred = model(x_batch_train)[:,-1]
        logits = tf.concat([logits, pred], 0)
        true = tf.concat([true, y_batch_train[:,-1]], 0)

    if step % 20 == 0:
        print(step)
        print(tf.size(true))
    
    if step == 10:
        break

# evaluate and make confusion matrix
x = logits
# get max class
logits2 = tf.where(tf.equal(tf.reduce_max(x, axis=1, keepdims=True), x), 
                   tf.constant(1, shape=x.shape), 
                   tf.constant(0, shape=x.shape))

max_true = tf.reduce_sum(true,0)
max_log = tf.reduce_sum(logits2,0)
print(max_true.numpy())
#print(true.numpy())
print(max_log.numpy())
alogits = tf.math.argmax(logits, -1) 
atrue = tf.math.argmax(true, -1)
#logits = tf.math.argmax(logits, -1)
#true = tf.math.argmax(true, -1)
#print(logits.numpy())
conf = tf.math.confusion_matrix(atrue, alogits, num_classes=5)
con_mat = conf.numpy()
#print(con_mat)
con_mat_norm = np.around(con_mat.astype('float') / (con_mat.sum(axis=1)[:, np.newaxis] + 1e-12), decimals=2)
#print(con_mat_norm)
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
print(con_mat_df)
#try:
#    figure = plt.figure(figsize=(8, 8))
#    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.savefig("confusion.png")
#except:
#    print("no fig")
#   

# save
con_mat_df.to_pickle("confusion.pkl") 
np.save("trueprednum.npy", np.c_[max_true.numpy(), max_log.numpy()])    
np.save("true.npy", true.numpy())    
np.save("pred.npy", pred.numpy())
