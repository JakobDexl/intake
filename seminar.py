#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:50:27 2019
Main run script for training training the network
@author: jakob
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tf_tools
import lossmetric as lm
from models import cnnlstm

# tf stuff
tf.compat.v1.disable_eager_execution()
tf.config.set_soft_device_placement(True)
#tf.config.threading.set_intra_op_parallelism_threads(12)

# Params
seq_len = 16
num_classes = 5
size = 224
batch_size = 8
epochs = 3

# get model
model = cnnlstm(seq_len, num_classes,size, freeze=0, weights=True)

# compile model
#model.compile(loss=tf.keras.losses.categorical_crossentropy,
#              optimizer=tf.keras.optimizers.Adam(),
#              metrics=['accuracy', lm.f_none, lm.f_fetch, lm.f_eat, lm.f_drink, 
#                       lm.f_ret])
model.compile(loss=[lm.categorical_focal_loss(alpha=.25, gamma=2)],
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy', lm.f_none, lm.f_fetch, lm.f_eat, lm.f_drink, 
                       lm.f_ret])

history = lm.AccuracyHistory()

# if class weighing is activated
class_weight = {0: .5, 1: 2., 2: 2., 3: 2., 4: 2.}

# print model 
#model.summary()

# get the data 
val = tf_tools.load_train(batch_size, seq_len, False)
data = tf_tools.load_val(batch_size, seq_len)

# 400 steps müssten drin sein bei train und val
his = model.fit(data, epochs=epochs, verbose=1, steps_per_epoch=200,
                validation_data=val, validation_steps=100)#, class_weight=class_weight)

# save 
model.save('model4.h5')
np.save("history.npy", his.history)

## remember
#class_weight=class_weight,

#        oh = tf.keras.backend.one_hot(anno, 4)
#    model.save()
# window stack function for stacking frames in width packages
#def window_stack(a, stepsize=1, width=3):
#    n = a.shape[0]
#    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )
