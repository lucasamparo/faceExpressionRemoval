import math
import os
import time
from math import ceil

import cv2
import sys
import matplotlib

from scipy import signal
from scipy import ndimage

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from PIL import Image

from conv2d import Conv2d
from max_pool_2d import MaxPool2d
from recog_model import inception_resnet_v1 as model
import load_recog as recog

np.set_printoptions(threshold=np.nan)

class Network:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, skip_connections=False):
        with tf.variable_scope('encoder-decoder'):
            if layers == None:
                    layers = []
                    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
                    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
                    layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections))

                    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
                    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
                    layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))

                    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
                    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))            
                    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_3'))

            self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS], name='inputs')
            self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
            self.is_training = tf.placeholder_with_default(False, [], name='is_training')
            self.description = ""

            self.layers = {}

            net = self.inputs

            # ENCODER
            for layer in layers:
                    self.layers[layer.name] = net = layer.create_layer(net)
                    self.description += "{}".format(layer.get_description())

            layers.reverse()
            Conv2d.reverse_global_variables()

            # DECODER
            layers_len = len(layers)
            for i, layer in enumerate(layers):
                    if i == (layers_len-1):
                            self.segmentation_result = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name], last_layer=True)
                    else:
                            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
    
        self.variables = tf.contrib.framework.get_variables(scope='encoder-decoder')
       
        self.final_result = self.segmentation_result

        self.rec1 = Recognizer(self.final_result, reuse=tf.AUTO_REUSE)
        self.rec2 = Recognizer(self.inputs, reuse=tf.AUTO_REUSE)
        self.rec3 = Recognizer(self.targets, reuse=tf.AUTO_REUSE)

        # MSE loss
        # Expression Removal with MSE loss function
        output = self.segmentation_result
        inputv = self.targets
        mean = tf.reduce_mean(tf.square(output - inputv))
        # Recognition feature sets
        rec1_loss = self.rec1.modelo
        rec2_loss = self.rec2.modelo
        rec3_loss = self.rec3.modelo
        output_weight = tf.constant(3, shape=[], dtype=tf.float32)
        cost_rec1 = tf.multiply(output_weight, tf.reduce_sum(tf.reduce_mean(tf.abs(rec1_loss - rec3_loss), 0)))
        cost_rec2 = tf.reduce_sum(tf.reduce_mean(tf.abs(rec1_loss - rec2_loss), 0))
		
        self.cost_rec = cost_rec1 + cost_rec2
        self.cost_mse = mean
        
		self.train_op_rec = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(0.00001, 1, 10000, 0.0000001)).minimize(self.cost_rec)
        self.train_op_mse = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(0.0001, 1, 10000, 0.00001)).minimize(self.cost_mse)

    def loadRecog(self, sess):
        self.rec1.load(sess)
        self.rec2.load(sess)
        self.rec3.load(sess)
