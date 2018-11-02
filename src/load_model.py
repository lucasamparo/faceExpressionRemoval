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
from PIL import Image

from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io
import utils
import gc
import tensorflow.contrib.slim as slim
from recog_model import inception_resnet_v1 as model
import load_recog as recog

from network import Network
from dataset import Dataset

np.set_printoptions(threshold=np.nan)

def saveImage(image, height, width, path):
    fig = plt.figure(frameon=False, dpi=100)
    fig.set_size_inches(height/100, width/100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap="gray")
    fig.savefig(path)
    plt.close(fig)
    del fig
    gc.collect()

def create_dirs(model_path):
    if(not os.path.exists(model_path+"/inputs")):
        os.makedirs(os.path.join(model_path, "inputs/valid"))
        os.makedirs(os.path.join(model_path, "inputs/test"))
        os.makedirs(os.path.join(model_path, "inputs/train"))

    if(not os.path.exists(model_path+"/targets")):
        os.makedirs(os.path.join(model_path, "targets/valid"))
        os.makedirs(os.path.join(model_path, "targets/test"))
        os.makedirs(os.path.join(model_path, "targets/train"))

    print("All directories created")

def load(model_path, output_path, image_path):
    network = Network(has_translator=has_translator)
    saver = tf.train.Saver()
    
    dataset = Dataset(folder=image_path, batch_size=128)
    
    config = tf.ConfigProto(
            device_count = {'GPU' : 0}
    )
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "{}/checkpoint.data".format(model_path))
        print("General Model Restored")
        create_dirs(os.path.join("result/export", model_path))
        
        dataset.reset_batch_pointer()
        count = 0

        for batch_i in range(dataset.num_batches_in_epoch()):
            batch_inputs, paths = dataset.next_batch()
            batch_inputs = np.reshape(batch_inputs,(dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
            batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

            start = time.time()
            image = sess.run([network.segmentation_result], feed_dict={network.inputs: batch_inputs,network.is_training: True})
            end = time.time()
            total_time = round(end-start,5)
            image = np.array(image[0])
            for j in range(image.shape[0]):
                save_image = np.squeeze(image[j])
                split_path = paths[j].split("/")
                path = "{}/{}/{}/{}".format(output_path, model_path,os.path.join(split_path[1],split_path[2]), split_path[3])
                saveImage(save_image, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, path)
                print("Saving {} em {}s ({} de {})".format(split_path[1]+"/"+split_path[2]+"/"+split_path[3], total_time, count, len(dataset.train_inputs)))
                count += 1
        print("All images exported")
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
        type=str,
        required=True,
        help='Path to the trained model')

    parser.add_argument('--output_path',
        type=str,
        required=True,
        help='Path to save the images')

    parser.add_argument('--image_path',
        type=str,
        required=True,
        help='Path to input images')

    FLAGS = parser.parse_args()
    
    load(FLAGS.model_path, FLAGS.output_path, FLAGS.image_path)
