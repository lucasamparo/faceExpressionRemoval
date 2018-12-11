import os, time
import numpy as np
import tensorflow as tf
from network_model import Network
from dataset import Dataset

class Process:
  def __init__(self, model_ckpt_path):
    network = Network()
    self.model_path = model_ckpt_path

	def inference(self, batch):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			saver = tf.train.Saver()
			saver.restore(sess, self.model_path)

			result, test_accuracy, mse = sess.run([network.final_result], feed_dict={network.inputs: batch, network.is_training: False})
														  
      return result
