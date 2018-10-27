import tensorflow as tf

import utils
from layer import Layer
from libs.activations import lrelu


class Conv2d(Layer):
    # global things...
    layer_index = 0
    
    def __init__(self, kernel_size, strides, output_channels, name):
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_channels = output_channels
        self.name = name

    @staticmethod
    def reverse_global_variables():
        Conv2d.layer_index = 0

    def create_layer(self, input, is_training=True):
        self.input_shape = utils.get_incoming_shape(input)
        number_of_input_channels = self.input_shape[3]

        with tf.variable_scope('conv', reuse=False):
            W = tf.get_variable('W{}'.format(self.name),
                                shape=(self.kernel_size, self.kernel_size, number_of_input_channels, self.output_channels))
            b = tf.Variable(tf.zeros([self.output_channels]))
        self.encoder_matrix = W
        Conv2d.layer_index += 1

        output = tf.nn.conv2d(input, W, strides=self.strides, padding='SAME')

        #output = lrelu(tf.add(tf.contrib.layers.batch_norm(output, scope="norm{}".format(self.name), is_training=is_training), b))
        output = lrelu(tf.add(output, b))
        return output

    def create_layer_reversed(self, input, prev_layer=None, last_layer=False, is_training=True):
        with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W{}'.format(self.name))
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

        output = tf.nn.conv2d_transpose(
            input, W, tf.stack([tf.shape(input)[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
            strides=self.strides, padding='SAME')

        Conv2d.layer_index += 1
        output.set_shape([None, self.input_shape[1], self.input_shape[2], self.input_shape[3]])

        if last_layer:
            #output = tf.add(tf.contrib.layers.batch_norm(output, scope="tnorm{}".format(self.name), is_training=is_training), b, name="output")
            output = tf.add(output, b, name="output")
        else:
            #output = lrelu(tf.add(tf.contrib.layers.batch_norm(output, scope="tnorm{}".format(self.name), is_training=is_training), b))
            output = lrelu(tf.add(output, b))

        return output

    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.strides[1])
