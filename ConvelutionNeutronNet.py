import tensorflow as tf
import copy
import numpy as np


class CNNNetLayer:

    def __init__(self, input_node_size, output_node_size):
        self.filter = None
        self.strides = None
        self.padding = 'SAME'
        self.pooling_ksize = [1, 2, 2, 1]
        self.pooling_strides = [1, 2, 2, 1]
        self.pooling_padding = 'VALID'
        #self.bias = tf.Variable(tf.zeros[output_node_size])
        self.input_node_size = input_node_size
        self.output_node_size = output_node_size
        self.input_data = None
        self.output_data = None

    def if_load_data(self):
        if self.input_data is None:
            return False
        return True

    def load_data(self, input_data):
        self.input_data = input_data

    def set_filter(self, filter_size):
        self.filter = tf.Variable(tf.random_normal(filter_size))

    def set_strides(self, strides):
        self.strides = strides

    def set_padding(self, padding):
        self.padding = padding

    def pooling(self):
        self.output_data = tf.nn.max_pool(self.output_data, self.pooling_ksize, self.pooling_strides, self.pooling_padding)

    def convolution_calculate(self):
        self.output_data = tf.nn.relu(tf.nn.conv2d(self.input_data, self.filter, self.strides, self.padding))


class NetLayer:

    def __init__(self, input_node_num, output_node_num):
        self.W = tf.Variable(tf.truncated_normal[input_node_num, output_node_num])
        self.bias = tf.Variable(tf.zeros[output_node_num])
        self.input_node_num = input_node_num
        self.output_node_num = output_node_num
        self.input_data = None
        self.output_data = None

    def if_load_data(self):
        if self.input_data is None:
            return False
        return True

    def load_data(self, input_data):
        self.input_data = input_data

    def calculate(self):
        self.output_data = tf.nn.relu(tf.matmul(self.input_data, self.W) + self.bias)


class CNN:

    def __init__(self):
        self.input_data = None
        self.output_data = None
        self.true_output_data = None
        self.net_layer_set = []

    def calculate(self, data=None):
        if not (data is None):
            input_data = data
        else:
            input_data = self.input_data
        output_data = None
        for net_layer in self.net_layer_set:
            net_layer.load_data(input_data)
            output_data = net_layer.calculate()
            input_data = output_data
        self.output_data = output_data
        return output_data

    def add_net_layer(self, net_layer):
        self.net_layer_set.append(copy.deepcopy(net_layer))

    def load_data(self, data):
        self.input_data = data

    def load_data_copy(self, data):
        self.input_data = copy.deepcopy(data)

    def get_output(self):
        return self.output_data

    def get_output_copy(self):
        return copy.deepcopy(self.output_data)

    def load_true_output_data(self, data):
        self.true_output_data = data

    def get_error_benchmark(self):
        return tf.reduce_mean(tf.square(tf.sub(self.true_output_data, self.output_data)))



