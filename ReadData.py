from random import shuffle
import numpy as np
import glob
import tensorflow as tf
import sys
import os
import struct
import copy


class Data:

    @staticmethod
    def read_train_data():
        train_file_name = './data/train_file.tfrecords'
        writer = tf.python_io.TFRecordWriter(train_file_name)

        f = open('./data/train-images.idx3-ubyte', 'rb')
        f_label = open('./data/train-labels.idx1-ubyte', 'rb')
        magic = f.read(4)
        num = f.read(4)
        num = struct.unpack('!i', num)[0]
        raw = f.read(4)
        raw = struct.unpack('!i', raw)[0]
        col = f.read(4)
        col = struct.unpack('!i', col)[0]

        f_label.read(4)
        f_label.read(4)

        for i in range(num):
            pixs = f.read(raw*col)
            label = f_label.read(1)
            label = struct.unpack('i', label+'\00\00\00')[0]

            feature = {'train/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'train/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pixs]))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())
            if i%100==0:
                print(i)


        writer.close()
        f.close()
        f_label.close()

    def __init__(self, dp):
        self.data_path = dp
        self.sess = tf.Session()
        feature = {
            'train/image': tf.FixedLenFeature([], tf.string),
            'train/label': tf.FixedLenFeature([], tf.int64)
        }
        filename_queue = tf.train.string_input_producer([self.data_path], shuffle=False, num_epochs=30)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.uint8)
        self.label = tf.cast(features['train/label'], tf.int32)
        self.label = tf.one_hot(self.label, 10, 1, 0)
        self.label = tf.reshape(self.label, [1, 1, 10])
        self.label = tf.cast(self.label, tf.float32)
        self.image = tf.reshape(image, [28, 28, 1])
        self.image = tf.to_float(self.image)
        self.image = tf.subtract(tf.div(self.image, 255.), 0.5)
        #images, labels = tf.train.shuffle_batch([image, label], batch_size=1, capacity=30, min_after_dequeue=10)
        self.images, self.labels = tf.train.shuffle_batch([self.image, self.label], batch_size=500, capacity=3 * 5000, min_after_dequeue=10000)
        
        init_op = tf.local_variables_initializer()
        self.sess.run(init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def read_a_record(self):
        return self.sess.run([self.image, self.label])
    
    def read_records(self):

        return self.sess.run([self.images, self.labels])

    def read_record_batch(self):
        return [self.images, self.labels]

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

#d = Data('./data/train_file.tfrecords')
#print d.read_records()[0]
#d.close()



