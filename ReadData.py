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
                print i


        writer.close()
        f.close()
        f_label.close()


with tf.Session() as sess:
    data_path = './data/train_file.tfrecords'
    feature = {
        'train/image': tf.FixedLenFeature([], tf.string),
        'train/label': tf.FixedLenFeature([], tf.int64)
    }
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    print 'begin read'
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    print 'decode'
    image = tf.decode_raw(features['train/image'], tf.uint8)

    label = tf.cast(features['train/label'], tf.int32)

    image = tf.reshape(image, [28, 28])

    images, labels = tf.train.shuffle_batch([image, label], batch_size=1, capacity=30, min_after_dequeue=10)
    print 'init'
    init_op = tf.initialize_all_variables()
    sess.run(init_op)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print 'run images'
    img = sess.run(images)
    print img
    coord.request_stop()
    coord.join(threads)
    sess.close()
