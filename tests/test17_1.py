# -*- coding: utf-8 -*-
"""
自己创建和使用TFRecord
使用np.random.randint[0, 256, [h, w, 3]]创建数据

=> img.tostring()和img.tobytes()效果一致
=> h=256时，千条record有180多MB，而h=128时，只有40MB
=> 使用batch时，原始数据尺寸可以不一致，但经过dataset.map(parser)后，得到的数据要尺寸一致
=> 如果parser后得到img, label，那么batch得到的也是[img], [label]
"""

import os, random, sys, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class TFRecord:
    def __init__(self):
        pass

    def write(self, nums=1522):
        max_num = 1000
        writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        path = './record/tfrecord{n}'
        for num in range(nums):
            if num % max_num == 0:
                try:
                    writer.close()
                except:
                    pass
                writer = tf.python_io.TFRecordWriter(path.format(n=num // max_num), options=writer_options)

            example = self.get_example(num % 256)
            writer.write(example.SerializeToString())
        writer.close()

    def get_example(self, label):
        h = np.random.randint(112, 128)
        w = np.random.randint(112, 128)
        img = np.random.randint(0, 256, [h, w, 3], dtype=np.uint8)
        features = tf.train.Features(
            feature={'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                     'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[h, w]))})
        example = tf.train.Example(features=features)
        return example

    def read(self):
        record_dir = './record'
        paths = [os.path.join(record_dir, n) for n in os.listdir(record_dir)]
        dataset = tf.data.TFRecordDataset(paths, compression_type='ZLIB').map(self.parser).batch(10)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def parser(self, record):
        data = tf.parse_single_example(record, features={'img': tf.FixedLenFeature([], dtype=tf.string),
                                                         'label': tf.FixedLenFeature([], dtype=tf.int64),
                                                         'shape': tf.FixedLenFeature([2], dtype=tf.int64), })
        label = tf.cast(data['label'], tf.int32)
        shape = tf.cast(data['shape'], tf.int32)
        img = tf.decode_raw(data['img'], tf.uint8)
        img = tf.reshape(img, [shape[0], shape[1], 3])
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [112, 112])

        return img, label


tf.reset_default_graph()
tfrecord = TFRecord()
tfrecord.write()
next_element = tfrecord.read()
with tf.Session() as sess:
    imgs, labels = sess.run(next_element)
    imgs, labels = sess.run(next_element)

print(imgs.shape)
print(labels)

