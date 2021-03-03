"""
二进制转图像
"""
from io import BytesIO
import os, sys, math, random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 生成feature
def bytes_feature(feature, is_list=False):
    if not is_list: feature = [feature]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=feature))


def int64_feature(feature, is_list=False):
    if not is_list: feature = [feature]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=feature))


tf.reset_default_graph()


class TFRecordMaker:
    def __init__(self, max_num_per_record=1000):
        self._max_num = max_num_per_record

    def writeTFRecord(self, path, record_path):
        print(f'START: write TFRecord.')

        writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

        writer = tf.python_io.TFRecordWriter(record_path, writer_options)

        img = self.get_img(path)
        features = tf.train.Features(feature={'img': bytes_feature(img)})
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

        print(f'FINISH: write TFRecord.')
        print('-' * 100)

    # 获取
    def get_img(self, path):
        with tf.gfile.GFile(path, 'rb') as gf:
            return gf.read()

path = r'E:\TEST\ChangeFace\recognition/zz.jpg'
record_path = f'./zz.tfrecord'
tfrmaker = TFRecordMaker()
tfrmaker.writeTFRecord(path, record_path)

tf.reset_default_graph()


class DatasetMaker:
    def __init__(self):
        pass

    def read(self, path):
        dataset = tf.data.TFRecordDataset(path, compression_type="ZLIB")
        # # 当硬件不足时，buffer_size帮助确定每次读入多少数据
        dataset = dataset.map(self.parser)
        # # num_parallel_calls确定同时多少数据进入parser

        return dataset

    def parser(self, record):
        features = tf.parse_single_example(record,
                                           features={'img': tf.FixedLenFeature([], dtype=tf.string)})
        #         img = tf.decode_raw(features['img'], tf.uint8)
        img = tf.image.decode_image(features['img'], channels=3)
        # tf.image.decode_png()

        return img

dm = DatasetMaker()
record_path = f'./zz.tfrecord'
dataset = dm.read(record_path).repeat()
img = dataset.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    img = sess.run(img)
plt.imshow(img)
plt.show()





