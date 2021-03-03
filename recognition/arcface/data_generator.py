# -*- coding: utf-8 -*-
import os, sys, random, time, math
# import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from recognition.arcface.configs import config


# 获取数据集文件夹的path_labels
def get_pl_from_dir(root_dir, start_label=0):
    # 获取类别文件夹
    cls_dirs = [os.path.join(root_dir, _clsname) for _clsname in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, _clsname))]
    next_label = start_label
    totals, trains, tests = [], [], []
    test_ratio = 0.25
    for cls_dir in cls_dirs:
        paths = get_all_files(cls_dir)  # 获取文件夹的所有子文件
        random.shuffle(paths)
        test_num = int(test_ratio * len(paths))
        tests.extend([[p, next_label] for p in paths[:test_num]])  # 测试
        trains.extend([[p, next_label] for p in paths[test_num:]])  # 训练
        next_label += 1
    train_num, test_num = len(trains), len(tests)

    print(f'label num: {next_label - start_label} - train num: {train_num} - test num: {test_num}')

    return trains, tests, next_label


# 从多个根目录中获取path_label
def get_pl_from_dirs(root_dirs):
    next_label = 0
    trains, tests = [], []
    for root_dir in root_dirs:
        _trains, _tests, next_label = get_pl_from_dir(root_dir, next_label)
        trains.extend(_trains)
        tests.extend(_tests)
    print(f'TOTAL: label num: {next_label} - train num: {len(trains)} - test num: {len(tests)}')
    return trains, tests


# 获取文件夹下所有图片子文件
def get_all_files(folder, format=('bmp', 'jpg', 'png', 'tif', 'jpeg')):
    paths = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and p.split('.')[-1] in format:
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(get_all_files(p))
    return paths


# 生成feature
def bytes_feature(feature, is_list=False):
    if not is_list: feature = [feature]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=feature))


def int64_feature(feature, is_list=False):
    if not is_list: feature = [feature]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=feature))


class TFRecordMaker:
    def __init__(self, max_num_per_record=1000):
        self._max_num = max_num_per_record

    def writeTFRecord_from_dirs(self, root_dirs, name):
        """
        - root_dir 1
            - cls 1
                - pics 1
                - pics 2
            - cls 2
                ...
            ...
        :param root_dirs: [root_dir1, root_dir12, root_dir13, ...]
        :param name: 数据集名称
        :param max_num_per_record: 单个record最多条数
        :return:
        """
        print(f'START: write TFRecord from {root_dirs}.')

        # 给出文件夹列表中的所有样本path, label
        train, test = get_pl_from_dirs(root_dirs)

        record_path = f'../datasets/{name}_' + '{cate}{n}.tfrecord'
        writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        for cate in ('train', 'test'):
            with open(f'../datasets/{name}_{cate}.txt', 'w') as f:
                f.writelines([f'{p}, {l}\n' for p, l in eval(f'{cate}')])
            print(f'FINISH: write txt into "../datasets/{name}_{cate}.txt".')
            writer = tf.python_io.TFRecordWriter(record_path.format(cate=cate, n=0), writer_options)
            for num, [path, label] in enumerate(eval(f'{cate}')):
                if num % self._max_num == 0 and num > 0:
                    writer.close()
                    writer = tf.python_io.TFRecordWriter(record_path.format(cate=cate, n=num // self._max_num),
                                                         writer_options)
                    print()
                img = self.get_img(path)
                features = tf.train.Features(feature={'img': bytes_feature(img),
                                                      'label': int64_feature(label)})
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                print(f'\rwrite {cate}: {num // self._max_num} - {num % self._max_num}...', end='')
            writer.close()
            print()
        print(f'FINISH: write TFRecord from {root_dirs}.')
        print('-' * 100)

    # 获取
    def get_img(self, path):
        with tf.gfile.GFile(path, 'rb') as gf:
            return gf.read()


class DatasetMaker:
    def __init__(self, output_size=config.img_size):
        self.output_size = output_size

    def read(self, dataset_name, cate='test', aug=False):
        """
        tfrecord存放位置：../datasets/xxx
        :param dataset_name:
        :param cate:
        :return:
        """
        filenames = self.get_filenames(dataset_name, cate)

        dataset = tf.data.TFRecordDataset(filenames, compression_type="ZLIB", buffer_size=256 << 20)
        # # 当硬件不足时，buffer_size帮助确定每次读入多少数据

        dataset = dataset.map(lambda x: self.parser(x, aug), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # # num_parallel_calls确定同时多少数据进入parser

        return dataset

    def parser(self, record, aug=True):
        features = tf.parse_single_example(record,
                                           features={'img': tf.FixedLenFeature([], dtype=tf.string),
                                                     'label': tf.FixedLenFeature([], dtype=tf.int64)})
        # img = tf.decode_raw(features['img'], tf.uint8)
        # img = tf.reshape(img, features['shape'])
        img = tf.image.decode_image(features['img'], channels=3)
        img.set_shape([None, None, 3])
        img = tf.cast(img, dtype=tf.float32)

        return self.image_processing(img, aug), tf.cast(features['label'], tf.int32)

    def image_processing(self, img, aug=True):
        if aug:
            img = tf.image.resize(img,
                                  tf.random.uniform([2], self.output_size, self.output_size + 16,
                                                    dtype=tf.int32))

            # img = self.tf_rotate(img)  # 旋转效果很差啊

            img = tf.image.random_crop(img, [self.output_size, self.output_size, 3])

            img = tf.image.random_brightness(img, 0.02)
            img = tf.image.random_hue(img, 0.01)
            img = tf.image.random_contrast(img, 0.99, 1.01)
            img = tf.image.random_flip_left_right(img)
        else:
            img = tf.image.resize(img, [self.output_size, self.output_size])

        img = tf.clip_by_value(img, 0, 255) / 127.5 - 1.

        return img

    # 旋转
    def tf_rotate(self, input_image, angle=math.pi / 6):
        '''
        TensorFlow对图像进行随机旋转
        :param input_image: 图像输入
        :param angle: 旋转角度范围
        :return: 旋转后的图像
        '''
        random_angles = tf.random.uniform(shape=(), minval=-angle, maxval=angle)
        x = tf.contrib.image.transform(input_image,
                                       tf.contrib.image.angles_to_projective_transforms(
                                           random_angles, tf.cast(tf.shape(input_image)[0], tf.float32),
                                           tf.cast(tf.shape(input_image)[0], tf.float32)))
        return x

    # 得到与数据集相关的TFRecord路径列表
    def get_filenames(self, dataset_name, cate):
        filenames = []
        root_dir = r'E:\TEST\ChangeFace\recognition\datasets'  # '../datasets'
        fns = os.listdir(root_dir)
        for fn in fns:
            if dataset_name in fn and cate in fn and fn.split('.')[-1] == 'tfrecord':
                filenames.append(os.path.join(root_dir, fn))
        return filenames

    # 根据数据集名称获取图片匹配对的数据集 [img1, img2, 1/0]
    def get_path_dataset(self, dataset_name='faces', cate='test', num=5000, aug=False):
        path = f'../datasets/{dataset_name}_{cate}.txt'
        with open(path, 'r') as f:
            _path_label = f.readlines()
        # paths: {label: [paths], ...}
        paths = {}
        for p, l in [p.strip().split(',') for p in _path_label]:
            l = int(l)
            if l not in paths: paths[l] = []
            paths[l].append(p)

        paths1 = []
        paths2 = []
        labels = []

        oo = 0
        while oo < num / 2:
            cls = np.random.randint(0, len(paths))
            pic_paths = paths[cls]  # 选中的相同的类别图片列表
            if len(pic_paths) > 0:
                im_no1 = np.random.randint(0, len(pic_paths))
                im_no2 = np.random.randint(0, len(pic_paths))
                if im_no1 != im_no2:
                    paths1.append(pic_paths[im_no1])
                    paths2.append(pic_paths[im_no2])
                    labels.append(1)
                    oo = oo + 1

        nn = 0
        while nn < num / 2:
            cls1 = np.random.randint(0, len(paths))
            cls2 = np.random.randint(0, len(paths))
            if cls1 != cls2:
                pic_paths1 = paths[cls1]
                pic_paths2 = paths[cls2]
                if len(pic_paths1) > 0 and len(pic_paths2) > 0:
                    im_no1 = np.random.randint(0, len(pic_paths1))
                    im_no2 = np.random.randint(0, len(pic_paths2))
                    paths1.append(pic_paths1[im_no1])
                    paths2.append(pic_paths2[im_no2])
                    labels.append(0)
                    nn = nn + 1

        dataset = tf.data.Dataset.from_tensor_slices((paths1, paths2, labels))
        dataset = dataset.map(lambda p1, p2, l: self.parser_path(p1, p2, l, aug),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def parser_path(self, path1, path2, label, aug=False):
        img1 = self._parser_path(path1, aug)
        img2 = self._parser_path(path2, aug)

        return img1, img2, label

    def _parser_path(self, path, aug=False):
        image_raw = tf.read_file(path)
        img = tf.image.decode_image(image_raw, channels=3)
        img.set_shape([None, None, 3])
        img = tf.cast(img, dtype=tf.float32)
        img = self.image_processing(img, aug)

        return img


if __name__ == '__main__':
    root_dirs = ['E:\TEST\AI\datasets\cfp-dataset\Data\Images',
                 'E:\TEST\AI\datasets\cnface_face',
                 'E:\TEST\AI\datasets\jpface_face',
                 'E:\TEST\AI\datasets\krface_face']
    names = ['cfp', 'faces']

    # get_pl_from_dir('E:\TEST\AI\datasets\cfp-dataset\Data\Images')

    make_tfrecord = 0
    read_tfrecord = 1
    if make_tfrecord:
        root_dirs = ['E:\TEST\AI\datasets\cfp-dataset\Data\Images']
        name = 'cfp'
        dm = TFRecordMaker()
        dm.writeTFRecord_from_dirs(root_dirs, name)
    if read_tfrecord:
        dm = DatasetMaker()
        dataset = dm.read('faces', 'train', aug=True).shuffle(13854).batch(128)
        iterator = dataset.make_initializable_iterator()
        data_init = iterator.initializer
        # iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        print(images.shape)
        with tf.Session() as sess:
            print('run data_init..')
            sess.run(data_init)
            print('done!')
            i = 0
            while True:
                try:
                    time.sleep(0.1)
                    print(f'\r{i}...', end='')
                    i += 1
                    xs, ys = sess.run([images, labels])
                except:
                    break
            print('-' * 100)
            for i, x in enumerate(xs[:10]):
                plt.imshow(((x + 1) / 2))
                plt.title(ys[i])
                plt.show()

            print('run data_init..')
            sess.run(data_init)
            print('done!')
            i = 0
            while True:
                try:
                    time.sleep(0.1)
                    print(f'\r{i}...', end='')
                    i += 1
                    xs, ys = sess.run([images, labels])
                except:
                    break
            print('-' * 100)
