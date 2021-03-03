# -*- coding: utf-8 -*-
"""
创建和使用TFRecord

"""

import tempfile
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

"""创建数据2"""

path = '../../data/Imgnet2015_32/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def img_to_feature_2(img_path):
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
        feature = tf.train.Features(
            feature={'img': _bytes_feature(encoded_jpg),
                     'format': _bytes_feature('jpeg'.encode('utf8'))
                     })
        return feature


def creat_tfrecord(class_path):
    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path="Imgnet2015_32.tfrecords", options=writer_options)
    for img_name in os.listdir(class_path):  # os.listdir 返回指定目录下所有的文件和目录
        img_path = class_path + img_name  # 每一个图片的地址
        features = img_to_feature_2(img_path)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


slim_example_decoder = tf.contrib.slim.tfexample_decoder

def get_read_data(filename, size):
    filename_queue = tf.train.string_input_producer([filename])
    tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader(options=tfrecord_options)
    _, serialized_example = reader.read(filename_queue)
    keys_to_features = {
        'img': tf.FixedLenFeature((), tf.string, default_value=''),
        'format': tf.FixedLenFeature((), tf.string, default_value='jpeg')
    }
    items_to_handlers = {
        'img': slim_example_decoder.Image(image_key='img', format_key='format', channels=3),
        'format': slim_example_decoder.Tensor('format')
    }
    serialized_example = tf.reshape(serialized_example, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    # format = tensor_dict['format']
    img_tensor = tensor_dict['img']
    img_tensor = tf.reshape(img_tensor, [size, size, 3])
    img_float = tf.cast(img_tensor, tf.float32)
    return img_tensor, img_float


# 用tf.cast转为tf.float格式后保存无法还原成原图像
def run(input_path, size):
    img_tensor, img_float = get_read_data(input_path, size)
    img_batch = tf.train.shuffle_batch([img_float], batch_size=8, capacity=64, min_after_dequeue=32)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            img_b, img = sess.run([img_batch, img_tensor])
            image = Image.fromarray(img, 'RGB')  # 这里Image是之前提到的
            image.save('./output/' + str(i) + '_''Label_' + '.jpg')  # 存下图片
            # print ('img:', img)
            print('img shape:', img_b.shape)
            print('img shape:', img.shape)
            # print('img format:', format_name)

        coord.request_stop()
        coord.join(threads)


creat_tfrecord(path)
# run('test32.tfrecords',32)


"""创建数据"""
# 图片路径
cwd = r'E:\Convert-between-TFRecords-and-Images\flower_photos'
# 生成的tfrecord文件路径
filepath = r'E:\Convert-between-TFRecords-and-Images\tfrecord'
# 存放图片个数
bestnum = 1000
# 第几个图片
num = 0
# 第几个TFRecord文件
recordfilenum = 0
# 类别
classes = ['daisy',
           'dandelion',
           'roses',
           'sunflowers',
           'tulips']
# tfrecords格式文件名
ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
# 通过这一句将数据写入到TFRecord文件
writer = tf.python_io.TFRecordWriter(filepath + ftrecordfilename)
# 类别和路径
for index, name in enumerate(classes):
    print(index)
    print(name)
    class_path = cwd + '\\' + name + '\\'
    for img_name in os.listdir(class_path):
        num = num + 1
        if num > bestnum:
            num = 1
            recordfilenum = recordfilenum + 1  # 每一个tfrecord文件打包1000张图片
            # tfrecords格式文件名
            ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
            # 通过tf.python_io.TFRecordWriter 将数据写入到TFRecords文件
            writer = tf.python_io.TFRecordWriter(filepath + ftrecordfilename)
        # print('路径',class_path)
        # print('第几个图片：',num)
        # print('文件的个数',recordfilenum)
        # print('图片名：',img_name)

        img_path = class_path + img_name  # 每一个图片的地址
        img = Image.open(img_path, 'r')
        size = img.size
        # print(size[1],size[0])
        # print(size)
        # print(img.mode)
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        # 将一张图的4个信息打包到 TFRecord 中
        feature = {'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                   'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                   'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                   'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()

"""解析数据"""

# 图片路径
swd = r'E:\Convert-between-TFRecords-and-Images\flower_photos'
# TFRecord文件路径
data_path = r'E:\Convert-between-TFRecords-and-Images\tfrecordtraindata.tfrecords-000'
# 获取文件名列表
data_files = tf.gfile.Glob(data_path)
print(data_files)
# 文件名列表生成器
# string_input_producer()用于读取大的数据集，每次放出一个文件名
filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
# 取出包含image和label的feature对象
features = tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64),
                                                                 'img_raw': tf.FixedLenFeature([], tf.string),
                                                                 'img_width': tf.FixedLenFeature([], tf.int64),
                                                                 'img_height': tf.FixedLenFeature([], tf.int64), })
# tf.decode_raw可以将字符串解析成图像对应的像素数组
image = tf.decode_raw(features['img_raw'], tf.uint8)
height = tf.cast(features['img_height'], tf.int32)
width = tf.cast(features['img_width'], tf.int32)
label = tf.cast(features['label'], tf.int32)
channel = 3
image = tf.reshape(image, [height, width, channel])

with tf.Session() as sess:  # 开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 启动多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(15):
        # image_down = np.asarray(image_down.eval(), dtype='uint8')
        plt.imshow(image.eval())
        plt.show()
        single, l = sess.run([image, label])  # 在会话中取出image和label
        img = Image.fromarray(single, 'RGB')  # 将数组转换为image
        img.save(swd + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
        # print(single,l)
    coord.request_stop()
    coord.join(threads)

"""解析数据2"""


# 以下解析TFRecord文件里的数据。读取文件为本章第一节创建的文件
def parser(record):
    features = tf.parse_single_example(record, features={'image_raw': tf.FixedLenFeature([], tf.string),
                                                         'pixels': tf.FixedLenFeature([], tf.int64),
                                                         'label': tf.FixedLenFeature([], tf.int64)})

    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int32)
    return images, labels, pixels


# 从TFRecord文件创建数据集。这里可以提供多个文件。
input_files = ["output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)  # 看，看，看，这次又换了

# map()函数表示对数据集中的每一条数据进行调用解析方法。
dataset = dataset.map(parser)  # 这是一个很常用的套路，要学会， 表示对dataset中的数据进行parser操作

# 定义遍历数据集的迭代器。
iterator = dataset.make_one_shot_iterator()

# 读取数据，可用于进一步计算
image, label, _ = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image, label])
        print(y)
