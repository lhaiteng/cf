# -*- coding: utf-8 -*-
"""
tf.data.TFRecordDataset
如果数据不够一个batch怎么办？
=> 最后一个批次输出剩余的数据，不满batch_size也输出

shuffle+batch会不会取到相同的数据？
=> shuffle是从剩余的所有数据中取，只要shuffle的buffer_size不能包括repeat后的所有数据，一个batch中就可能会出现重复数据

shuffle+batch+map？
=> map在batch前、后，结果一致xxx  不一致！！
=> 若使用x*random.random()，在构成图时，所乘随机数就固定下来了
"""
import random
import tensorflow as tf

"""创建数据生成器"""

# 根据文件创建
filenames = './test_dir/tfrecord'
dataset = tf.data.TFRecordDataset(filenames)  # fileNames指的是你要用tfrecord文件的路径
# # 读取的一个数据作为一个element

# 内置的数据生成
dataset = tf.data.Dataset.range(10)

dataset = dataset.repeat(count=2)  # 把原数据扩充count次，作为全部数据。None或-1表示无限扩充。0则清空数据了。
dataset = dataset.shuffle(buffer_size=100000)
# # 维护一个有buffer_size个element的buffer，从buffer中随机抽取element。每抽取一个，从剩余的样本中随机补充一个。
# # 当buffer_size >= 全部(repeat)数据总数时，效果较好。
dataset = dataset.batch(batch_size=32)  # 将batch_size个element作为一个输出返回
dataset = dataset.map(map_func=lambda x: x)  # 对输入的element做处理。
# # 如果使用的是用tfRecord数据，函数是解析数据的方式。


"""
1  迭代器
"""

"""
1.1  单次迭代器
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
最简单的迭代器形式，仅支持对数据集进行一次迭代，不需要显式初始化。
单次迭代器可以处理基于队列的现有输入管道支持的几乎所有情况，但它们不支持参数化
（也就是不能给他们赋不同的dataSet-别人的理解）
"""
# 简易创建数据
dataset = tf.data.Dataset.range(10)  # 等同于range(10)

# 数据迭代器
iterator = dataset.make_one_shot_iterator()
# 下个数据
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(100):  # 到10之后就会报错了
        value = sess.run(next_element)
        print(f'i={i}  value={value}')

# 设置无限循环
dataset = tf.data.Dataset.range(100)
dataset = dataset.shuffle(100)
# 数据迭代器
iterator = dataset.make_one_shot_iterator()
# 下个数据
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(100):
        value = sess.run(next_element)
        print(f'i={i}  value={value}')

# 使用batch
dataset = tf.data.Dataset.range(20).batch(2)  # 等同于range(10)
# 数据迭代器
iterator = dataset.make_one_shot_iterator()
# 下个数据
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(10):
        value = sess.run(next_element)
        print(f'i={i}  value={value}')

# 如果数据不够一个batch怎么办？
# => 最后一个批次输出剩余的数据，不满5个也输出
dataset = tf.data.Dataset.range(13).batch(5)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(10):
        value = sess.run(next_element)
        print(f'i={i}  value={value}')

# shuffle+batch会不会取到相同的数据？
# => shuffle是从剩余的所有数据中取，只要shuffle的buffer_size不能包括repeat后的所有数据，一个batch中就可能会出现重复数据
dataset = tf.data.Dataset.range(13).repeat().shuffle(20).batch(5)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(10):
        value = sess.run(next_element)
        print(f'i={i}  value={value}')

# shuffle+batch+map？
# => 此例中，使用x*random.random()，在构成图时，所乘随机数就固定下来了
# => map在batch前、后，结果一致
print('multiply random')
dataset = tf.data.Dataset.range(13).repeat().shuffle(20).batch(5)
dataset = dataset.map(lambda x: tf.cast(x, tf.float32) * random.random())
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(10):
        value = sess.run(next_element)
        print(f'{len(value) == len(set(value))}\ti={i}\tvalue={value}')
print('-' * 100)

# shuffle的buffer_size>range + repeat()
dataset = tf.data.Dataset.range(10).repeat().shuffle(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(20):
        print(i, sess.run(next_element))

"""
1.2  可初始化迭代器
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
需要先运行显式 iterator.initializer 操作，然后才能使用可初始化迭代器。
虽然有些不便，但它允许您使用一个或多个 tf.placeholder() 张量（可在初始化迭代器时馈送）参数化数据集的定义。
"""

# 使用占位符初始化

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        assert i == value

    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        assert i == value

# 使用占位符初始化

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(10).repeat(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 2})
    for i in range(20):
        value = sess.run(next_element)
        print(i, value)

# 使用初始化器初始化

tf.reset_default_graph()
max_value = tf.get_variable('max_value', [], tf.int64, initializer=tf.initializers.random_uniform(10, 50))
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('max_value=', sess.run(max_value))
    # Initialize
    sess.run(iterator.initializer)
    for i in range(10):
        value = sess.run(next_element)
        print(f'i={i}\tvalue={value}')

"""1.3 可重新初始化"""

iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)

"""1.4  可馈送"""

dataset1 = tf.data.Dataset.range(10, 20).batch(2)
dataset2 = tf.data.Dataset.range(5)
iterator1 = dataset1.make_initializable_iterator()
iterator2 = dataset2.make_one_shot_iterator()

handle = tf.placeholder(tf.string, [])
iterator = tf.data.Iterator.from_string_handle(handle, dataset1.output_types)  # , dataset1.output_shapes)
element = iterator.get_next()


with tf.Session() as sess:
    handle1, handle2 = sess.run([iterator1.string_handle(), iterator2.string_handle()])

    # 使用dataset1
    sess.run(iterator1.initializer)
    sess.run(element, {handle: handle1})
    # 使用dataset2
    sess.run(element, {handle: handle2})
    # 继续使用dataset1
    sess.run(element, {handle: handle1})
    sess.run(element, {handle: handle1})
    # 继续使用dataset2
    sess.run(element, {handle: handle2})
    sess.run(element, {handle: handle2})
"""
2  使用
使用时，当一个数据集到达末尾时候，会引发tf.errors.OutOfRangeError， 捕获这个error即代表数据集结束，
取数据用next_element = iterator.get_next()
"""

"""
3  例程
个人用官方的感觉有点复杂
个人理解并使用的是
用可重新初始化迭代器初始化训练数据(为了可以有训练完一个数据集的信号)，
用单次迭代器配合无限重复次数使用验证集（我的程序会运行一次训练集的同时运行测试集判断有没有过拟合）。
"""
"""3.1  创建数据集函数"""


def create_dataset(filenames, batch_size=8, is_shuffle=False, n_repeats=0):
    """

    :param filenames: record file names
    :param batch_size:
    :param is_shuffle: 是否打乱数据
    :param n_repeats:
    :return:
    """
    dataset = tf.data.TFRecordDataset(filenames)
    if n_repeats > 0:
        dataset = dataset.repeat(n_repeats)  # for train
    if n_repeats == -1:
        dataset = dataset.repeat()  # for val to
    dataset = dataset.map(lambda x: parse_single_exmp(x, labels_nums=NUM_CLASS))
    if is_shuffle:
        dataset = dataset.shuffle(10000)  # shuffle
    dataset = dataset.batch(batch_size)
    return dataset


""""3.2  预处理tfRecord解析函数"""


def parse_single_exmp(serialized_example, labels_nums=2):
    """
    解析tf.record
    :param serialized_example:
    :param opposite: 是否将图片取反
    :return:
    """
    features = tf.parse_single_example(serialized_example, features={'image_raw': tf.FixedLenFeature([], tf.string),
                                                                     'height': tf.FixedLenFeature([], tf.int64),
                                                                     'width': tf.FixedLenFeature([], tf.int64),
                                                                     'depth': tf.FixedLenFeature([], tf.int64),
                                                                     'label': tf.FixedLenFeature([], tf.int64)})
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)  # 获得图像原始的数据
    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
    tf_image = tf.reshape(tf_image, [224, 224, 3])  # 设置图像的维度
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image = prepeocess(tf_image, choice=True)
    tf_label = tf.one_hot(tf_label, labels_nums, 1, 0)
    print(tf_image)
    return tf_image, tf_label


"""运行"""

import tensorflow as tf
import os
import tensorflow.contrib.slim as slim

# 读取tfrecord文件并列成列表，train_dir是存放的路径
train_file_names = [os.path.join(train_dir, i) for i in os.listdir(train_dir)]
val_file_names = [os.path.join(val_dir, i) for i in os.listdir(val_dir)]

# 定义数据集
training_dataset = create_dataset(train_file_names, batch_size=BATCH_SIZE,
                                  is_shuffle=True, n_repeats=0)  # train_filename
# train_dataset 用epochs控制循环
validation_dataset = create_dataset(val_file_names, batch_size=BATCH_SIZE,
                                    is_shuffle=False, n_repeats=-1)  # val

# 定义迭代器
train_iterator = training_dataset.make_initializable_iterator()
# make_initializable_iterator 每个epoch都需要初始化
val_iterator = validation_dataset.make_one_shot_iterator()
# make_one_shot_iterator不需要初始化，根据需要不停循环
train_images, train_labels = train_iterator.get_next()
val_images, val_labels = val_iterator.get_next()

for epoch in range(NUM_EPOCHS):
    print('Starting epoch %d / %d' % (epoch + 1, NUM_EPOCHS))
    sess.run(train_iterator.initializer)
    while True:
        try:
            train_batch_images, train_batch_labels = sess.run([train_images, train_labels])
            _, train_loss, train_acc = sess.run([fc8_train_op, loss, accuracy],
                                                feed_dict={is_training: True,
                                                           images: train_batch_images,
                                                           labels: train_batch_labels})
            val_batch_images, val_batch_label = sess.run([val_images, val_labels])
            val_loss, val_acc = sess.run([loss, accuracy],
                                         feed_dict={is_training: False,
                                                    images: val_batch_images,
                                                    labels: val_batch_label})
            # step = sess.run(global_step)
            # print("global_step:{0}".format(step))
            print("epoch:{0}, train loss:{1},train-acc:{2}".format(epoch, train_loss, train_acc))
            print("epoch:{0}, val loss:{0},val-acc:{1}".format(epoch, val_loss, val_acc))
        except tf.errors.OutOfRangeError:
            break
