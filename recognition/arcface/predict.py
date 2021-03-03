# -*- coding: utf-8 -*-
import os, sys, math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from recognition.arcface.train import parse_args
from recognition.arcface.utils import print_variables
from recognition.arcface.configs import config
from recognition.arcface.build_tensors import get_emb
from recognition.arcface.data_generator import DatasetMaker


def resize_img(img):
    if img.ndim < 3: img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    if img.shape[0] != config.img_size or img.shape[1] != config.img_size:
        img = cv2.resize(img, (config.img_size, config.img_size), interpolation=cv2.INTER_AREA)
    return img


class Predict:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build()

            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)

            # 读取保存变量
            restore_path = os.path.join(args.model_dir, f'v{args.restore_version}')
            loader = tf.train.Saver()
            loader.restore(self.sess, restore_path)
            print(f'SUCCEED: restore from {restore_path}.')

    def build(self):
        # 占位符
        self.inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])

        # 语义向量
        self.emb, _, uemb = get_emb(self.inputs, 'arcface', reuse=False, training=False, keep_prob=1,
                                    summary=False, get_unl2=True)

        # 相似度矩阵
        self.sim_matrix = tf.matmul(self.emb, self.emb, transpose_b=True)

        # 语义向量均值
        emb_average = tf.reduce_sum(self.emb, axis=0)
        self.emb_average = tf.nn.l2_normalize(emb_average)
        uemb_average = tf.reduce_sum(uemb, axis=0)
        self.uemb_average = tf.nn.l2_normalize(uemb_average)

    def debuge(self):

        path = r'E:\TEST\AI\datasets\test_face1\{name}'
        path1, path2 = path.format(name='lht1.jpg'), path.format(name='hd4.png')
        predict.get_sim_from_paths([path1, path2])

        path1, path2 = path.format(name='bqb1.png'), path.format(name='bqb3.png')
        predict.get_sim_from_paths([path1, path2])

        path1, path2 = path.format(name='cyh1.png'), path.format(name='cyh2.png')
        predict.get_sim_from_paths([path1, path2])

        path1, path2 = path.format(name='cyh1.png'), path.format(name='bqb3.png')
        predict.get_sim_from_paths([path1, path2])

        path1, path2 = path.format(name='hd2.png'), path.format(name='hd4.png')
        predict.get_sim_from_paths([path1, path2])

        path1, path2 = path.format(name='lmyly1.png'), path.format(name='lmyly2.png')
        predict.get_sim_from_paths([path1, path2])

    # 从single_dir中获取平均语义，把test_dir逐一与之对比
    def debuge1(self, single_dirs, test_dir):
        emb_averages = []
        for single_dir in single_dirs:
            _emb_average, _ = self.get_single_emb_from_dir(single_dir)
            emb_averages.append(_emb_average)
        emb_averages = np.array(emb_averages)  # [n, id_size]

        # test_dir中所有图片的语义
        names = [fn for fn in os.listdir(test_dir)
                 if os.path.isfile(os.path.join(test_dir, fn)) and fn.split('.')[-1] in ('png', 'jpg')]
        paths = [os.path.join(test_dir, fn) for fn in names]
        data_init, imgs = self.get_data_op(paths, len(paths))
        self.sess.run(data_init)
        imgs = self.sess.run(imgs)
        emb = self.sess.run(self.emb, {self.inputs: imgs})

        # 得到test_dir的所有图片与single_dirs的相似度
        sims = np.matmul(emb, emb_averages.T)
        print(f'FINISH: get similarity between {test_dir} and {single_dirs}.')

        n = len(names)
        for i, single_dir in enumerate(single_dirs):
            plt.figure(figsize=[10, 10])
            plt.plot(range(n), sims[:, i], linestyle="--", marker="*", linewidth=2.0, label=single_dir)
            plt.axis([-1, n, 0, 1])
            plt.xticks(range(n), [n.split('.')[0] for n in names])
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.legend()
            plt.grid()
            plt.xlabel('name')
            plt.xlabel('similarity')
            plt.title('name - sim with single')
            plt.show()

    def get_single_emb_from_dir(self, single_dir):
        # 得到single_dir的平均人脸
        paths = [os.path.join(single_dir, fn) for fn in os.listdir(single_dir)
                 if os.path.isfile(os.path.join(single_dir, fn)) and fn.split('.')[-1] in ('png', 'jpg')]
        data_init, imgs = self.get_data_op(paths, len(paths))
        self.sess.run(data_init)
        imgs = self.sess.run(imgs)
        emb_average, uemb_average = self.sess.run([self.emb_average, self.uemb_average], {self.inputs: imgs})
        print(f'FINISH: get face emb from {single_dir}.')
        return emb_average, uemb_average

    def get_sim_from_paths(self, paths, **kwargs):
        data_init, imgs = self.get_data_op(paths, len(paths))

        return self.get_sim(imgs, data_init, **kwargs)

    def get_data_op(self, paths, batch_size):
        with self.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices(paths).map(self._read_path).batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            data_init = iterator.initializer
            imgs = iterator.get_next()
        return data_init, imgs

    def _read_path(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.cast(img, tf.float32)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [config.img_size, config.img_size])
        img = img / 127.5 - 1
        return img

    def get_sim(self, imgs, data_init, plot=True, save_path=''):
        self.sess.run(data_init)
        imgs = self.sess.run(imgs)
        show_imgs = (imgs + 1) / 2

        sim_matrix = self.sess.run(self.sim_matrix, {self.inputs: imgs})
        sim = sim_matrix[0, 1]
        plt.figure(figsize=[14, 7])
        plt.subplot(121)
        plt.imshow(show_imgs[0])
        plt.subplot(122)
        plt.imshow(show_imgs[1])
        plt.suptitle(f'similarity: {sim:.3%}')
        if save_path: plt.imsave(save_path)
        if plot: plt.show()
        plt.close()

        return sim

    def get_sim_matrix_from_dir(self, pic_dir, **kwargs):
        names = [name.split('.')[0] for name in os.listdir(pic_dir)
                 if os.path.isfile(os.path.join(pic_dir, name))]
        paths = [os.path.join(pic_dir, name) for name in os.listdir(pic_dir)
                 if os.path.isfile(os.path.join(pic_dir, name))]

        data_init, imgs = self.get_data_op(paths, len(paths))

        if 'xylim' not in kwargs: kwargs['xylim'] = [-0.5, len(names) + 0.5]
        if 'xyticks' not in kwargs: kwargs['xyticks'] = [np.arange(0.5, len(names)), names]

        return self.get_sim_matrix(imgs, data_init, **kwargs)

    def get_sim_matrix(self, imgs, data_init, plot=True, save_path='', xylim=None, xyticks=None):
        self.sess.run(data_init)
        imgs = self.sess.run(imgs)

        sim_matrix = self.sess.run(self.sim_matrix, {self.inputs: imgs})

        # 分级
        sim_matrix[sim_matrix < 0] = 0
        sim_matrix = np.around(sim_matrix, 1)
        plt.figure(figsize=[sim_matrix.shape[0] // 2, sim_matrix.shape[0] // 2])
        sns.heatmap(sim_matrix, annot=True)
        if xylim is not None:
            plt.xlim(xylim)
            plt.ylim(xylim)
        if xyticks is not None:
            plt.xticks(*xyticks)
            plt.yticks(*xyticks)
        if save_path: plt.imsave(save_path)
        if plot: plt.show()
        plt.close()

        return sim_matrix

    def predict_dataset(self):
        with self.graph.as_default():
            dm = DatasetMaker()
            num = 10
            train_dataset = dm.get_path_dataset(args.dataset, cate='train', num=num, aug=False)
            test_dataset = dm.get_path_dataset(args.dataset, cate='test', num=num, aug=False)
            train_dataset = train_dataset.repeat().shuffle(num)
            test_dataset = test_dataset.repeat().shuffle(num)
            train_iterator = train_dataset.make_one_shot_iterator()
            test_iterator = test_dataset.make_one_shot_iterator()

            iterator_handle = tf.placeholder(tf.string, [])
            iterator = tf.data.Iterator.from_string_handle(iterator_handle, train_dataset.output_types,
                                                           train_dataset.output_shapes)

        train_handle, test_handle = self.sess.run([train_iterator.string_handle(), test_iterator.string_handle()])

        def _show(img1, img2, title):
            plt.figure(figsize=[14, 7])
            plt.subplot(121)
            plt.imshow((img1 + 1) / 2)
            plt.subplot(122)
            plt.imshow((img2 + 1) / 2)
            plt.suptitle(title)
            plt.show()

        # 对训练集
        for i in range(5):
            img1, img2, label = self.sess.run(iterator.get_next(), {iterator_handle: train_handle})
            sim = self.sess.run(self.sim_matrix, {self.inputs: [img1, img2]})
            _show(img1, img2, title=f'train_{i} - {label}: {sim[0, 1]:.3f}')

        # 对测试集
        for i in range(5):
            img1, img2, label = self.sess.run(iterator.get_next(), {iterator_handle: test_handle})
            sim = self.sess.run(self.sim_matrix, {self.inputs: [img1, img2]})
            _show(img1, img2, title=f'test_{i} - {label}: {sim[0, 1]:.3f}')

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    global args
    args = parse_args()

    predict = Predict()

    # # 从训练集、数据集中抽取相同、不同标签进行图片相似度对比
    # predict.predict_dataset()

    # 从single_dir中提取平均语义，与test_dir逐一比较
    single_dirs = [r'E:\TEST\AI\datasets\test_single_a',
                   r'E:\TEST\AI\datasets\test_single_cf',
                   r'E:\TEST\AI\datasets\test_single_lht',
                   r'E:\TEST\AI\datasets\test_single_pzy', ]
    test_dir = r'E:\TEST\AI\datasets\test_face1'
    predict.debuge1(single_dirs, test_dir)

    predict.debuge()

    # path = r'E:\TEST\AI\datasets\test_face1\{name}'
    # path1, path2 = path.format(name='lht1.jpg'), path.format(name='lht2.jpg')
    # predict.get_sim_from_paths([path1, path2])

    # # 从文件夹中得到相似度矩阵
    # predict.get_sim_matrix_from_dir('../valid_data')

    # pic_dir = r'E:\TEST\AI\datasets\test_face1'
    # predict.get_sim_matrix_from_dir(pic_dir)
    #
    # pic_dir = r'E:\TEST\AI\datasets\test_face2'
    # predict.get_sim_matrix_from_dir(pic_dir)

    predict.close()
