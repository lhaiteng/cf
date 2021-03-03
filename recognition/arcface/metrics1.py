# -*- coding: utf-8 -*-
"""
基于cls_weight，
计算准确率、夹角分布，混淆矩阵与K、F1score
"""
import os, sys, math
import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from recognition.arcface.train import parse_args
from recognition.arcface.configs import config, default, update_config
from recognition.arcface.build_tensors import get_emb
from recognition.arcface.data_generator import DatasetMaker
from recognition.arcface.utils import init_vars, histplot_angles, print_variables


class Metrics:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.build()
            print_variables(tf.global_variables(), 'global_variables')

            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)

            # self.sess.run(tf.global_variables_initializer())
            # 读取保存变量
            restore_path = os.path.join(args.model_dir, f'v{args.restore_version}')
            loader = tf.train.Saver()
            loader.restore(self.sess, restore_path)
            print(f'SUCCEED: restore from {restore_path}.')

            self.save_dir = f'../record/result/{args.restore_version}'
            self.check_dir(self.save_dir)

    def build(self):
        dm = DatasetMaker()
        train_dataset = dm.read(args.dataset, cate='train', aug=False)
        train_dataset = train_dataset.shuffle(1024).batch(128)
        train_iterator = train_dataset.make_initializable_iterator()
        self.train_dataset_init = train_iterator.initializer
        self.train_imgs, self.train_labels = train_iterator.get_next()
        self.train_labels_onehot = tf.one_hot(self.train_labels, config.num_cls)
        print(f'SUCCEED: make train dataset from {args.dataset}.')

        # # 人脸语义均值的占位符
        # self.input_emb_average = tf.placeholder(tf.float32, [config.num_cls, config.embedding_size],
        #                                         name='input_emb_average')

        # 得到语义向量 - 已l2规范化
        self.train_emb, self.l2_cls_weight = get_emb(self.train_imgs, name='arcface', reuse=False, training=False,
                                                     keep_prob=1, summary=False, get_unl2=False)

        _args = [self.train_emb, self.l2_cls_weight, self.train_labels]
        self.train_angles, self.train_predict, self.train_acc_num = self.get_metrics(*_args)

        """用于测试的部分"""
        test_dataset = dm.read(args.dataset, cate='test', aug=False)
        test_dataset = test_dataset.shuffle(1024).batch(128)
        test_iterator = test_dataset.make_initializable_iterator()
        self.test_dataset_init = test_iterator.initializer
        self.test_imgs, self.test_labels = test_iterator.get_next()
        self.test_labels_onehot = tf.one_hot(self.test_labels, config.num_cls)
        print(f'SUCCEED: make test dataset from {args.dataset}.')

        # 得到语义向量 - 已l2规范化
        emb, _, = get_emb(self.test_imgs, name='arcface', reuse=True, training=False,
                          keep_prob=1, summary=False, get_unl2=False)

        _args = [emb, self.l2_cls_weight, self.test_labels]
        self.test_angles, self.test_predict, self.test_acc_num = self.get_metrics(*_args)

    # 对样本进行评价，得到夹角矩阵，预测结果，预测正确的数量
    def get_metrics(self, emb, emb_average, labels):
        # cosines = tf.matmul(emb, emb_average, transpose_b=True)
        cosines = tf.matmul(emb, emb_average)
        radians = tf.acos(cosines)
        angles = radians * 180 / math.pi  # 夹角

        predict = tf.argmax(cosines, axis=1, output_type=labels.dtype)
        acc_num = tf.reduce_sum(tf.cast(tf.equal(predict, labels), tf.float32))

        return angles, predict, acc_num

    def com_all(self):

        self._com_all('train')
        self._com_all('test')

    def _com_all(self, cate):
        if cate == 'train':
            rum_list = [self.train_angles, self.train_labels, self.train_labels_onehot,
                        self.train_predict, self.train_acc_num]
            dataset_init = self.train_dataset_init
        else:
            rum_list = [self.test_angles, self.test_labels, self.test_labels_onehot,
                        self.test_predict, self.test_acc_num]
            dataset_init = self.test_dataset_init

        cum_acc_num, cum_num = 0, 0
        cum_angles1, cum_angles2 = [], []
        cum_labels, cum_predict = [], []
        self.sess.run(dataset_init)
        while True:
            try:
                res = self.sess.run(rum_list)
                angles, labels, labels_onehot, predict, acc_num = res
                cum_num += labels.size
                cum_acc_num += acc_num
                cum_angles1.extend(angles[labels_onehot == 1])
                cum_angles2.extend(angles[labels_onehot == 0])
                cum_labels.extend(labels)
                cum_predict.extend(predict)
                print(f'\r{cate} count num: {cum_num} cum_acc: {cum_acc_num / cum_num:.3%}...', end='')
            except:
                print()
                break

        # 最终的夹角均值和夹角分布
        acc = cum_acc_num / cum_num
        print(f'\n{cate}: num: {cum_num} acc: {acc:.3%} '
              f'mean_angles1: {np.mean(cum_angles1):.3f} mean_angles2: {np.mean(cum_angles2):.3f}')
        np.save(os.path.join(self.save_dir, f'{cate}_acc.npy'), acc)
        np.save(os.path.join(self.save_dir, f'{cate}_angles1.npy'), cum_angles1)
        np.save(os.path.join(self.save_dir, f'{cate}_angles2.npy'), cum_angles2)
        # 画图
        save_path = os.path.join(self.save_dir, f'{cate}_angle_distribution.png')
        histplot_angles([cum_angles1, cum_angles2], ['angles1', 'angles2'],
                        title=f'{cate} angle distribution', show=True, save_path=save_path)

        # 混淆矩阵
        confusion_matrix = np.zeros([config.num_cls, config.num_cls])
        lp = list(zip(cum_labels, cum_predict))
        np.save(os.path.join(self.save_dir, f'{cate}_label_predict'), np.array(lp))

        rcn = np.array([[i[0], i[1], lp.count(i)] for i in set(zip(cum_labels, cum_predict))])
        confusion_matrix[rcn[:, 0], rcn[:, 1]] += rcn[:, 2]
        np.save(os.path.join(self.save_dir, f'{cate}_confusion_matrix'), confusion_matrix)
        print(f'\nSUCCEED get_confusion_matrix from {cate}.')

        # 画出混淆矩阵
        plt.figure(figsize=[10, 10])
        sns.heatmap(confusion_matrix)
        plt.title(f'{cate} confusion_matrix')
        plt.xlim([-0.5, config.num_cls + 0.5])
        plt.ylim([-0.5, config.num_cls + 0.5])
        plt.savefig(os.path.join(self.save_dir, f'{cate}_confusion_matrix.png'))
        plt.show()

        # 评价混淆矩阵
        self.metrics_cm(confusion_matrix, f'{cate}')

    # 对混淆矩阵进行评价
    def metrics_cm(self, confusion_matrix, title):
        total = np.sum(confusion_matrix)
        recalls, precisions = np.zeros(config.num_cls), np.zeros(config.num_cls)
        for i in range(config.num_cls):
            TPi = confusion_matrix[i, i]
            FPi = np.sum(confusion_matrix[:, i]) - TPi
            FNi = np.sum(confusion_matrix[i, :]) - TPi
            TNi = total - TPi - FPi - FNi
            recalls[i] = 0 if TPi + FNi == 0 else TPi / (TPi + FNi)
            precisions[i] = 0 if TPi + FPi == 0 else TPi / (TPi + FPi)

        plt.figure(figsize=[12, 12])

        plt.subplot(211)
        plt.plot(range(config.num_cls), recalls, label=f'recall={np.mean(recalls):.3f}')
        plt.plot(range(config.num_cls), precisions, label=f'precisions={np.mean(precisions):.3f}')
        plt.title(title)
        plt.legend()

        plt.subplot(212)
        F1 = 2 * recalls * precisions / (recalls + precisions + 1e-15)
        plt.plot(range(config.num_cls), F1, label=f'F1={np.mean(F1):.3f}')
        plt.title(title)
        plt.legend()

        TP = confusion_matrix[range(config.num_cls), range(config.num_cls)]
        P0 = sum(TP) / total
        Pe = sum(np.sum(confusion_matrix, axis=1) * TP) / total ** 2
        K = (P0 - Pe) / (1 - Pe)
        plt.suptitle(f'{title} K={K:.3f}')
        plt.show()
        print(f'K={K:.3f}')

        print('-' * 100)

    # 计算训练集的平均语义，并保存
    def com_emb_average(self):
        emb_sum = np.zeros([config.num_cls, config.embedding_size])
        num_count = np.zeros([config.num_cls, 1])
        cum_labels, cum_emb = [], []
        self.sess.run(self.train_dataset_init)
        i = 0
        while True:
            try:
                labels, emb = self.sess.run([self.train_labels, self.train_uemb])
                # 可能有重复的labels，所以和混淆矩阵一样，应该先统计所有的emb和labels，再计算
                i += labels.size
                cum_labels.extend(labels)
                cum_emb.extend(emb)
                print(f'\rcom emb average count num: {i}...', end='')
            except:
                print()
                break

        cum_labels = np.array(cum_labels)
        cum_emb = np.array(cum_emb)

        for i in range(config.num_cls):
            emb_sum[i] = np.sum(cum_emb[cum_labels == i, :], axis=0)
            # num_count[i] = cum_labels[cum_labels == i].size

        # emb_average = emb_sum / np.maximum(num_count, 1e-9)
        # emb_average = emb_average / np.sqrt(np.sum(np.square(emb_average), axis=1, keepdims=True))
        emb_average = emb_sum / np.linalg.norm(emb_sum, axis=1, keepdims=True)

        save_path = os.path.join(self.save_dir, 'emb_average.npy')
        np.save(save_path, emb_average)
        print(f'SUCCEED: save emb_average into {save_path}.')

        return emb_average

    def close(self):
        self.sess.close()

    def check_dir(self, fdir):
        if not os.path.isdir(fdir):
            os.makedirs(fdir)


if __name__ == '__main__':
    global args
    args = parse_args()
    metrics = Metrics()

    metrics.com_all()

    metrics.close()
