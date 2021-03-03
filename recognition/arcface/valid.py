# -*- coding: utf-8 -*-
"""
使用test数据集，把类内、类间的人脸两两配对，标签为1是同类、0不是同类，计算其tp、tn、fp、fn
"""
import os, time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from recognition.arcface.configs import config, default, update_config
from recognition.arcface.data_generator import DatasetMaker
from recognition.arcface.build_tensors import get_emb


def check_dir(fdir):
    if not os.path.isdir(fdir):
        os.makedirs(fdir)


def parse_args():
    parser = argparse.ArgumentParser(description='Train arcface network')
    # 使用输入的网络结构、数据集、损失
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--dataset', default=default.dataset, help='dataset config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    parser.add_argument('--initializer', default=default.initializer, help='initializer config')
    parser.add_argument('--optimizer', default=default.optimizer, help='optimizer config')
    # 根据输入的网络结构、数据集、损失，进行配置
    args, argv = parser.parse_known_args()  # 得到当前的输入
    update_config(args.network, args.dataset, args.loss, args.initializer, args.optimizer)

    # 载入其余用户参数
    for k, v in default.items():
        if k not in ('network', 'dataset', 'loss', 'initializer', 'optimizer'):
            _default, _type, _help = v
            parser.add_argument(f'--{k}', default=_default, type=_type, help=_help)

    # # To show the results of the given option to screen.
    # print('-' * 100)
    # print(f'USER arguments:')
    # for name, value in parser.parse_args()._get_kwargs():
    #     print(f'{name}: {value}')
    # print('-' * 100)

    args = parser.parse_args()  # 产生args

    return args


class Valid:
    def __init__(self, valid_dataset, valid_cate='test', valid_num=5000):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build(valid_dataset, valid_cate, valid_num)

            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)

            # 读取保存变量
            restore_path = os.path.join(args.model_dir, f'v{args.restore_version}')
            loader = tf.train.Saver()
            loader.restore(self.sess, restore_path)
            print(f'SUCCEED: restore from {restore_path}.')

            self.save_dir = f'../record/result/{args.restore_version}'
            check_dir(self.save_dir)

            self.cum_sims, self.cum_labels = self._get_sim_label()

    def build(self, valid_dataset, valid_cate, valid_num):
        dm = DatasetMaker()
        dataset = dm.get_path_dataset(valid_dataset, valid_cate, num=valid_num, aug=False)
        dataset = dataset.shuffle(valid_num)  # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(128)
        iterator = dataset.make_initializable_iterator()
        self.dataset_init = iterator.initializer
        imgs1, imgs2, self.labels = iterator.get_next()
        print(f'SUCCEED: make dataset from {valid_dataset}.')

        # 得到语义向量 - 已l2规范化
        emb1, _, = get_emb(imgs1, name='arcface', reuse=False, training=False,
                           keep_prob=1, summary=False, get_unl2=False)
        emb2, _, = get_emb(imgs2, name='arcface', reuse=True, training=False,
                           keep_prob=1, summary=False, get_unl2=False)

        self.sims = tf.reduce_sum(emb1 * emb2, axis=1)

    def get_metric(self, thresh=0.2, below_fpr=0.001):
        acc, p, r, fpr = self._cal_metric(self.cum_sims, self.cum_labels, thresh)
        acc_fpr, p_fpr, r_fpr, thresh_fpr = self._cal_metric_fpr(self.cum_sims, self.cum_labels, below_fpr)
        return acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr

    def _get_sim_label(self):
        self.sess.run(self.dataset_init)
        cum_sims, cum_labels = [], []
        cum_num = 0
        while True:
            try:
                sims, labels = self.sess.run([self.sims, self.labels])
                cum_num += labels.size
                cum_sims.extend(sims)
                cum_labels.extend(labels)
                print(f'\rcount num: {cum_num}...', end='')
            except:
                print(f'\nFINISHI: cum sims and labes.')
                break
        # 把sims和labels按照sim从大到小排列
        cum_sims, cum_labels = np.array(cum_sims), np.array(cum_labels)
        sorted_inds = np.argsort(cum_sims)[::-1]
        cum_sims = cum_sims[sorted_inds]
        cum_labels = cum_labels[sorted_inds]
        return cum_sims, cum_labels

    def _cal_metric(self, sims, labels, thresh):
        tp = tn = fp = fn = 0
        predict = np.zeros_like(sims)
        predict[sims >= thresh] = 1
        for i in range(sims.shape[0]):
            if predict[i] and labels[i]:
                tp += 1
            elif predict[i] and not labels[i]:
                fp += 1
            elif not predict[i] and not labels[i]:
                tn += 1
            else:
                fn += 1

        acc = 0 if sims.shape[0] == 0 else (tp + tn) / sims.shape[0]
        p = 0 if tp + fp == 0 else tp / (tp + fp)
        r = 0 if tp + fn == 0 else tp / (tp + fn)
        fpr = 0 if fp + tn == 0 else fp / (fp + tn)

        return acc, p, r, fpr

    def _cal_metric_fpr(self, sims, labels, below_fpr=0.001):
        acc = p = r = thresh = 0
        for t in np.linspace(-1, 1, 100):
            thresh = t
            acc, p, r, fpr = self._cal_metric(sims, labels, thresh)
            if fpr <= below_fpr:
                break

        return acc, p, r, thresh

    def draw_curve(self):
        save_dir = f'../record/result/{args.restore_version}'
        P = []
        R = []
        TPR = []
        FPR = []
        for thresh in np.linspace(-1, 1, 1000):
            acc, p, r, fpr = self._cal_metric(self.cum_sims, self.cum_labels, thresh)
            P.append(p)
            R.append(r)
            TPR.append(r)
            FPR.append(fpr)

        plt.axis([0, 1, 0, 1])
        plt.xlabel("R")
        plt.ylabel("P")
        plt.plot(R, P, color="r", linestyle="--", marker="*", linewidth=1.0)
        plt.title('P-R curve')
        plt.savefig(os.path.join(save_dir, 'P-R curve.png'))
        plt.show()

        plt.axis([0, 1, 0, 1])
        plt.xlabel("FRP")
        plt.ylabel("TPR")
        plt.plot(FPR, TPR, color="r", linestyle="--", marker="*", linewidth=1.0)
        plt.title('FRP-TPR curve')
        plt.savefig(os.path.join(save_dir, 'FRP-TPR curve.png'))
        plt.show()

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    global args
    args = parse_args()

    valid_dataset, valid_cate, valid_num = 'faces', 'train', 5000
    vd = Valid(valid_dataset, valid_cate, valid_num)
    acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = vd.get_metric(thresh=0.2, below_fpr=0.001)
    print(f'valid dataset {valid_dataset} - {valid_cate}.')
    print(f'thres=0.2: acc={acc:.3%}, p={p:.3f}, r={r:.3f}, fpr={fpr:.3f}')
    print(f'below_fpr=0.001: acc_fpr={acc_fpr:.3%}, p_fpr={p_fpr:.3f}, r_fpr={r_fpr:.3f}, thresh_fpr={thresh_fpr:.3f}')
    print('-' * 100)
    vd.draw_curve()
    vd.close()

    time.sleep(5)

    valid_dataset, valid_cate, valid_num = 'faces', 'test', 5000
    vd = Valid(valid_dataset, valid_cate, valid_num)
    acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = vd.get_metric(thresh=0.2, below_fpr=0.001)
    print(f'valid dataset {valid_dataset} - {valid_cate}.')
    print(f'thres=0.2: acc={acc:.3%}, p={p:.3f}, r={r:.3f}, fpr={fpr:.3f}')
    print(f'below_fpr=0.001: acc_fpr={acc_fpr:.3%}, p_fpr={p_fpr:.3f}, r_fpr={r_fpr:.3f}, thresh_fpr={thresh_fpr:.3f}')
    print('-' * 100)
    vd.draw_curve()
    vd.close()



