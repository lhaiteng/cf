# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# 获得lrs数组
def get_lrs(max_lr, end_epoch, lr_decay, num_epoch):
    start_lr, end_lr = max_lr / 100, max_lr / 100
    lrs = np.ones(num_epoch) * end_lr

    warmup_stage = 0
    lrs[:warmup_stage] = np.linspace(start_lr, max_lr, warmup_stage)

    decay_stage = end_epoch - warmup_stage
    _cos = np.cos(np.linspace(0, np.pi, decay_stage)) + 1
    lrs[warmup_stage: decay_stage + warmup_stage] = end_lr + (max_lr - end_lr) * _cos / 2

    plt.plot(range(num_epoch), lrs)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('lr-epoch curve')
    plt.show()

    return lrs


# 激活函数
def act(x, act_type, name=None):
    if act_type == 'relu':
        x = tf.nn.relu(x)
    elif act_type == 'leaky_relu':
        x = tf.nn.leaky_relu(x)
    else:
        x = tf.nn.relu(x)
    return x


def init_vars(sess, restore_path, restore_vars=None):
    try:
        loader = tf.train.Saver(var_list=restore_vars)
        loader.restore(sess, restore_path)
        print(f'SUCCEED: restore from {restore_path}')
    except:
        print(f'FAILED: restore from {restore_path}')


def print_variables(vars, name):
    print('*' * 100)
    print(f'{name}:')

    def get_var_size(var):
        s = 1
        for i in var.shape:
            s *= int(i)
        return s

    # 统计大小
    train_num = 0
    for v in vars:
        s = get_var_size(v)
        train_num += s
    # 打印占比
    for v in tf.trainable_variables():
        s = get_var_size(v)
        ratio = s / train_num if train_num > 0 else 0
        print('-' * 100)
        print(f'{v.name}\t\t{v.shape}\t\t{s}\t\t{ratio:.2%}')
    print('-' * 100)
    print(f'{name} parameters: {train_num}')
    print('*' * 100)


def check_dir(_dir):
    if not os.path.isdir(_dir):
        os.makedirs(_dir)


def check_dirs(*dirs):
    for _dir in dirs:
        check_dir(_dir)


def histplot_values(values, names, title='',
                    bins=None, xlim=None, xticks=None, save_path='', show=True):
    """
    把所有的数据分布显示在一起。values是二维数组，包含多个显示的图
    :param values:
    :param names:
    :param title:
    :param bins:
    :param xlim:
    :param xticks:
    :param save_path:
    :param show:
    :return:
    """
    plt.figure(figsize=[7, 7])
    num_distplot = len(values)

    for i in range(num_distplot):
        sns.distplot(values[i], bins=bins, hist=True, kde=True, rug=False,
                     vertical=False, hist_kws={'alpha': 0.3, 'edgecolor': 'k'}, label=names[i])
    plt.title(title)
    plt.legend()
    if xlim is not None: plt.xlim(xlim)
    if xticks is not None: plt.xticks(xticks)
    if save_path: plt.savefig(save_path)
    if show: plt.show()
    plt.close()


def histplot_angles(angles, names, title, **kwargs):
    assert len(angles) == len(names)
    if 'bins' not in kwargs: kwargs['bins'] = range(0, 180, 2)
    if 'xlim' not in kwargs: kwargs['xlim'] = [0, 180]
    if 'xticks' not in kwargs: kwargs['xticks'] = range(0, 190, 10)
    histplot_values(angles, names, title, **kwargs)


def evaluate_confusion_matrix(path=None, title=''):
    confusion_matrix = np.load(path)
    total = np.sum(confusion_matrix)
    recalls, precisions = np.zeros(284), np.zeros(284)
    for i in range(284):
        TPi = confusion_matrix[i, i]
        FPi = np.sum(confusion_matrix[:, i]) - TPi
        FNi = np.sum(confusion_matrix[i, :]) - TPi
        TNi = total - TPi - FPi - FNi
        recalls[i] = TPi / (TPi + FNi)
        precisions[i] = TPi / (TPi + FPi)
    plt.plot(range(284), recalls, label=f'recall={np.mean(recalls):.3f}')
    plt.plot(range(284), precisions, label=f'precisions={np.mean(precisions):.3f}')
    if title: plt.title(title)
    plt.legend()
    plt.show()

    F1 = 2 * recalls * precisions / (recalls + precisions + 1e-15)
    plt.plot(range(284), F1, label=f'F1={np.mean(F1):.3f}')
    if title: plt.title(title)
    plt.legend()
    plt.show()

    TP = confusion_matrix[range(284), range(284)]
    P0 = sum(TP) / total
    Pe = sum(np.sum(confusion_matrix, axis=1) * TP) / total ** 2
    K = (P0 - Pe) / (1 - Pe)
    print(f'K={K:.3f}')

    print('-' * 40)


if __name__ == '__main__':
    path = '../record/result/03-44/{cate}_confusion_matrix.npy'
    train_path = path.format(cate='train')
    evaluate_confusion_matrix(path=train_path, title='train')
    test_path = path.format(cate='test')
    evaluate_confusion_matrix(path=test_path, title='test')
