# -*- coding: utf-8 -*-
import os, sys, math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from recognition.arcface.configs import config, default, update_config
from recognition.arcface.build_tensors import get_emb, get_loss, get_optimizer
from recognition.arcface.data_generator import DatasetMaker
from recognition.arcface.utils import init_vars, print_variables, check_dir, check_dirs, get_lrs


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


class Trainer:
    def __init__(self):

        check_dirs(args.model_dir, args.log_dir, args.gen_pic_dir)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.build()
            print_variables(tf.trainable_variables(), 'trainable_variables')

            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)

            # 保存器
            self.saver = tf.train.Saver()
            # print_variables(tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS),
            #                 'SAVEABLE_OBJECTS')

            print('START: init all vars.')
            self.sess.run(tf.global_variables_initializer())

            # # 挑选变量读取
            # # restore_vars = [v for v in tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
            # #                 if 'arcface' in v.name and 'cls_weight' not in v.name and 'Adam' not in v.name]
            # restore_path = os.path.join(args.model_dir, f'v{args.restore_version}')
            # print_variables(restore_vars, 'restore_vars')
            # init_vars(self.sess, restore_path, restore_vars)

            # # 继续训练
            # restore_path = os.path.join(args.model_dir, f'v{args.restore_version}')
            # self.saver.restore(self.sess, restore_path)
            # print(f'SUCCEED: restore from {restore_path}.')

    def build(self):
        dm = DatasetMaker()
        train_dataset = dm.read(args.dataset, cate='train', aug=True)
        test_dataset = dm.read(args.dataset, cate='test', aug=False)
        train_dataset = train_dataset.shuffle(config.num_train_pics).batch(args.batch_size)
        test_dataset = test_dataset.repeat().shuffle(config.num_test_pics).batch(128)
        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_one_shot_iterator()

        self.iterator_handle = tf.placeholder(tf.string, [])
        iterator = tf.data.Iterator.from_string_handle(self.iterator_handle, train_dataset.output_types,
                                                       train_dataset.output_shapes)
        imgs, labels = iterator.get_next()
        labels_onehot = tf.one_hot(labels, config.num_cls)
        print(f'SUCCEED: make dataset from {args.dataset}.')

        # 占位符
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        self.training = tf.placeholder(tf.bool, [], name='training')
        # 全局步数
        self.global_step = tf.get_variable('global_step', shape=[],
                                           initializer=tf.constant_initializer(0), trainable=False)

        # 得到语义向量 - 已l2规范化
        emb, self.l2_cls_weight = get_emb(imgs, name='arcface', reuse=False, training=self.training,
                                          keep_prob=args.keep_prob, summary='train')

        _loss = self._get_loss(emb, labels, labels_onehot, summary='train')
        self.mean_angle1, self.mean_angle2, self.acc, self.loss = _loss

        # 优化器
        opt = get_optimizer(self.lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = opt.minimize(self.loss, global_step=self.global_step)

        # 记录
        self.summary_train = tf.summary.merge(tf.get_collection('train_summary'))

        """用于测试的部分"""
        # 得到语义向量 - 已l2规范化
        emb, _ = get_emb(imgs, name='arcface', reuse=True, training=False, keep_prob=1, summary='test')

        _loss = self._get_loss(emb, labels, labels_onehot, summary='test')
        self.test_mean_angle1, self.test_mean_angle2, self.test_acc, self.test_loss = _loss

        self.summary_test = tf.summary.merge(tf.get_collection('test_summary'))

    def _get_loss(self, emb, labels, labels_onehot, summary=''):
        # 夹角
        cosines = tf.matmul(emb, self.l2_cls_weight)
        radians = tf.acos(cosines)
        angles = radians * 180. / math.pi
        _bs = tf.cast(tf.shape(labels_onehot)[0], tf.float32)
        _ncls = tf.cast(labels_onehot.shape[1], tf.float32)

        sum_angels = tf.reduce_sum(angles)
        # 夹角矩阵
        angles1 = labels_onehot * angles
        # 求均值
        sum_angels1 = tf.reduce_sum(angles1)
        mean_angle1 = sum_angels1 / _bs
        mean_angle2 = (sum_angels - sum_angels1) / (_bs * (_ncls - 1))
        # 准确率
        acc = tf.equal(tf.argmax(cosines, axis=1, output_type=labels.dtype), labels)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

        if summary:
            tf.add_to_collection(f'{summary}_summary',
                                 tf.summary.histogram(f'angle/{summary}_distribution_all', angles))
            tf.add_to_collection(f'{summary}_summary',
                                 tf.summary.histogram(f'angle/{summary}_distribution_label',
                                                      tf.reduce_sum(angles1, axis=1)))
            tf.add_to_collection(f'{summary}_summary',
                                 tf.summary.scalar(f'angle/{summary}_mean_label', mean_angle1))
            tf.add_to_collection(f'{summary}_summary',
                                 tf.summary.scalar(f'angle/{summary}_mean_unlabel', mean_angle2))
            tf.add_to_collection(f'{summary}_summary',
                                 tf.summary.scalar(f'acc/{summary}', acc))

        # 损失
        loss = get_loss(cosines, radians, labels_onehot, summary)

        return mean_angle1, mean_angle2, acc, loss

    def train(self):
        print('-' * 100)
        print(f'USER arguments:')
        for name, value in vars(args).items():
            print(f'{name}: {value}')
        print('-' * 100)
        print(f'TRAIN config:')
        for name, value in config.items():
            print(f'{name}: {value}')
        print('-' * 100)

        save_path = os.path.join(args.model_dir, f'v{args.version}')
        self.saver.export_meta_graph(save_path + '.meta')

        # 步数和批次
        num_epoch = args.num_epoch
        batch_size = args.batch_size
        num_batch = config.num_train_pics // batch_size
        sess = self.sess
        # 得到数据集handle
        train_handle, test_handle = sess.run([self.train_iterator.string_handle(), self.test_iterator.string_handle()])
        self.callback_epoch(test_handle)  # 运行一次测试集进行测试

        writer = tf.summary.FileWriter(args.log_dir, graph=self.graph)
        step = 0
        run_list = [self.train_op, self.acc, self.mean_angle1, self.mean_angle2, self.loss, self.summary_train]
        feed_dict = {self.training: True, self.iterator_handle: train_handle}
        lrs = get_lrs(args.max_lr, args.end_epoch, args.lr_decay, num_epoch)
        # lrs = np.ones_like(lrs) * lrs[-1]
        print(f'lrs = \n{lrs}')
        for i in range(num_epoch):
            sess.run(self.train_iterator.initializer)
            feed_dict[self.lr] = lrs[i]
            cum_acc = 0  # 本轮的累积准确率
            for j in range(num_batch):
                step = sess.run(self.global_step)
                _, acc, mean_angle1, mean_angle2, loss, summary = sess.run(run_list, feed_dict)
                cum_acc = cum_acc * 0.95 + acc * 0.05
                print(f'\rEPOCH: {i}/{num_epoch} BATCH: {j}/{num_batch} step: {step} cum_acc: {cum_acc:.3%} - '
                      f'acc: {acc:.3%} mean_angle1: {mean_angle1:.3f} mean_angle2: {mean_angle2:.3f} loss: {loss:.3f}',
                      end='')
                if step % 10 == 0: writer.add_summary(summary, global_step=step)
                # j += 1
            self.saver.save(sess, save_path, global_step=i, write_meta_graph=False)
            print(f'\tsucceed save model into {save_path}')
            self.callback_epoch(test_handle, writer, step)
            print('-' * 100)

    def callback_epoch(self, handle, writer=None, step=0):
        # 对测试集测试
        if writer is None:
            run_list = [self.test_acc, self.test_mean_angle1, self.test_mean_angle2, self.test_loss]
            feed_dict = {self.training: False, self.iterator_handle: handle}
            acc, mean_angle1, mean_angle2, loss = self.sess.run(run_list, feed_dict)
        else:
            run_list = [self.test_acc, self.test_mean_angle1, self.test_mean_angle2, self.test_loss,
                        self.summary_test]
            feed_dict = {self.training: False, self.iterator_handle: handle}
            acc, mean_angle1, mean_angle2, loss, summary = self.sess.run(run_list, feed_dict)
            writer.add_summary(summary, global_step=step)
        print(f'TEST - acc: {acc:.3%} mean_angle1: {mean_angle1:.3f} mean_angle2: {mean_angle2:.3f} loss: {loss:.3f}')

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    global args

    # 预训练
    args = parse_args()
    trainer = Trainer()

    trainer.train()

    trainer.close()
