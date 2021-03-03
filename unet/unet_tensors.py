# -*- coding: utf-8 -*-
import tensorflow as tf
from unet.unet_config import unet_cfg as cfg
from backbones.unet_v1 import unet


class Tensor:
    def __init__(self):
        with tf.device('/gpu: 0'):
            # 占位符
            self.lr = tf.placeholder(tf.float32, shape=[], name='unet_lr')
            self.training = tf.placeholder(tf.bool, shape=[], name='unet_training')
            self.keep_prob = tf.placeholder(tf.float32, shape=[], name='unet_keep_prob')
            # 初始化器
            normal_init = tf.initializers.truncated_normal(cfg.init_mean,
                                                           cfg.init_std)
            # 优化器
            opt = tf.train.AdamOptimizer(self.lr)

            with tf.variable_scope('unet', reuse=False):
                # 计步器
                self.global_step = tf.get_variable('global_step', shape=[],
                                                   initializer=tf.constant_initializer(0),
                                                   trainable=False)
                self.add_global_step = tf.assign_add(self.global_step, 1)

        self.sub_ts = []
        for ind_gpu in range(cfg.gpu_num):
            first = False if ind_gpu == 0 else True
            with tf.device(f'/gpu:{ind_gpu}'):
                print(f'GPU: {ind_gpu}')
                self.sub_ts.append(SubTensor(normal_init, opt,
                                             self.training, self.keep_prob, first))
        print('FINISH appending subtensors.')

        with tf.device('/gpu:0'):
            print('Merging grads...')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                grads = self.merge_grads(lambda ts: ts.grads)
                # # [(grad, var), (grad, avr), ...]

            """在tf中进行多批次平均优化"""
            # 更新一次需要几个批次
            self.train_num_batch = tf.placeholder(tf.float32, shape=[], name='train_num_batch')

            # 先得到相同形状的grads以待求和
            # 必须是变量形式，否则不能assign。不可trainable
            self.grads = {gv[1]: tf.get_variable(name=f'train_grads{ind}', shape=gv[0].shape,
                                                 initializer=tf.initializers.zeros(), trainable=False)
                          for ind, gv in enumerate(grads)}
            # 分别对每个梯度进行初始化。对应的变量v应该只是个指针不是个值，所以不用再次指向新的变量值
            self.assign_zero_grads = tf.initialize_variables([g for v, g in self.grads.items()])
            # 赋值op的列表, 分别将梯度累加进去
            self.assign_grads = [tf.assign_add(self.grads[v], g)
                                 for g, v in grads]

            self.update_grads = opt.apply_gradients([(g / self.train_num_batch, v)
                                                     for v, g in self.grads.items()])

            self.print_variable()

            print('Reduce_meaning loss1...')
            self.loss = tf.reduce_mean([_ts.loss for _ts in self.sub_ts])

            # 用来恢复的var_list
            vars_restore = [v for v in tf.trainable_variables()
                            if 'unet' in v.name]
            vars_restore_moving = [v for v in tf.global_variables()
                                   if 'unet' in v.name and 'moving' in v.name]
            self.vars_restore = vars_restore + vars_restore_moving

            # 用来储存的var_list
            vars_save = [v for v in tf.trainable_variables() if 'unet' in v.name]
            vars_save_moving = [v for v in tf.global_variables()
                                if 'unet' in v.name and 'moving' in v.name]
            self.vars_save = vars_save + vars_save_moving
            print(f'unet vars_save:\n{self.vars_save}')

    def merge_grads(self, f):
        """
        ts.grads [(grad, var), (grad, var), ...]
        :return: [(grad, var), (grad, var), ...]
        """
        var_grad = {}  # var: [grad1, grad2, ...]
        var_IndexedSlices = {}  # var: [IndexedSlices1, IndexedSlices2, ...]
        for ts in self.sub_ts:
            for grad, var in f(ts):
                if grad is None:
                    continue
                if isinstance(grad, tf.IndexedSlices):
                    if var not in var_IndexedSlices:
                        var_IndexedSlices[var] = []
                    var_IndexedSlices[var].append(grad)
                else:
                    if var not in var_grad:
                        var_grad[var] = []
                    var_grad[var].append(grad)

        # 返回用来求梯度的gv对
        # 普通var-grads直接求平均
        grad_var = [(tf.reduce_mean(var_grad[var], axis=0), var) for var in var_grad]
        # grad_var = [(var_grad[var][0], var) for var in var_grad]
        # 切片，则把不同GPU得到的切片值、索引，拼接起来，再形成新的切片
        for var in var_IndexedSlices:
            IndexedSlices = var_IndexedSlices[var]  # [IndexedSlices1, IndexedSlices2, ...]
            indices = tf.concat([i.indices for i in IndexedSlices], axis=0)
            values = tf.concat([i.values for i in IndexedSlices], axis=0)
            new_IndexedSlices = tf.IndexedSlices(values, indices)
            grad_var.append((new_IndexedSlices, var))
        return grad_var

    def print_variable(self, with_name=None):
        train_num = 0
        for v in tf.trainable_variables():
            if with_name is None:
                s = self.get_var_size(v)
                train_num += s
                print('-' * 100)
                print(f'{v.name}\t\t{v.shape}\t\t{s}')
            elif with_name in v.name:
                s = self.get_var_size(v)
                train_num += s
                print('-' * 100)
                print(f'{v.name}\t\t{v.shape}\t\t{s}')
        print('-' * 100)
        print(f'trainable parameters: {train_num}')

    def get_var_size(self, var):
        s = 1
        for i in var.shape:
            s *= int(i)
        return s


class SubTensor:
    def __init__(self, normal_init, opt, training, keep_prob, first):
        self._normal_init = normal_init
        self._training = training
        self._keep_prob = keep_prob
        self._first = first

        # 占位符
        self.imgs = tf.placeholder(tf.float32, shape=[None, cfg.cnn_shape, cfg.cnn_shape, 3], name='unet_imgs')
        imgs = self.imgs / 255.

        print('BUILDING: unet...')
        self.re_imgs = unet(imgs, name='unet', filters=cfg.unet_filters, reuse=first, return_atts=False,
                            norm_init=self._normal_init, training=self._training)

        with tf.variable_scope('unet', reuse=first):
            loss = tf.square(self.re_imgs - imgs)
            self.loss = tf.reduce_mean(loss)
            self.grads = opt.compute_gradients(self.loss)


if __name__ == '__main__':
    ts = Tensor()
