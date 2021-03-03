# -*- coding: utf-8 -*-
import math
from arcface.arcface_config import arcface_cfg as cfg
import tensorflow as tf
import numpy as np
from backbones.resnet_v1 import resNet_v1_layers
from losses.rec_loss import get_arcface_loss, get_center_loss
from util.arcface_utils import get_arcface_result
from util.activations import activation


class ArcFace:
    def __init__(self):
        global_vars_save, global_vars_restore = set(), set()
        # 占位符
        self.lr, self.keep_prob, self.training, self.para_arcface, self.should_com_center_loss = self.get_net_placeholder()
        # 优化器
        opt = tf.train.AdamOptimizer(self.lr)
        # 初始化器
        normal_init = tf.initializers.truncated_normal(cfg.init_mean, cfg.init_std)
        leaky_slop = cfg.leaky_slop
        # 计步器。更新一次权重是一步
        self.global_step = tf.get_variable('global_step', shape=[],
                                           initializer=tf.constant_initializer(0), trainable=False)
        global_vars_save.add(self.global_step)
        global_vars_restore.add(self.global_step)
        # 分类网络的最终分类权重
        cls_weight = tf.get_variable(name='unnormed_arcface_weight',
                                     shape=[cfg.id_size, cfg.num_cls], dtype=tf.float32,
                                     initializer=tf.initializers.random_normal())
        global_vars_save.add(cls_weight)
        global_vars_restore.add(cls_weight)
        self.cls_weight = tf.nn.l2_normalize(cls_weight, axis=0, name='normed_arcface_weight')
        # 分类权重的两两相似度矩阵
        self.sim_matrix_cls_weight = tf.matmul(self.cls_weight, self.cls_weight, transpose_a=True)
        # 向量中心点
        self.prenorm_centers = tf.get_variable(name='prelogit_center',
                                               shape=[cfg.num_cls, cfg.id_size], dtype=tf.float32,
                                               initializer=tf.constant_initializer(0), trainable=False)
        global_vars_save.add(self.prenorm_centers)
        global_vars_restore.add(self.prenorm_centers)
        # 向量中心点的两两相似度矩阵
        normed_centers = tf.nn.l2_normalize(self.prenorm_centers, axis=1)  # [cfg.num_cls, cfg.id_size]
        self.sim_matrix_centers = tf.matmul(normed_centers, normed_centers, transpose_b=True)
        # 向量中心点与cle_weight的两两相似度矩阵
        self.sim_matrix_w_centers = tf.matmul(normed_centers, self.cls_weight)

        # 占位符
        # 输入图片
        self.inputs = tf.placeholder(tf.float32, shape=[None, cfg.cnn_shape, cfg.cnn_shape, 3],
                                     name='arcface_inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None, ], name='arcface_labels')
        labels_onehot = tf.one_hot(self.labels, cfg.num_cls)

        # 使用滑动均值
        if cfg.move_imgs_decay > 0:
            moving_mean_imgs = tf.get_variable('moving_mean_imgs', shape=[3], dtype=tf.float32,
                                               initializer=tf.initializers.zeros(), trainable=False)
            global_vars_save.add(moving_mean_imgs)
            global_vars_restore.add(moving_mean_imgs)
            tf.summary.histogram('moving_mean_imgs', moving_mean_imgs)
            x = self._get_moving_imgs(moving_mean_imgs)
        else:
            # x = self.inputs / 255.
            x = self.inputs - 127.5  # [-127.5, 127.5]
            x = x * 0.0078125  # [-0.99609375, 0.99609375]

        # 也许是因为想求L1 loss，所以先做出了未L2正则化的id?
        # 论文作者代码求center loss使用的也是从ResNet中求出的未L2正则化的id
        print('START: build arcface...')
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.l2_loss_factor) if cfg.l2_loss_factor > 0 else None
        _scope_kw = {'name_or_scope': 'arcface', 'reuse': False,
                     'initializer': normal_init, 'regularizer': weights_regularizer}
        x = resNet_v1_layers(x, scope_kw=_scope_kw, training=self.training, include_top=True,
                             embedding_size=cfg.id_size, filters_base=cfg.resnet_filters_base,
                             leaky_slop=leaky_slop, act_type='leaky_relu')

        # 论文对prenormed_id使用bn+dropout+fc+bn
        _scope_kw = {'name_or_scope': 'arcface_addition_layers', 'reuse': False,
                     'initializer': normal_init, 'regularizer': weights_regularizer}
        with tf.variable_scope(**_scope_kw):
            x = tf.layers.batch_normalization(x, training=self.training)
            # x = activation(x, 'leaky_relu', name='act1')
            x = tf.layers.dropout(x, rate=self.keep_prob)
            x = tf.layers.dense(x, cfg.id_size, use_bias=False)
            prenormed_id = tf.layers.batch_normalization(x, training=self.training)
        self.arcface_id = tf.nn.l2_normalize(prenormed_id, axis=1)  # [None, id_size]

        # logits = tf.matmul(prenormed_id, self.cls_weight)  # [None, num_cls]
        # norm_logits = tf.norm(prenormed_id, axis=1, keepdims=True)
        # self.cosines = logits / norm_logits  # 与各类别的夹角余弦
        cosines = tf.matmul(self.arcface_id, self.cls_weight)  # [None, num_cls]
        thetas = tf.acos(cosines)  # 与各类别的夹角弧度 [None, num_cls]
        self.thetas_angle = thetas * 180 / math.pi  # 夹角角度 [None, num_cls]
        # 本批次的图片之间的相似度矩阵
        self.sim_matrix = tf.matmul(self.arcface_id, self.arcface_id, transpose_b=True)

        # 前向计算的分类结果。以夹角角度计算
        self.acc, self.theta_label, self.theta_unlabel = \
            get_arcface_result(self.thetas_angle, self.labels, labels_onehot, cfg.num_cls)

        # 计算损失
        self.loss = self.get_loss(cls_weight, prenormed_id, thetas, labels_onehot)

        # 计算梯度更新的操作
        train_op = opt.minimize(self.loss, global_step=self.global_step)

        # # 滑动平均
        # ema = tf.train.ExponentialMovingAverage(cfg.ema_decay)
        # ema_op = ema.apply(tf.trainable_variables())
        # # 把shadow_var加入变量列表中
        # if cfg.ema_decay > 0:
        #     ema_var_list = ema.variables_to_restore(tf.trainable_variables)
        #     global_vars_save |= ema_var_list
        #     global_vars_restore |= ema_var_list

        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign_add(self.global_step, 1))
        # print(f'UPDATE_OPS\n{tf.get_collection(tf.GraphKeys.UPDATE_OPS)}')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 先BN和全局步数更新
            with tf.control_dependencies([train_op]):  # 再做模型更新
                # if cfg.ema_decay > 0:  # 最后使用滑动平均进行更新参数
                #     with tf.control_dependencies([ema_op]):
                #         self.train_arcface = tf.no_op()
                # else:
                self.train_arcface = tf.no_op()

        self.print_trainable_variables()

        # 用来储存的var_list
        vars_save = {v for v in tf.trainable_variables()}
        vars_save_moving = {v for v in tf.global_variables()
                            if 'arcface' in v.name and 'moving' in v.name}
        self.vars_save = list(global_vars_save | vars_save | vars_save_moving)
        # print(f'VARS to save:\n{self.vars_save}')

        # 用来恢复的var_list
        vars_restore = {v for v in tf.trainable_variables()}
        vars_restore_moving = {v for v in tf.global_variables()
                               if 'arcface' in v.name and 'moving' in v.name}
        self.vars_restore = list(global_vars_restore | vars_restore | vars_restore_moving)
        # print(f'VARS to save:\n{self.vars_restore}')

        # 需要tensorboard观察的
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('theta_label', self.theta_label)
        tf.summary.scalar('theta_unlabel', self.theta_unlabel)
        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('cls_weight_unnormed', cls_weight)
        tf.summary.histogram('cls_weight_normed', self.cls_weight)
        tf.summary.histogram('prenormed_id', prenormed_id)
        tf.summary.histogram('arcface_id', self.arcface_id)
        if cfg.center_loss_factor is not None:
            tf.summary.histogram('prenorm_centers', self.prenorm_centers)
        self.merged = tf.summary.merge_all()

    def _get_moving_imgs(self, moving_mean_imgs):
        def _update_move():
            new_mean = tf.reduce_mean(self.inputs, axis=[0, 1, 2])
            decay = cfg.move_imgs_decay
            assign_op = tf.assign(moving_mean_imgs, decay * moving_mean_imgs + (1 - decay) * new_mean)
            with tf.control_dependencies([assign_op]):
                return self.inputs - moving_mean_imgs

        def _no_update_move():
            return self.inputs - moving_mean_imgs

        x = tf.cond(self.training, _update_move, _no_update_move)
        return x

    def get_loss(self, cls_weight, prenormed_id, thetas, labels_onehot):
        # l1 loss1
        if cfg.l1_loss_factor_id > 0:
            print('creat l1 loss of prenormed_id...')
            l1_loss1 = tf.reduce_mean(tf.norm(prenormed_id, ord=1, axis=1))
            l1_loss1 = cfg.l1_loss_factor_id * l1_loss1
            tf.losses.add_loss(l1_loss1)
            tf.summary.scalar('loss_l1_prenormed_id', l1_loss1)
        if cfg.l1_loss_factor_w > 0:
            print('creat l1 loss of cls_weight...')
            l1_loss2 = tf.reduce_mean(tf.norm(cls_weight, ord=1, axis=0))
            l1_loss2 = cfg.l1_loss_factor_w * l1_loss2
            tf.losses.add_loss(l1_loss2)
            tf.summary.scalar('loss_l1_cls_weight', l1_loss2)
        # center loss1  函数中自动更新了centers变量
        if cfg.center_loss_factor > 0:
            print('creat center loss of prenormed_id...')
            center_loss, self.prenorm_centers = get_center_loss(prenormed_id, self.labels,
                                                                self.prenorm_centers, cfg.center_alpha,
                                                                self.should_com_center_loss)
            center_loss = cfg.center_loss_factor * center_loss
            tf.losses.add_loss(center_loss)
            tf.summary.scalar('loss_center_prenormed_id', center_loss)
        # arcface loss1
        print('creat arcface loss...')
        s, m1, m2, m3 = self.para_arcface[0], self.para_arcface[1], self.para_arcface[2], self.para_arcface[3]
        arc_loss = get_arcface_loss(thetas, labels_onehot, m1, m2, m3, s,
                                    alpha=cfg.fl_alpha, gamma=cfg.fl_gamma)
        tf.losses.add_loss(arc_loss)
        tf.summary.scalar('loss_arcface', arc_loss)

        # 权重的l2正则化损失
        if cfg.l2_loss_factor > 0:
            print('collect l2 loss of weights...')
            # l2_loss_list = tf.losses.get_regularization_losses()  # 获取计算正则损失的张量列表
            # print(f'l2 loss list:\n{l2_loss_list}')
            l2_loss = tf.losses.get_regularization_loss()  # 获取损失值
            tf.losses.add_loss(l2_loss)
            tf.summary.scalar('l2_loss', l2_loss)

        # 整体loss
        loss = tf.add_n(tf.losses.get_losses())

        return loss

    def get_net_placeholder(self):
        lr = tf.placeholder(tf.float32, shape=[], name='lr')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        training = tf.placeholder(tf.bool, shape=[], name='training')
        # 保存s ,m1, m2, m3
        para_arcface = tf.placeholder(tf.float32, shape=[4], name='para_arcface')
        # 计算center loss1
        should_com_center_loss = tf.placeholder(tf.bool, shape=[], name='should_com_center_loss')
        return lr, keep_prob, training, para_arcface, should_com_center_loss

    def get_cls_result(self, cls_logits, labels, cls_theta):
        # 预测结果
        cls_pred = tf.argmax(cls_logits, axis=1, output_type=tf.int32)
        # 预测准确率
        cls_acc = tf.reduce_mean(tf.cast(tf.equal(cls_pred, self.labels), tf.float32))
        # 标签项的平均夹角
        cls_label_theta = tf.reduce_mean(tf.reduce_sum(tf.multiply(cls_theta, labels), axis=1))
        # 非标签的平均夹角
        cls_unlabel_theta = tf.reduce_sum(tf.multiply(cls_theta, 1 - labels), axis=1)
        cls_unlabel_theta = cls_unlabel_theta / tf.cast(cls_theta.shape[1] - 1, tf.float32)
        cls_unlabel_theta = tf.reduce_mean(cls_unlabel_theta)

        return cls_acc, cls_label_theta, cls_unlabel_theta

    def print_trainable_variables(self):
        train_num = 0
        for v in tf.trainable_variables():
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


if __name__ == '__main__':
    ts = ArcFace()
#
#     datas = np.random.randint(0, 255, (100 * arcface_cfg.train_batch_size, 224, 224, 3))
#     labels_onehot = np.random.randint(0, arcface_cfg.num_cls, (100 * arcface_cfg.train_batch_size,))
#
#     sub_ts = ts.sub_ts[0]
#
#     with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(1000):
#             j = i % arcface_cfg.train_batch_size
#             x = datas[j * arcface_cfg.train_batch_size:(j + 1) * arcface_cfg.train_batch_size]
#             label = labels_onehot[j * arcface_cfg.train_batch_size:(j + 1) * arcface_cfg.train_batch_size]
#             feed_dict = {ts.lr: arcface_cfg.train_lr, ts.keep_prob: arcface_cfg.train_keep_prob, ts.training: True,
#                          sub_ts.inputs: x, sub_ts.labels_onehot: label}
#
#             _, lo = sess.run([ts.train_op1, ts.loss1], feed_dict)
#             print(lo)
