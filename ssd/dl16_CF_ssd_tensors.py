"""
检验Tensors:
    构建张量图
"""
# -*- coding: utf-8 -*-
import tensorflow as tf
from ssd.dl16_CF_ssd_config import ssd_cfg
from backbones.dl16_CF_vgg16 import Vgg16
from ssd.dl16_CF_ssd_utils import get_result


class Tensors:
    def __init__(self):
        self.sub_ts = []
        with tf.device('/gpu:0'):
            self.training = tf.placeholder(tf.bool, [], 'training')
            self.lr = tf.placeholder(tf.float32, [], name='lr')
            opt = tf.train.AdamOptimizer(self.lr)
            kernel_initializer = tf.initializers.truncated_normal(ssd_cfg.train_var_mean, ssd_cfg.train_var_std)
            # 全局步数
            self.global_step = tf.get_variable(name='global_step', shape=[],
                                        initializer=tf.initializers.zeros())
            assign_global_step = tf.assign_add(self.global_step, 1)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_global_step)

        print('APPENDING sub_ts...')
        with tf.variable_scope('app'):
            for gpu_index in range(ssd_cfg.train_gpu_num):
                self.sub_ts.append(
                    SubTensors(gpu_index, self.training,
                               opt, kernel_initializer))
                tf.get_variable_scope().reuse_variables()

        with tf.device('/gpu:0'):
            print('MERGING grads...')
            grads = self.merge_grads(lambda ts: ts.grads)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                # 训练
                self.train_op = opt.apply_gradients(grads)

        # 显示的损失
        self.loss_cla = tf.reduce_mean([ts.loss_cla for ts in self.sub_ts])
        self.loss_reg = tf.reduce_mean([ts.loss_reg for ts in self.sub_ts])

        # su_rpn_cla = tf.summary.scalar('loss_cla', self.loss_cla)
        # su_rpn_reg = tf.summary.scalar('loss_reg', self.loss_reg)
        # self.summary_all_op = tf.summary.merge_all()

        self.print_variables()

    def merge_grads(self, f):
        """
        :param f: 提取ts中不同梯度-变量对的函数
        :return:
        """
        # 把各gpu得到的梯度-变量对，整合到统一的{变量-[梯度]}中
        vg = {}
        for ts in self.sub_ts:
            for grad, var in f(ts):
                if grad is not None:
                    if var not in vg:
                        vg[var] = [grad]
                    else:
                        vg[var].append(grad)
        # # vg: {v1: [g11, g12, g13, ...], v2: [g21, g22, g23], ...}
        # 求变量对应的各gpu梯度均值，形成最终的梯度-变量对
        grads = [(tf.reduce_mean(vg[var], axis=0), var) for var in vg]

        return grads

    def print_variables(self):
        vars = tf.trainable_variables()
        total_num = 0
        for var in vars:
            num = 1
            for sh in var.shape:
                num *= sh.value
            print(var.name, '\t', var.shape, '\t', num)
            print('-' * 100)
            total_num += num
        print(f'Total training parameters: {total_num}')


class SubTensors:
    def __init__(self, gpu_index, training, opt, ki):
        self.kernel_initializer = ki
        self.training = training
        # with tf.device(f'/gpu:{gpu_index}'):
        # 输入的样本原图，只有1张
        self.x = tf.placeholder(tf.int32, [1, None, None, 3], 'imgs')
        x = tf.cast(self.x / 255, tf.float32)
        # 原图标注框信息
        self.gt_infos = tf.placeholder(tf.int32, [None, 4], 'gt_infos')
        # 所有特征图的锚框标签
        self.cla_labels = tf.placeholder(tf.float32, [None, ], 'cla_labels')
        self.reg_labels = tf.placeholder(tf.float32, [None, 4], 'reg_labels')

        fms = []  # 用来目标检测的特征图
        vgg_fm, fms = self.get_vgg_fm(x)  # 经过VGG之后的特征图

        x = tf.layers.conv2d(vgg_fm, ssd_cfg.add_conv_filters, 3, 1, padding='same', name='conv6',
                             use_bias=False, kernel_initializer=self.kernel_initializer)
        x = tf.layers.batch_normalization(x, axis=-1, name='bn6', training=self.training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, ssd_cfg.add_conv_filters, 1, 1, padding='same', name='conv7',
                             use_bias=False, kernel_initializer=self.kernel_initializer)
        x = tf.layers.batch_normalization(x, axis=-1, name='bn7', training=self.training)
        x = tf.nn.relu(x)
        fms.append(x)
        for i in range(ssd_cfg.num_add_conv):
            n1 = 4 if i == 0 else 8
            n2 = 2 if i == 0 else 4
            x = tf.layers.conv2d(x, ssd_cfg.add_conv_filters / n1, 1, 1, padding='same', name=f'conv{i + 8}_1x1_1',
                                 use_bias=False, kernel_initializer=self.kernel_initializer)
            x = tf.layers.batch_normalization(x, axis=-1, name=f'bn{i + 8}_1x1_1', training=self.training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, ssd_cfg.add_conv_filters / n1, 3, 1, padding='same', name=f'conv{i + 8}_3x3',
                                 use_bias=False, kernel_initializer=self.kernel_initializer)
            x = tf.layers.batch_normalization(x, axis=-1, name=f'bn{i + 8}_3x3', training=self.training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, ssd_cfg.add_conv_filters / n2, 1, 1, padding='same', name=f'conv{i + 8}_1x1_2',
                                 use_bias=False, kernel_initializer=self.kernel_initializer)
            x = tf.layers.batch_normalization(x, axis=-1, name=f'bn{i + 8}_1x1_2', training=self.training)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x, 2, 2, padding='valid')
            fms.append(x)

        # 计算分类、回归
        cla_logits, cla_probs, regs = self.cla_reg_layer(fms)
        # # [num_anchor_boxes*fms.size], [num_anchor_boxes*fms, 4]

        self.fms = fms
        self.cla_probs = cla_probs
        self.regs = regs

        # 计算损失
        loss_cla, loss_reg = self.get_loss(self.cla_labels, self.reg_labels,
                                           cla_logits, regs, ssd_cfg.train_lambda_reg)
        losses = loss_cla + loss_reg

        # 计算梯度
        self.grads = opt.compute_gradients(losses)

        # 统计显示用的损失。
        self.loss_cla = loss_cla
        self.loss_reg = loss_reg

        # 目标检测的结果
        # id, score
        ob_scores, ob_boxes = tf.py_func(get_result,
                                         [tf.shape(self.x)[1:3], cla_probs, regs],
                                         [tf.double, tf.double])
        self.ob_scores = tf.reshape(tf.cast(ob_scores, tf.float32), [-1])
        self.ob_boxes = tf.reshape(tf.cast(ob_boxes, tf.float32), [-1, 4])

    def get_vgg_fm(self, x):
        # vgg16 = VGG16(include_top=False, input_tensor=xx)
        # # 设置各层都不训练
        # for layer in vgg16.layers:
        #     layer.trainable = False
        # # 提取某一层的结果
        # xx = vgg16.get_layer('block5_conv3').output
        # 以下方法提取的vgg16各卷积权重都是常量，因而根本不会训练。
        with tf.variable_scope('vgg16'):
            vgg16 = Vgg16(ssd_cfg.vgg16_npy_path)
            vgg16.build(x)
        x = vgg16.conv5_3
        fms = [vgg16.conv4_3]  # conv4_3
        return x, fms

    def cla_reg_layer(self, fms):
        cla_logits, regs = None, None
        for i, fm in enumerate(fms):
            with tf.variable_scope(f'cla_reg_fm{i}'):
                channel = fm.shape[-1] if i != 0 else ssd_cfg.add_conv_filters
                fm = tf.layers.conv2d(fm, channel, 3, 1, padding='same', name='c3s1',
                                      use_bias=False, kernel_initializer=self.kernel_initializer)
                fm = tf.layers.batch_normalization(fm, axis=-1, training=self.training, name='bn')
                fm = tf.nn.relu(fm)

                # cla = tf.layers.conv2d(fm, ssd_cfg.num_anchor_box * (ssd_cfg.num_classes + 1), 1, 1,
                #                        padding='same', name='cla', kernel_initializer=self.kernel_initializer)
                # cla = tf.reshape(cla, [-1, ssd_cfg.num_classes + 1])
                cla = tf.layers.conv2d(fm, ssd_cfg.num_anchor_box, 1, 1,
                                       padding='same', name='cla', kernel_initializer=self.kernel_initializer)
                cla = tf.reshape(cla, [-1, 1])                
                cla_logits = tf.concat([cla_logits, cla], axis=0) if cla_logits is not None else cla

                reg = tf.layers.conv2d(fm, ssd_cfg.num_anchor_box * 4, 1, 1,
                                       padding='same', name='regs', kernel_initializer=self.kernel_initializer)
                reg = tf.reshape(reg, [-1, 4])
                regs = tf.concat([regs, reg], axis=0) if regs is not None else reg  
                # # cla_logits和regs每经过一张图就concat一次，应该比较费时间，
                # # 可以考虑add_collection，循环结束后执行一次tf.concat(tf.get_collection())应该就可以了。

        cla_logits = tf.reshape(cla_logits, [-1])  # [total_anchor_boxes]
        cla_probs = tf.nn.sigmoid(cla_logits)

        return cla_logits, cla_probs, regs

    def get_loss(self, cla_label, reg_label, cla_logits, reg, lambda_reg):
        """
        :param cla_label: [None] 锚框的样本标签
        :param reg_label: [sum_anchor_box, 4] 锚框中心到gt框的回归标签
        :param cla_logits: [sum_anchor_box]
        :param reg: [sum_anchor_box, 4]
        :param lambda_reg:
        :return:
        """
        # 选中的样本
        all_inds = tf.reshape(tf.where(tf.not_equal(cla_label, -1)), [-1])
        # 正样本
        pos_inds = tf.reshape(tf.where(cla_label > 0), [-1])

        # 分类损失
        labels = tf.gather(cla_label, all_inds)
        logits = tf.gather(cla_logits, all_inds)
        # loss_cla = tf.nn.sparse_softmax_cross_entropy_with_logits(labels_onehot=labels_onehot, logits=logits)
        loss_cla = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss_cla = tf.reduce_mean(loss_cla)

        # 回归损失只计算正样本的
        # smooth l1 loss1
        reg_label = tf.gather(reg_label, pos_inds)
        regs = tf.gather(reg, pos_inds)
        loss_reg = self.smooth_l1_loss(reg_label, regs, lambda_reg)
        # # mse loss1
        # loss_reg = lambda_reg * tf.reduce_mean((regs[pos_inds] - reg_label[pos_inds]) ** 2)

        return loss_cla, loss_reg

    def smooth_l1_loss(self, reg_label, reg, lambda_reg):
        """
        :param reg_label: [None, 4] 锚框中心到gt框的回归标签
        :param reg: [None, 4]
        :param lambda_reg: 回归损失系数
        :return:
        """
        x = reg - reg_label
        x_abs = tf.abs(x)  # [None, 4]

        # tf.less 返回 True or False； x<y,返回True， 否则返回False。
        # stop_gradient的内容只在前向传播时起作用，反向传播时忽略
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(x_abs, 1)))

        # 实现公式中的条件分支
        loss_reg = 0.5 * (x ** 2) * smoothL1_sign + (x_abs - 0.5) * (1. - smoothL1_sign)
        # # [None, 4]

        loss_reg = lambda_reg * tf.reduce_mean(loss_reg)

        return loss_reg
