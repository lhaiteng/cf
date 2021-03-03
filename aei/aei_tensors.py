# -*- coding: utf-8 -*-
from aei.aei_config import aei_cfg as cfg
from arcface.arcface_config import arcface_cfg
from backbones.resnet_v1 import resNet_v1_50
from backbones.unet_v1 import unet
from backbones.aad_v1 import aad
from backbones.discriminator_v1 import discriminator
import tensorflow as tf
import numpy as np


class AEI:
    def __init__(self):
        with tf.device('/gpu:0'):
            # 占位符
            self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
            self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
            self.training = tf.placeholder(tf.bool, shape=[], name='training')
            # 初始化器
            normal_init = tf.initializers.truncated_normal(cfg.aei_init_mean, cfg.aei_init_std)
            # 优化器
            opt = tf.train.AdamOptimizer(self.lr)  # , beta1=cfg.aei_beta1, beta2=cfg.aei_beta2)

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
                grads_d = self.merge_grads(lambda ts: ts.grads_d)
                grads_g = self.merge_grads(lambda ts: ts.grads_g)
                # # [(grad, var), (grad, avr), ...]

            """在tf中进行多批次平均优化"""
            # 更新一次需要几个批次
            self.train_num_batch = tf.placeholder(tf.float32, shape=[], name='train_num_batch')

            # 先得到相同形状的grads以待求和
            # 对d
            # 必须是变量形式，否则不能assign。不可trainable
            self.grads_d = {gv[1]: tf.get_variable(name=f'train_grads_d{ind}', shape=gv[0].shape,
                                                   initializer=tf.initializers.zeros(), trainable=False)
                            for ind, gv in enumerate(grads_d)}
            # 分别对每个梯度进行初始化。对应的变量v应该只是个指针不是个值，所以不用再次指向新的变量值
            self.assign_zero_grads_d = tf.initialize_variables([g for v, g in self.grads_d.items()])
            # 赋值op的列表, 分别将梯度累加进去
            self.assign_grads_d = [tf.assign_add(self.grads_d[v], g)
                                   for g, v in grads_d]
            # 对g
            self.grads_g = {gv[1]: tf.get_variable(name=f'train_grads_g{ind}', shape=gv[0].shape,
                                                   initializer=tf.initializers.zeros(), trainable=False)
                            for ind, gv in enumerate(grads_g)}
            # 分别对每个梯度进行初始化。对应的变量v应该只是个指针不是个值，所以不用再次指向新的变量值
            self.assign_zero_grads_g = tf.initialize_variables([g for v, g in self.grads_g.items()])
            # 赋值op的列表, 分别将梯度累加进去
            self.assign_grads_g = [tf.assign_add(self.grads_g[v], g)
                                   for g, v in grads_g]

            self.train_d = opt.apply_gradients([(g / self.train_num_batch, v)
                                                for v, g in self.grads_d.items()])
            self.train_g = opt.apply_gradients([(g / self.train_num_batch, v)
                                                for v, g in self.grads_g.items()])

            self.print_variable(with_name='aei')

            print('Reduce_meaning loss1...')
            self.loss_adv_d = tf.reduce_mean([_ts.loss_adv_d for _ts in self.sub_ts])
            self.loss_adv_g = tf.reduce_mean([_ts.loss_adv_g for _ts in self.sub_ts])
            self.loss_att = tf.reduce_mean([_ts.loss_att for _ts in self.sub_ts])
            self.loss_id = tf.reduce_mean([_ts.loss_id for _ts in self.sub_ts])
            self.loss_rec = tf.reduce_mean([_ts.loss_rec for _ts in self.sub_ts])

            # 用来恢复的var_list
            vars_restore = [v for v in tf.trainable_variables()
                            if 'aei' in v.name]
            vars_restore_moving = [v for v in tf.global_variables()
                                   if 'aei' in v.name and 'moving' in v.name]
            self.vars_restore = [self.global_step] + vars_restore + vars_restore_moving

            # 用来储存的var_list
            vars_save = [v for v in tf.trainable_variables()
                         if 'aei' in v.name]
            vars_save_moving = [v for v in tf.global_variables()
                                if 'aei' in v.name and 'moving' in v.name]
            self.vars_save = [self.global_step] + vars_save + vars_save_moving

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
        with tf.variable_scope('aei', reuse=first):  # 这句话很关键！
            self._normal_init = normal_init
            self._training = training
            self._keep_prob = keep_prob
            self._first = first

            # 占位符
            # 输入图片
            self.src = tf.placeholder(tf.float32, shape=[None, cfg.cnn_shape, cfg.cnn_shape, 3], name='source')
            self.target = tf.placeholder(tf.float32, shape=[None, cfg.cnn_shape, cfg.cnn_shape, 3], name='target')
            self.is_rec = tf.placeholder(tf.float32, shape=[None, ], name='is_rec')

            src = self.src / 255.
            target = self.target / 255.

        print('COMPUTE src_id...')
        id_src = resNet_v1_50(self.src, name='arcface', reuse=first, norm_init=self._normal_init,
                              training=False, include_top=True, embedding_size=arcface_cfg.id_size,
                              filters_base=32)
        id_src = tf.nn.l2_normalize(id_src, axis=1)
        print('COMPUTE target_atts...')
        atts_tar, _ = unet(target, name='unet', reuse=first, return_atts=True,
                           norm_init=self._normal_init, training=False)  # 底部以及向上合并后的向量，作为图片的属性

        with tf.variable_scope('aei', reuse=first):
            print('BUILD aad generator...')
            self.st = aad(id_src, atts_tar, name='aadgen', reuse=first,
                          training=self._training, norm_init=self._normal_init)
            print('BUILD discriminator...')
            dis_src = discriminator(src, name='discriminator', filters=cfg.aei_discriminater_filters,
                                    norm_init=self._normal_init, reuse=first, training=self._training,
                                    leaky_slop=cfg.aei_leaky_slop)
            dis_target = discriminator(target, name='discriminator', filters=cfg.aei_discriminater_filters,
                                       norm_init=self._normal_init, reuse=True, training=self._training,
                                       leaky_slop=cfg.aei_leaky_slop)
            dis_st = discriminator(self.st, name='discriminator', filters=cfg.aei_discriminater_filters,
                                   norm_init=self._normal_init, reuse=True, training=self._training,
                                   leaky_slop=cfg.aei_leaky_slop)

        print('COMPUTE st_id...')
        id_st = resNet_v1_50(self.st, name='arcface', reuse=True, norm_init=self._normal_init,
                             training=False, include_top=True, embedding_size=arcface_cfg.id_size,
                             filters_base=32)
        id_st = tf.nn.l2_normalize(id_st, axis=1)
        print('COMPUTE st_atts...')
        atts_st, _ = unet(self.st, name='unet', reuse=True, return_atts=True,
                          norm_init=self._normal_init, training=False)  # 底部以及向上合并后的向量，作为图片的属性

        print('COMPUTE loss1 atts...')
        loss_att = self.get_loss_att(atts_st, atts_tar)
        self.loss_att = cfg.aei_lambda_att * loss_att
        print('COMPUTE loss1 adv...')
        self.loss_adv_d, self.loss_adv_g = self.get_loss_adv(dis_src, dis_target, dis_st)
        print('COMPUTE loss1 id...')
        loss_id = self.get_loss_id(id_st, id_src)
        self.loss_id = cfg.aei_lambda_id * loss_id
        print('COMPUTE loss1 rec...')
        loss_rec = self.get_loss_rec(self.st, target, self.is_rec)
        self.loss_rec = cfg.aei_lambda_rec * loss_rec

        loss_d = self.loss_adv_d
        loss_g = self.loss_adv_g + self.loss_att + self.loss_id + self.loss_rec

        var_d = [v for v in tf.trainable_variables()
                 if 'aei' in v.name and 'discriminator' in v.name]
        var_g = [v for v in tf.trainable_variables()
                 if 'aei' in v.name and 'aadgen' in v.name]
        print('COMPUTE grads d...')
        self.grads_d = opt.compute_gradients(loss_d, var_list=var_d)
        print('COMPUTE grads g...')
        self.grads_g = opt.compute_gradients(loss_g, var_list=var_g)

    def get_loss_adv(self, dis_src, dis_target, dis_st, cate='wgan_gp'):
        if cate == 'square':
            return self.get_loss_adv_square(dis_src, dis_target, dis_st)
        elif cate == 'wgan_gp':
            # 求中间图片
            alpha = tf.random_uniform(shape=[tf.shape(self.src)[0]], minval=0., maxval=1.)
            alpha = tf.reshape(alpha, [-1, 1, 1, 1])
            interpolates = alpha * (self.src + self.target) / 510. + (1 - alpha) * self.st
            with tf.variable_scope('aei', reuse=True):
                gen_interpolates = discriminator(interpolates, name='discriminator',
                                                 filters=cfg.aei_discriminater_filters,
                                                 norm_init=self._normal_init, reuse=True,
                                                 training=self._training,
                                                 leaky_slop=cfg.aei_leaky_slop)
            # 梯度
            gradients = tf.gradients(gen_interpolates, [interpolates])  # 应当是[四维w, 四维w, ...]啊
            # # [<tf.Tensor 'gradients/aei_2/discriminator/Pad_grad/Slice_1:0' shape=(?, 128, 128, 3) dtype=float32>]
            # 梯度的二范数
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[2, 3, 4]))
            # 惩罚梯度
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            loss_g = -tf.reduce_mean(dis_st)
            loss_d = -tf.reduce_mean(dis_src) - tf.reduce_mean(dis_target) - loss_g \
                     + cfg.aei_lambda_wgan_gp * gradient_penalty

            return loss_d, loss_g

    def get_loss_adv_square(self, dis_src, dis_target, dis_st):
        # 对于d
        loss_d = None
        for i, j in zip((dis_src, dis_target, dis_st), (1, 1, 0)):
            if loss_d is None:
                loss_d = tf.reduce_mean(tf.square(i - j))
            else:
                loss_d += tf.reduce_mean(tf.square(i - j))
        loss_d = tf.reduce_mean(loss_d)
        loss_g = tf.reduce_mean(tf.square(dis_st - 1))
        return loss_d, loss_g

    def get_loss_att(self, atts_st, atts_tar):
        loss_att = None
        for st, t in zip(atts_st, atts_tar):
            if loss_att is None:
                loss_att = tf.reduce_mean(tf.square(st - t))
            else:
                loss_att += tf.reduce_mean(tf.square(st - t))
        loss_att = tf.reduce_mean(loss_att)
        return loss_att

    def get_loss_id(self, id_st, id_src):
        # # [None, id_size]
        # st = tf.nn.l2_normalize(id_st, axis=1)
        # src = tf.nn.l2_normalize(id_src, axis=1)
        # # 默认得到的id本来就是l2归一化后的
        cos = tf.reduce_sum(tf.multiply(id_st, id_src), axis=1)
        loss_id = tf.reduce_mean(1 - cos)
        return loss_id

    def get_loss_rec(self, st, target, is_rec):
        num_rec = tf.reduce_sum(is_rec) + cfg.epsilon
        loss_rec = tf.reduce_mean(tf.square(st - target), axis=[1, 2, 3])
        is_rec = tf.reshape(is_rec, [-1, 1])
        loss_rec = tf.reduce_sum(is_rec * loss_rec) / num_rec
        return loss_rec

# if __name__ == '__main__':
#     ts = AEI()
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
