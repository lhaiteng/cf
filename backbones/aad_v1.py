# -*- coding: utf-8 -*-
import tensorflow as tf


class AADGen_v1:
    def __init__(self, filters, norm_init=None):
        self.filters = filters
        self.norm_init = norm_init

    def apply(self, ids, atts, reuse=False, training=True, **kwargs):
        ids = tf.reshape(ids, shape=[-1, 1, 1, ids.shape[1]])
        f = self.filters[0]
        x = tf.layers.conv2d_transpose(ids, f, 2, 1, padding='valid', name='dc2s1p0_0')

        for ind, (f, att) in enumerate(zip(self.filters[1:], atts), 1):
            x = self.aad_resblock(x, att, ids, f, name=f'add_res{ind}', reuse=reuse, training=training)
            # 除最后一层外都将要上采样。
            # ind从1开始记，则最后一层ind=len(atts)
            if ind < len(atts):
                x = tf.image.resize_images(x, [2 * x.shape[1], 2 * x.shape[2]])
        x = tf.nn.sigmoid(x)  # 图像应在0~1之间。
        return x

    def aad_resblock(self, x, att, zid, f, name, reuse=False, training=True):
        # zid [None, 1, 1, id_size]
        with tf.variable_scope(name, reuse=reuse):
            xin = x
            x = self.aad(x, att, zid, name='add0', reuse=reuse, training=True)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, f, 3, 1, padding='same', name=f'c3s1p1_0',
                                 use_bias=False,
                                 kernel_initializer=self.norm_init)
            x = tf.layers.batch_normalization(x, name='bn0', training=training)
            x = self.aad(x, att, zid, name='add1', reuse=reuse, training=True)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, f, 3, 1, padding='same', name=f'c3s1p1_1',
                                 use_bias=False,
                                 kernel_initializer=self.norm_init)
            x = tf.layers.batch_normalization(x, name='bn1', training=training)
            if xin.shape[3] != x.shape[3]:
                xin = self.aad(xin, att, zid, name='add2', reuse=reuse, training=True)
                xin = tf.nn.relu(xin)
                xin = tf.layers.conv2d(xin, f, 3, 1, padding='same', name=f'c3s1p1_2',
                                       use_bias=False,
                                       kernel_initializer=self.norm_init)
                xin = tf.layers.batch_normalization(xin, name='bn2', training=training)
            x = xin + x
        return x

    def aad(self, x, att, zid, name, reuse, training=True):
        # zid [None, 1, 1, id_size]
        with tf.variable_scope(name, reuse=reuse):
            x = self.instance_norm(x)
            xf = x.shape[3]
            M = tf.layers.conv2d(x, 1, 3, 1, padding='same', name='M_c3s1p1',
                                 use_bias=False,
                                 kernel_initializer=self.norm_init)
            M = tf.layers.batch_normalization(M, name='M_bn', training=training)
            M = tf.nn.sigmoid(M)
            att_gama = tf.layers.conv2d(att, xf, 3, 1, padding='same', name='att_gamma',
                                        use_bias=False,
                                        kernel_initializer=self.norm_init)
            att_gama = tf.layers.batch_normalization(att_gama, name='att_gamma_bn', training=training)
            att_gama = tf.nn.relu(att_gama)
            att_beta = tf.layers.conv2d(att, xf, 3, 1, padding='same', name='att_beta',
                                        use_bias=False,
                                        kernel_initializer=self.norm_init)
            att_beta = tf.layers.batch_normalization(att_beta, name='att_beta_bn', training=training)
            att_beta = tf.nn.relu(att_beta)
            A = x * att_gama + att_beta
            zid_gama = tf.layers.conv2d(zid, xf, 1, 1, padding='valid', name='zid_gama',
                                        use_bias=False,
                                        kernel_initializer=self.norm_init)
            zid_gama = tf.layers.batch_normalization(zid_gama, name='zid_gama_bn', training=training)
            zid_gama = tf.nn.relu(zid_gama)
            zid_beta = tf.layers.conv2d(zid, xf, 1, 1, padding='valid', name='zid_beta',
                                        use_bias=False,
                                        kernel_initializer=self.norm_init)
            zid_beta = tf.layers.batch_normalization(zid_beta, name='zid_beta_bn', training=training)
            zid_beta = tf.nn.relu(zid_beta)
            I = x * zid_gama + zid_beta
            x = (1 - M) * A + M * I
        return x

    def instance_norm(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x = (x - mean) / tf.sqrt(var)
        return x


def aad(ids, atts, name='aadgen', filters=None, reuse=False, training=True, norm_init=None, **kwargs):
    if filters is None:
        filters = (256, 256, 256, 256, 128, 64, 32, 3)
    with tf.variable_scope(name, reuse=reuse):
        _aadgen = AADGen_v1(filters, norm_init=norm_init)
        return _aadgen.apply(ids, atts, reuse=reuse, training=training, **kwargs)
