# -*- coding: utf-8 -*-
import tensorflow as tf
from unet.unet_config import unet_cfg as cfg


class UNet_v1:
    def __init__(self, filters, norm_init=None):
        self.filters = filters  # 从上往下各层卷积通道数
        self.norm_init = norm_init

    def apply(self, inputs, reuse=False, training=True, return_atts=False, **kwargs):
        xs = self.unet_down(inputs, 'down', reuse, training)
        xs, reimg = self.unet_up(xs, 'up', reuse, training)
        if return_atts:
            return xs, reimg
        else:
            return reimg

    def unet_down(self, x, name, reuse=False, training=True):
        xs = []
        with tf.variable_scope(name, reuse=reuse):
            for ind, f in enumerate(self.filters):
                x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
                x = tf.layers.conv2d(x, f, 4, 2, padding='valid', name=f'c4s2p1_{ind}',
                                     use_bias=False, kernel_initializer=self.norm_init)
                x = tf.layers.batch_normalization(x, name=f'bn{ind}', training=training)
                x = tf.nn.leaky_relu(x, cfg.leaky_slop)
                xs.append(x)
        return xs

    def unet_up(self, _xs, name, reuse=False, training=True):
        xs = []
        with tf.variable_scope(name, reuse=reuse):
            x = _xs[-1]
            xs.append(x)
            for ind, (f, _x) in enumerate(zip(self.filters[:-1][::-1], _xs[:-1][::-1])):
                x = tf.layers.conv2d_transpose(x, f, 4, 2, padding='same', name=f'dc4s2p1_{ind}',
                                               use_bias=False, kernel_initializer=self.norm_init)
                x = tf.layers.batch_normalization(x, name=f'bn{ind}', training=training)
                x = tf.nn.leaky_relu(x, cfg.leaky_slop)
                x = tf.concat([x, _x], axis=3)
                xs.append(x)
            # AAD使用的最后一层属性是上采样得到的图，不牵涉到提取的变量
            xx = tf.image.resize_images(x, [2 * x.shape[1], 2 * x.shape[2]])
            xs.append(xx)
            # unet生成的图像
            reimg = tf.layers.conv2d_transpose(x, 3, 4, 2, padding='same', name=f'reimg',
                                               use_bias=False, kernel_initializer=self.norm_init)
            reimg = tf.nn.sigmoid(reimg)

        return xs, reimg


def unet(inputs, name='unet', filters=None, reuse=False, return_atts=False,
         norm_init=None, training=True, **kwargs):
    if filters is None:
        filters = cfg.unet_filters
    with tf.variable_scope(name, reuse=reuse):
        _unet = UNet_v1(filters, norm_init=norm_init)
        return _unet.apply(inputs, reuse=reuse, training=training, return_atts=return_atts, **kwargs)
