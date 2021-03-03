# -*- coding: utf-8 -*-
import tensorflow as tf


class Discriminator_v1:
    def __init__(self, filters, norm_init=None, leaky_slop=0.2):
        self.filters = filters
        self.norm_init = norm_init
        self.leaky_slop = leaky_slop

    def apply(self, x, training=True, **kwargs):
        for ind, f in enumerate(self.filters):
            if ind > 0:
                x = self.instance_norm(x)
            s = 2 if ind < 3 else 1
            x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            x = tf.layers.conv2d(x, f, 4, s, padding='valid', name=f'c4s{s}p1_{ind}',
                                 use_bias=False,
                                 kernel_initializer=self.norm_init)
            # x = tf.layers.batch_normalization(x, name=f'bn{ind}', training=training)
            x = self.instance_norm(x)
            x = tf.nn.leaky_relu(x, self.leaky_slop)

        x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.layers.conv2d(x, 1, 4, 1, padding='valid', name=f'c4s1p1_last', use_bias=False,
                             kernel_initializer=self.norm_init)
        return x

    def instance_norm(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x = (x - mean) / tf.sqrt(var)
        return x


def discriminator(inputs, name='discriminator', filters=None, norm_init=None,
                  reuse=False, training=True, leaky_slop=0.2, **kwargs):
    if filters is None:
        filters = (16, 32, 64, 128)  # 输入是[None, 128, 128, 3]
    with tf.variable_scope(name, reuse=reuse):
        _dis = Discriminator_v1(filters, norm_init, leaky_slop=leaky_slop)
        return _dis.apply(inputs, training=training, **kwargs)
