# -*- coding: utf-8 -*-
from recognition.arcface.configs import config
from recognition.arcface.utils import act
import tensorflow as tf


def darknet_block(inputs, filter, training, **kwargs):
    act_type = kwargs.get('act_type', 'leaky_relu')
    x = tf.layers.batch_normalization(inputs, training=training)
    x = act(x, act_type=act_type)
    x = tf.layers.conv2d(x, filter // 2, (1, 1), (1, 1), padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = act(x, act_type=act_type)
    x = tf.layers.conv2d(x, filter, (3, 3), (1, 1), padding='same', use_bias=False)

    return x + inputs


def get_output(x, training=False):
    num_layers = config.num_layers
    base_filter = config.base_filter
    if num_layers == 53:
        layers = (1, 2, 8, 8, 4)
        # filters = (32, 64, 128, 256, 512, 1024)
    elif num_layers == 27:
        layers = (1, 2, 4, 4, 2)
    else:
        raise ValueError(f'ERROR: void num_layers {num_layers}.')

    filters = [base_filter * 2 ** i for i in range(6)]  # [16, 32, 64, 128, 256, 512]

    # 两处用到config之二
    kwargs = {'act_type': config.act_type}

    # 先处理一下
    act_type = kwargs.get('act_type', 'leaky_relu')
    x = tf.layers.conv2d(x, filters[0], (3, 3), (1, 1), padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = act(x, act_type=act_type)

    for i, layer in enumerate(layers):
        with tf.variable_scope(f'unit{i}'):
            x = tf.layers.conv2d(x, filters[i + 1], (3, 3), (2, 2), padding='same', use_bias=False)
            for j in range(layer):
                with tf.variable_scope(f'block{j}'):
                    x = darknet_block(x, filters[i + 1], training, **kwargs)

    x = tf.layers.batch_normalization(x, training=training)
    x = act(x, act_type=act_type)
    x = tf.reduce_mean(x, [1, 2])

    return x
