# -*- coding: utf-8 -*-
import sys, os
import tensorflow as tf

from recognition.arcface.configs import config


# 激活函数
def act(x, act_type, name=None):
    if act_type == 'relu':
        x = tf.nn.relu(x)
    elif act_type == 'leaky_relu':
        x = tf.nn.leaky_relu(x)
    else:
        x = tf.nn.relu(x)
    return x


# 残差块
def resnet_block(inputs, filters, strides, training, shape_match, bottle_neck, **kwargs):
    # 在论文来看，本函数中的x才是残差路径，而shortcut其实是信息流通路径。
    act_type = kwargs.get('act_type', 'leaky_relu')
    if bottle_neck:
        x = tf.layers.batch_normalization(inputs, training=training)
        x = act(x, act_type)
        x = tf.layers.conv2d(x, filters // 4, (1, 1), strides, padding='same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        x = act(x, act_type)
        x = tf.layers.conv2d(x, filters // 4, (3, 3), (1, 1), padding='same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        x = act(x, act_type)
        x = tf.layers.conv2d(x, filters, (1, 1), (1, 1), padding='same', use_bias=False)
    else:
        x = tf.layers.batch_normalization(inputs, training=training)
        x = act(x, act_type)
        x = tf.layers.conv2d(x, filters, (3, 3), strides, padding='same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        x = act(x, act_type)
        x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), padding='same', use_bias=False)

    if shape_match:
        shortcut = inputs
    else:
        shortcut = tf.layers.batch_normalization(inputs, training=training)
        shortcut = act(shortcut, act_type)
        shortcut = tf.layers.conv2d(shortcut, filters, (3, 3), strides, padding='same', use_bias=False)

    return x + shortcut


def get_output(x, training=False, include_top=False):
    num_layers = config.num_layers  # 两处用到config之一
    if num_layers >= 500:
        filters = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filters = [64, 64, 128, 256, 512]
        bottle_neck = False
    if num_layers == 18:
        layers = [3, 4, 6, 3]
    elif num_layers == 50:
        layers = [3, 4, 14, 3]
    elif num_layers == 100:
        layers = [3, 13, 30, 3]
    else:
        raise ValueError(f'ERROR: wrong num_layers: {num_layers}.')

    # 两处用到config之二
    kwargs = {'act_type': config.act_type}

    # 先处理一下
    act_type = kwargs.get('act_type', 'leaky_relu')
    x = tf.layers.conv2d(x, filters[0], (7, 7), (2, 2), padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = act(x, act_type=act_type)

    # 进入各块
    for i, layer in enumerate(layers):
        with tf.variable_scope(f'unit{i}'):
            for j in range(layer):
                with tf.variable_scope(f'block{j}'):
                    strides = (2, 2) if j == 0 else (1, 1)
                    shape_match = False if j == 0 else True
                    x = resnet_block(x, filters[i + 1], strides, training, shape_match,
                                     bottle_neck=bottle_neck, **kwargs)

    if include_top:
        x = tf.layers.batch_normalization(x, training=training)
        x = act(x, act_type)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.layers.dense(x, filters[-1], use_bias=False)
    return x
