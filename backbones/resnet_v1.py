# -*- coding: utf-8 -*-
import tensorflow as tf
from util.activations import activation


def basic_block(inputs, filters=64, strides=(1, 1), training=False, act_type='prelu', **kwargs):
    x = tf.layers.conv2d(inputs, filters, (3, 3), strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = activation(x, act_type, name='act1')  # x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    if inputs.shape == x.shape:
        res = inputs
    else:
        res = tf.layers.conv2d(inputs, filters, (3, 3), strides, padding='same', use_bias=False)
        res = tf.layers.batch_normalization(res, training=training)
    x = activation(x + res, act_type, name='act2')  # x = tf.nn.relu(x)
    return x


def bottle_neck(inputs, filters=64, strides=(1, 1), training=False,
                act_type='leaky_relu', **kwargs):
    x = tf.layers.conv2d(inputs, filters // 4, (1, 1), (1, 1), padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = activation(x, act_type, name='act1')  # x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters // 4, (3, 3), (1, 1), padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = activation(x, act_type, name='act2')  # x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters, (1, 1), strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    if inputs.shape == x.shape:
        res = inputs
    else:
        res = tf.layers.conv2d(inputs, filters, (1, 1), strides, padding='same', use_bias=False)
        res = tf.layers.batch_normalization(res, training=training)

    x = activation(x + res, act_type, name='act3')  # x = tf.nn.relu(x)

    return x


class ResNet_v1:
    def __init__(self, Block=bottle_neck, layers=(3, 4, 6, 3), include_top=True,
                 embedding_size=512, **kwargs):
        self.Block = Block
        self.layers = layers
        self.includ_top = include_top
        self.embedding_size = embedding_size
        self.leaky_slop = kwargs.get('leaky_slop', 0.2)
        self.act_type = kwargs.get('act_type', 'leaky_relu')

    def apply(self, inputs, training=False, reuse=False, filters_base=64):
        x = tf.layers.conv2d(inputs, filters_base, (7, 7), (2, 2), padding='same', use_bias=False)
        x = tf.layers.batch_normalization(x, training=training)
        x = activation(x, self.act_type, name='act1')
        # x = tf.layers.max_pooling2d(x, (3, 3), (2, 2), padding='same')

        for i_block, n_layer in enumerate(self.layers):
            filters = filters_base * 2 ** i_block
            with tf.variable_scope(f'block{i_block}', reuse=reuse):
                for layer in range(n_layer):
                    with tf.variable_scope(f'layer{layer}', reuse=reuse):
                        strides = (2, 2) if layer == 0 else (1, 1)  # if i_block > 0 and layer == 0 else (1, 1)
                        x = self.Block(x, filters=filters, strides=strides, training=training,
                                       act_type=self.act_type)
        if self.includ_top:
            # x = tf.reduce_mean(x, axis=[1, 2])
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, self.embedding_size, use_bias=False, name='resnet_dense')
        return x


def resNet_v1_18(inputs, name, reuse=False, training=False,
                 include_top=True, embedding_size=512, filters_base=64, **kwargs):
    with tf.variable_scope(name, reuse=reuse):
        resnet_v1 = ResNet_v1(Block=basic_block, layers=(2, 2, 2, 2),
                              include_top=include_top, embedding_size=embedding_size, **kwargs)
        return resnet_v1.apply(inputs, training=training, reuse=reuse, filters_base=filters_base)


def resNet_v1_34(inputs, name, reuse=False, training=False,
                 include_top=True, embedding_size=512, filters_base=64, **kwargs):
    with tf.variable_scope(name, reuse=reuse):
        resnet_v1 = ResNet_v1(Block=basic_block, layers=(3, 4, 6, 3),
                              include_top=include_top, embedding_size=embedding_size, **kwargs)
        return resnet_v1.apply(inputs, training=training, reuse=reuse, filters_base=filters_base)


def resNet_v1_50(inputs, name, reuse=False, training=False,
                 include_top=True, embedding_size=512, filters_base=64,
                 norm_init=tf.initializers.truncated_normal(), **kwargs):
    with tf.variable_scope(name, reuse=reuse, initializer=norm_init):
        resnet_v1 = ResNet_v1(Block=bottle_neck, layers=(3, 4, 6, 3), include_top=include_top,
                              embedding_size=embedding_size, **kwargs)
        return resnet_v1.apply(inputs, training=training, reuse=reuse, filters_base=filters_base)


# 可以自定义层数
def resNet_v1_layers(inputs, layers=(3, 4, 6, 3), training=False,
                     include_top=True, embedding_size=512, filters_base=64, act_type='leaky_relu',
                     **kwargs):
    default_scope_kw = {'name_or_scope': 'resNet_v1', 'reuse': False,
                        'initializer': tf.initializers.truncated_normal()}
    scope_kw = kwargs.get('scope_kw', default_scope_kw)
    reuse = scope_kw['reuse']
    with tf.variable_scope(**scope_kw):
        resnet_v1 = ResNet_v1(Block=bottle_neck, layers=layers, include_top=include_top,
                              embedding_size=embedding_size, act_type=act_type, **kwargs)
        return resnet_v1.apply(inputs, training=training, reuse=reuse, filters_base=filters_base)
