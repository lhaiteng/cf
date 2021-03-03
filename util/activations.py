# -*- coding: utf-8 -*-
"""
自定义的激活函数
"""
import tensorflow as tf


def prelu(x, name, shape=None, initializer=None, **kwargs):
    if shape is None: shape = x.shape[-1]
    alpha = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def activation(x, act_type, name=None, **kwargs):
    if act_type == 'sigmoid':
        return tf.nn.sigmoid(x, name)
    elif act_type == 'tanh':
        return tf.nn.tanh(x, name)
    elif act_type == 'relu':
        return tf.nn.relu(x, name)
    elif act_type == 'leaky_relu':
        leaky_slop = kwargs.get('leaky_slop', 0.2)
        return tf.nn.leaky_relu(x, leaky_slop, name)
    elif act_type == 'prelu':  # 函数似乎有问题
        kw = {k[6:]: v for k, v in kwargs if 'prelu' in k}
        return prelu(x, name=name, **kw)
    elif act_type == 'elu':
        return tf.nn.elu(x, name=name)
    elif act_type == 'crelu':
        axis = kwargs.get('crelu_axis', -1)
        return tf.nn.crelu(x, name=name, axis=axis)
    else:
        raise ValueError(f'act type {act_type} cannot be found.')
