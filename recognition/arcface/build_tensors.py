"""
参考的sim.arg_scope设置
def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=2e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.leaky_relu,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': None,  # Use fused batch norm if possible.
      'param_regularizers': {'gamma': slim.l2_regularizer(weight_decay)},
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm if use_batch_norm else None,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


arg_sc = resnet_arg_scope(weight_decay=5e-4, batch_norm_decay=0.9)

调用：
with slim.arg_scope(arg_sc):

"""

import sys, os, math
import tensorflow as tf
from recognition.arcface.configs import config
from recognition.arcface.utils import act
from recognition.backbones import resnet, darknet


# 得到变量的初始化器、正则化器
def get_variable_setting():
    # 初始化器
    initializer = config.get('initializer', None)
    if initializer == 'truncated_normal':
        initializer = tf.initializers.truncated_normal(config.init_mean, config.init_std)
    elif initializer == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    else:
        initializer = eval(f'tf.initializers.{initializer}')()

    # 正则化器
    regularizer = config.get('regularizer', None)
    if regularizer == 'l2_regularizer':
        regularizer = tf.contrib.layers.l2_regularizer(config.regular_factor)
    else:
        regularizer = None

    return initializer, regularizer


def _get_emb_resnet(inputs, training, keep_prob):
    x = resnet.get_output(inputs, training=training, include_top=False)
    # 附加网络 bn+dropout+fc+bn
    with tf.variable_scope('addition_layers'):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.layers.dropout(x, keep_prob, training=training)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, config.embedding_size, use_bias=False)
        emb = tf.layers.batch_normalization(x, training=training)
    return emb


def _get_emb_darknet(inputs, training, keep_prob):
    x = darknet.get_output(inputs, training=training)

    # 附加层
    with tf.variable_scope('addition_layers'):
        if config.addition_layer_type == 1:
            x = tf.layers.dropout(x, keep_prob, training=training)
            x = tf.layers.dense(x, config.embedding_size, use_bias=False)
            emb = tf.layers.batch_normalization(x, training=training)
        elif config.addition_layer_type == 2:
            x = tf.layers.dense(x, config.embedding_size, use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = act(x, 'leaky_relu')
            emb = tf.layers.dense(x, config.embedding_size, use_bias=False)
        elif config.addition_layer_type == 3:
            x = tf.layers.dropout(x, keep_prob, training=training)
            x = tf.layers.dense(x, config.embedding_size, use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = act(x, 'leaky_relu')
            emb = tf.layers.dense(x, config.embedding_size, use_bias=False)
        elif config.addition_layer_type == 4:
            for _ in range(2):
                x = tf.layers.dropout(x, keep_prob, training=training)
                x = tf.layers.dense(x, config.embedding_size, use_bias=False)
                x = tf.layers.batch_normalization(x, training=training)
                x = act(x, 'leaky_relu')
            emb = tf.layers.dense(x, config.embedding_size, use_bias=False)
        else:
            raise ValueError(f'ERROR: cannot find addition_type <{config.addition_type}>.')

    return emb


# 得到语义向量
def get_emb(inputs, name, reuse, training=False, keep_prob=1, summary='', get_unl2=False):
    # 得到变量的初始化器、正则化器
    initializer, regularizer = get_variable_setting()

    with tf.variable_scope(name, reuse=reuse, initializer=initializer, regularizer=regularizer):
        # 骨干网络
        with tf.variable_scope(config.net_name):
            if config.net_name == 'resnet':
                emb = _get_emb_resnet(inputs, training, keep_prob)
            elif config.net_name == 'darknet':
                emb = _get_emb_darknet(inputs, training, keep_prob)
            else:
                raise ValueError(f'ERROR: cannot find net <{config.net_name}>.')

        # 人脸分类权重
        cls_weight = tf.get_variable('cls_weight', [config.embedding_size, config.num_cls], tf.float32,
                                     initializer=tf.initializers.random_normal(mean=0, stddev=0.01))
        # l2规范化
        l2_emb = tf.nn.l2_normalize(emb, axis=1)
        l2_cls_weight = tf.nn.l2_normalize(cls_weight, axis=0)

    if summary:
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.histogram(f'embedding/{summary}_unormed', emb))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.histogram(f'embedding/{summary}_normed', l2_emb))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.histogram(f'cls_weight/{summary}_unormed', cls_weight))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.histogram(f'cls_weight/{summary}_normed', l2_cls_weight))

    if get_unl2:
        return l2_emb, l2_cls_weight, emb
    else:
        return l2_emb, l2_cls_weight


# 得到损失
def get_loss(cosines, radians, labels_onehot, summary=''):
    # arcface loss
    s, m1, m2, m3 = config.loss_s, config.loss_m1, config.loss_m2, config.loss_m3
    zeros = tf.zeros_like(labels_onehot)

    if config.arcface_loss_type == 0:  # 只有<pi的才执行变化
        # 标签项中，需要变化的
        changed_mask = tf.where(tf.less(m1 * radians + m2, math.pi), labels_onehot, zeros)
        # 新的余弦矩阵
        cosines = (tf.cos(m1 * radians + m2) - m3) * changed_mask + cosines * (1. - changed_mask)
    elif config.arcface_loss_type == 1:  # 只有<pi/2的才执行变化
        # 标签项中，需要变化的
        changed_mask = tf.where(tf.less(m1 * radians + m2, math.pi / 2), labels_onehot, zeros)
        # 新的余弦矩阵
        cosines = (tf.cos(m1 * radians + m2) - m3) * changed_mask + cosines * (1. - changed_mask)
    elif config.arcface_loss_type == 2:  # 标签项都执行cosine-m3
        # 标签项中，需要变化的
        changed_mask = tf.where(tf.less(m1 * radians + m2, math.pi / 2), labels_onehot, zeros)
        # 改变后的弧度
        radians = (m1 * radians + m2) * changed_mask + radians * (1. - changed_mask)
        # 改变后的余弦
        cosines = (tf.cos(radians) - m3) * labels_onehot + cosines * (1. - labels_onehot)
    else:
        raise ValueError(f'ERROR: cannot find arcface_loss_type <{config.arcface_loss_type}>.')

    logits = s * cosines
    loss_arcface = tf.nn.softmax_cross_entropy_with_logits_v2(labels_onehot, logits)
    p_all = tf.nn.softmax(logits, axis=1)
    p_label = tf.reduce_sum(p_all * labels_onehot, axis=1)
    if config.focal_loss_alpha > 0 and config.focal_loss_gamma > 0:
        alpha, gamma = config.focal_loss_alpha, config.focal_loss_gamma
        loss_arcface = alpha * (1. - p_label) ** gamma * loss_arcface
    loss_arcface = tf.reduce_mean(loss_arcface)

    # l2正则化损失
    # # 在定义运算符时定义regularizer
    # loss_l2 = tf.losses.get_regularization_loss()
    # # tf.losses.add_loss(loss_l2)  # tf.losses.get_total_loss() 可以获取正则损失
    # 手动算
    loss_l2 = 0
    if config.regular_factor > 0:
        for v in tf.trainable_variables():
            if 'beta' in v.name or 'gamma' in v.name:
                continue
            loss_l2 += tf.reduce_sum(tf.square(v))
        loss_l2 = config.regular_factor * loss_l2

    loss_total = loss_arcface + loss_l2

    # 写summary
    if summary:
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.histogram(f'probability/{summary}_distribution_all', p_all))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.histogram(f'probability/{summary}_distribution_label', p_label))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.scalar(f'probability/{summary}_mean_label', tf.reduce_mean(p_label)))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.scalar(f'loss/{summary}_arcface', loss_arcface))
        if config.regular_factor > 0: tf.add_to_collection(f'{summary}_summary',
                                                           tf.summary.scalar(f'loss/{summary}_l2', loss_l2))
        tf.add_to_collection(f'{summary}_summary',
                             tf.summary.scalar(f'loss/{summary}_total', loss_total))

    return loss_total


# 优化器
def get_optimizer(lr):
    optimizer_name = config.optimizer_name
    if optimizer_name == 'AdamOptimizer':
        opt = tf.train.AdamOptimizer(lr)
    elif optimizer_name == 'MomentumOptimizer':
        opt = tf.train.MomentumOptimizer(lr, momentum=config.momentum)
    elif optimizer_name == 'AdagradOptimizer':
        opt = tf.train.AdagradOptimizer(lr)
    else:
        raise ValueError(f'ERROR: cannot find optimizer {optimizer_name}.')
    return opt
