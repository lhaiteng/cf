# -*- coding: utf-8 -*-
import math
import tensorflow as tf


def get_arcface_loss(thetas, labels_onehot, m1, m2, m3, s, alpha=None, gamma=None):
    # norm_x = tf.norm(x, axis=1, keepdims=True)
    # cos_theta = normx_cos / norm_x
    # thetas = tf.acos(cos_theta)
    # mask = tf.one_hot(labels_onehot, depth=normx_cos.shape[-1])

    # 惩罚角、扩大系数
    zeros = tf.zeros_like(labels_onehot)
    cond = tf.where(tf.greater(thetas * m1 + m2, math.pi), zeros, labels_onehot)
    cond = tf.cast(cond, dtype=tf.bool)
    m1_theta_plus_m2 = tf.where(cond, thetas * m1 + m2, thetas)
    cos_m1_theta_plus_m2 = tf.cos(m1_theta_plus_m2)
    logits = tf.where(cond, cos_m1_theta_plus_m2 - m3, cos_m1_theta_plus_m2) * s

    if alpha is None or gamma is None:  # CrossEntropy Loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_onehot, logits=logits)
        loss = tf.reduce_mean(loss)
    else:  # Focal Loss
        # 提取样本分类的概率
        p = tf.nn.softmax(logits)
        p = tf.reduce_sum(labels_onehot * p, axis=1)  # [None, ]
        # 1-p
        one_p = tf.ones_like(p, dtype=p.dtype) - p
        # 计算损失
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_onehot, logits=logits)
        loss = alpha * one_p ** gamma * ce
        loss = tf.reduce_mean(loss)

    return loss


def get_center_loss(features, label, centers, alpha, com_loss):
    centers_batch = tf.gather(centers, label)
    diff = (1 - alpha) * (centers_batch - features)
    # 论文代码
    centers = tf.compat.v1.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    # # 因为会自动更新ref所指代的变量，所以认为不需要返回啊。
    # centers_op = tf.compat.v1.scatter_sub(centers, label, diff)
    # with tf.control_dependencies([centers_op]):
    #     loss = tf.cond(com_loss,  # 若不需要计算loss则返回0
    #                    lambda: tf.reduce_mean(tf.square(features - centers_batch)), lambda: 0.)
    return loss, centers
