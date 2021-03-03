"""
目标检测中使用iou或giou作为回归损失
"""
import numpy as np
import cv2
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import tensorflow as tf


# 用于tensorflow
class LossGen:
    def __init__(self):
        pass

    def get_loss_iou(self, gts, boxes):
        # [x, y, w, h]
        iou = self.get_iou(gts, boxes)

        return 1 - tf.reduce_mean(iou)

    def get_loss_giou(self, gts, boxes):
        giou = self.get_giou(gts, boxes)

        return 1 - tf.reduce_mean(giou)

    def get_iou(self, boxes1, boxes2):
        # [None, 4] -> [x, y, w, h]
        # 转成[r1, c1, r2, c2]
        b1x, b1y, b1w, b1h = [boxes1[:, i] for i in range(4)]
        b1r1, b1c1, b1r2, b1c2 = b1y - b1h / 2, b1y + b1h / 2, b1x - b1w / 2, b1x + b1w / 2
        b2x, b2y, b2w, b2h = [boxes2[:, i] for i in range(4)]
        b2r1, b2c1, b2r2, b2c2 = b2y - b2h / 2, b2y + b2h / 2, b2x - b2w / 2, b2x + b2w / 2

        inter_w = tf.nn.relu(tf.nn.relu(b1c2 - b2c1)
                             - (tf.nn.relu(b1c1 - b2c1) + tf.nn.relu(b1c2 - b2c2)))
        inter_h = tf.nn.relu(tf.nn.relu(b1r2 - b2r1)
                             - (tf.nn.relu(b1r1 - b2r1) + tf.nn.relu(b1r2 - b2r2)))

        inter_area = inter_w * inter_h
        union_area = b1w * b1h + b2w * b2h - inter_area

        return inter_area / union_area

    def get_giou(self, boxes1, boxes2):
        # [None, 4] -> [x, y, w, h]
        # 转成[r1, c1, r2, c2]
        b1x, b1y, b1w, b1h = [boxes1[:, i] for i in range(4)]
        b1r1, b1c1, b1r2, b1c2 = b1y - b1h / 2, b1y + b1h / 2, b1x - b1w / 2, b1x + b1w / 2
        b2x, b2y, b2w, b2h = [boxes2[:, i] for i in range(4)]
        b2r1, b2c1, b2r2, b2c2 = b2y - b2h / 2, b2y + b2h / 2, b2x - b2w / 2, b2x + b2w / 2
        inter_w = tf.nn.relu(tf.nn.relu(b1c2 - b2c1)
                             - (tf.nn.relu(b1c1 - b2c1) + tf.nn.relu(b1c2 - b2c2)))
        inter_h = tf.nn.relu(tf.nn.relu(b1r2 - b2r1)
                             - (tf.nn.relu(b1r1 - b2r1) + tf.nn.relu(b1r2 - b2r2)))

        inter_area = inter_w * inter_h
        union_area = b1w * b1h + b2w * b2h - inter_area

        # 区域C
        cr1, cr2 = tf.minimum(b1r1, b2r1), tf.maximum(b1r2, b2r2)
        cc1, cc2 = tf.minimum(b1c1, b2c1), tf.maximum(b1c2, b2c2)
        cw, ch = cc2 - cc1, cr2 - cr1
        c_area = cw * ch

        return inter_area / union_area - (c_area - union_area) / c_area


data = np.random.random(100000)
data = data[data > 1e-9]

# [x, y, w, h]
choice_data = np.random.choice(data, size=8000, replace=False).reshape(1000, 8)
data_gts, data_boxes = choice_data[:, :4], choice_data[:, 4:]

inputs_gts = tf.placeholder(tf.float32, [None, 4])
inputs = tf.placeholder(tf.float32, [None, 10, 10, 3])

x = tf.layers.conv2d(inputs, 10, 3, 1, padding='same')
x = tf.layers.conv2d(x, 10, 3, 1, padding='same')
x = tf.layers.conv2d(x, 4, 3, 1, padding='same')
x = tf.layers.flatten(x)  # [None, 100*4]
x = tf.layers.dense(x, 4)  # [None, 4]

lossgen = LossGen()

loss_iou = lossgen.get_loss_iou(inputs_gts, x)
loss_giou = lossgen.get_loss_giou(inputs_gts, x)

train_iou = tf.train.GradientDescentOptimizer(0.001).minimize(loss_iou)
train_giou = tf.train.GradientDescentOptimizer(0.001).minimize(loss_giou)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        _, lo_iou = sess.run([train_iou, loss_iou], {inputs_gts: data_gts,
                                                     inputs: np.random.random([1000, 10, 10, 3])})
        print(f'epoch = {epoch}\tloss_iou = {lo_iou:.6f}')

    sess.run(init)
    for epoch in range(100):
        _, lo_giou = sess.run([train_iou, loss_giou], {inputs_gts: data_gts,
                                                       inputs: np.random.random([1000, 10, 10, 3])})
        print(f'epoch = {epoch}\tloss_giou = {lo_giou:.6f}')
