# -*- coding: utf-8 -*-
"""
目的：
    - 能否把人脸id识别拿出来单独训练。
    - 能否进行微调。
验证功能：
    1. 使用tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, datasets='xxx')获取训练变量
        似乎不太好用。在构造函数中，可以正常返回所有变量，但在train函数中，得到的是空列表。
        猜想是因为初始化了？并不是，在构造函数中，如果初始化后还是能够正常返回
        可能是需要指定graph
    2. 同时训练完成后只改变一小部分网络，重训练其他网络
        成功！改变最后一层的分类数，成功读取其他层的权重，而初始化最后一层的权重
    3. 构建两个结构，使用两个saver对其操作，定义saver时可限定var_list。
    4. 第二个结构的inputs是第一个结构的输出张量。训练第二个结构的train，但喂入第一个结构的占位符。
"""
import tensorflow as tf
import numpy as np


class Config:
    def __init__(self):
        self.ts1_classes = 10
        self.epochs1 = 100
        self.save_path1 = './models/try_finetune_ts1'

        self.ts2_classes = 15
        self.epochs2 = 100
        self.save_path2 = './models/try_finetune_ts2'


class Ts1:
    def __init__(self, cfg=Config()):
        with tf.device('/gpu:0'):
            self.inputs = tf.placeholder(tf.float32, shape=[None, 5], name='ts1_inputs')
            self.labels = tf.placeholder(tf.int32, shape=[None, ], name='ts1_labels')
            labels = tf.one_hot(self.labels, cfg.ts1_classes)

            opt = tf.train.AdamOptimizer()  # 优化器的变量是不会放在trainable_variables中的

            with tf.variable_scope('ts1', reuse=False):
                x = tf.layers.dense(self.inputs, 3, name='dense1')
                x = tf.layers.batch_normalization(x, name='bn1', training=True)
                x = tf.layers.dense(x, 3, name='dense2')
                x = tf.layers.batch_normalization(x, name='bn2', training=True)
                x = tf.layers.dense(x, 3, name='dense3')
                self.output = tf.layers.dense(x, 3, name='output')
                logits = tf.layers.dense(self.output, cfg.ts1_classes, name='logits')

            # 用来恢复的变量
            # self.vars_to_restore = []
            self.vars_to_restore = [v for v in tf.trainable_variables()
                                    if 'ts1' in v.name]  # and 'logits' not in v.name]

            # # 用来初始化重训练的变量
            # self.vars_to_init = [v for v in tf.trainable_variables()
            #                      if 'ts1' in v.name and v not in self.vars_to_restore]

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
            # 全部的变量
            self.vars_all_trainable = [v for v in tf.trainable_variables()
                                       if 'ts1' in v.name]
            self.train_op = opt.minimize(loss, var_list=self.vars_all_trainable)


class Ts2:
    def __init__(self, cfg=Config(), inputs=None):
        with tf.device('/gpu:0'):
            if inputs is None:
                self.inputs = tf.placeholder(tf.float32, shape=[None, 3], name='ts2_inputs')
            else:
                self.inputs = inputs
            self.labels = tf.placeholder(tf.int32, shape=[None, ], name='ts2_labels')
            labels = tf.one_hot(self.labels, cfg.ts2_classes)

            opt = tf.train.AdamOptimizer()  # 优化器的变量是不会放在trainable_variables中的

            with tf.variable_scope('ts2', reuse=False):
                x = tf.layers.dense(self.inputs, 5, name='dense1')
                x = tf.layers.dense(x, 5, name='dense2')
                x = tf.layers.dense(x, 5, name='dense3')
                x = tf.layers.dense(x, 5, name='dense4')
                logits = tf.layers.dense(x, cfg.ts2_classes, name='logits')

            # 用来恢复的变量
            self.vars_to_restore = [v for v in tf.trainable_variables()
                                    if 'ts2' in v.name and 'logits' not in v.name]

            # # 用来初始化重训练的变量
            # self.vars_to_init = [v for v in tf.trainable_variables()
            #                      if 'ts2' in v.name and v not in self.vars_to_restore]

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
            # 全部的变量
            self.vars_all_trainable = [v for v in tf.trainable_variables()
                                       if 'ts2' in v.name or 'ts1' in v.name]
            self.train_op = opt.minimize(loss, var_list=self.vars_all_trainable)

            # 训练1和2
            vars_all_trainable = [v for v in tf.trainable_variables()
                                  if 'ts2' in v.name or 'ts1' in v.name]
            self.train_op_all = opt.minimize(loss, var_list=vars_all_trainable)



class APP:
    def __init__(self, cfg=Config()):
        self.cfg = cfg
        graph = tf.Graph()
        with graph.as_default():
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)
            ts_paths = []

            # 模型1
            self.ts1 = Ts1(cfg)
            # 建立新的保存模型，只保存ts1中的变量
            # self.vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, datasets='ts1')
            # # # 所有变量，但如果训练器有变量，是不会包括在内的
            self.saver1 = tf.train.Saver(var_list=self.ts1.vars_all_trainable)
            # 重新输出图，因为可能图中部分有些变化
            self.saver1.export_meta_graph(self.cfg.save_path1 + '.meta')
            ts_paths.append((self.ts1, self.cfg.save_path1))
            # 模型2
            self.ts2 = Ts2(cfg, self.ts1.output)  # 构建时已经把1和2连在了一起
            self.saver2 = tf.train.Saver(var_list=self.ts2.vars_all_trainable)
            self.saver2.export_meta_graph(self.cfg.save_path2 + '.meta')
            ts_paths.append((self.ts2, self.cfg.save_path2))

            # 先初始化所有变量，因为优化器的变量也要初始化，而他不在trainable_variables中。
            # 因而在ts中，只需写待restore的变量就可以了。
            self.sess.run(tf.global_variables_initializer())
            print('SUCCESS: init all vars.')
            self.restore_vars(ts_paths)

    def restore_vars(self, ts_paths):
        for ts, path in ts_paths:
            try:
                self._restore_vars(ts, path)
                print(f'SUCCESS: restore {path}.')
            except:
                print(f'FAIL: restore {path}.')

    def _restore_vars(self, ts, path):
        # 恢复部分变量
        if ts.vars_to_restore:
            _saver1 = tf.train.Saver(ts.vars_to_restore)
            _saver1.restore(self.sess, path)
        # # 初始化其余变量
        # self.sess.run(tf.variables_initializer(ts.vars_to_init))

    def train1(self):
        """
        小结：
            - 是否读取了训练前的变量？  是的！
            - 是否重置了想再训练的变量？  是的！
        """
        inputs = np.random.normal(size=[10, 5])
        labels = np.random.randint(0, self.cfg.ts1_classes, size=[10, ])
        # 输出训练前的变量
        print('*' * 50, 'ts1', '*' * 50)
        for v in self.ts1.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50)
        # 训练
        for epoch in range(self.cfg.epochs1):
            self.sess.run(self.ts1.train_op, {self.ts1.inputs: inputs, self.ts1.labels: labels})
            self.saver1.save(self.sess, self.cfg.save_path1,
                             write_meta_graph=False)  # 不输出图，只保存数据
        # 输出训练后的变量
        print('*' * 50, 'ts1', '*' * 50)
        for v in self.ts1.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50)

    def train2_1(self):
        # 把ts1的output输入到ts2中。
        # 在train_op中指定了训练的参数，可尝试能否对ts1的变量进行训练
        """
        小结：
            - 固定1的变量，只对2的变量，进行训练？  是的！
            - 固定1的变量，读取2的某些变量，初始化某些变量，进行训练？  是的！
        """
        inputs = np.random.normal(size=[10, 5])
        labels1 = np.random.randint(0, self.cfg.ts1_classes, size=[10, ])
        labels2 = np.random.randint(0, self.cfg.ts2_classes, size=[10, ])
        # 输出训练前ts1和ts2的变量
        print('*' * 50, 'ts1', '*' * 50)
        for v in self.ts1.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50, 'ts2', '*' * 50)
        for v in self.ts2.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50)
        # 训练
        for epoch in range(self.cfg.epochs2):
            self.sess.run(self.ts2.train_op, {self.ts1.inputs: inputs, self.ts2.labels: labels2})
        # 保存2
        # self.saver1.save(self.sess, self.arcface_cfg.save_path1, write_meta_graph=False)
        self.saver2.save(self.sess, self.cfg.save_path2, write_meta_graph=False)
        # 输出训练后ts1和ts2的变量
        print('*' * 50, 'ts1', '*' * 50)
        for v in self.ts1.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50, 'ts2', '*' * 50)
        for v in self.ts2.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50)

    def train2_2(self):
        # 把ts1的output输入到ts2中。
        # 在train_op中指定了训练的参数，可尝试能否对ts1的变量进行训练
        # 在2的train中设为1和2的变量
        """
        小结：
            - 不保存1，只保存2？  是的！
            - 同时保存1和2？  是的！
        """
        inputs = np.random.normal(size=[10, 5])
        labels1 = np.random.randint(0, self.cfg.ts1_classes, size=[10, ])
        labels2 = np.random.randint(0, self.cfg.ts2_classes, size=[10, ])
        # 输出训练前ts1和ts2的变量
        print('*' * 50, 'ts1', '*' * 50)
        for v in self.ts1.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50, 'ts2', '*' * 50)
        for v in self.ts2.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50)
        # 训练
        for epoch in range(self.cfg.epochs2):
            self.sess.run(self.ts2.train_op_all, {self.ts1.inputs: inputs, self.ts2.labels: labels2})
        # 保存1和2
        self.saver1.save(self.sess, self.cfg.save_path1, write_meta_graph=False)
        self.saver2.save(self.sess, self.cfg.save_path2, write_meta_graph=False)
        # 输出训练后ts1和ts2的变量
        print('*' * 50, 'ts1', '*' * 50)
        for v in self.ts1.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50, 'ts2', '*' * 50)
        for v in self.ts2.vars_all_trainable:
            val = self.sess.run(v)
            print(f'name:{v.name}:\n{val}')
        print('*' * 50)

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    app = APP()

    # app.train1()
    # app.train2_1()
    app.train2_2()

    app.close()
