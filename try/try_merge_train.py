"""
检测tensorflow能否在tf中求不同批次的grads均值，再更新权重
"""
import tensorflow as tf
import numpy as np
import time


class Confg:
    def __init__(self):
        self.name = 'try'
        self.version = 'merge_train'

        self.gpu_num = 1

        self.num_cls = 10

        self.save_path = f'./models/{self.name}_{self.version}'

        self.batch_size = 16


cfg = Confg()


class Tensor:
    def __init__(self):
        with tf.device('/gpu:0'):
            # 占位符
            self.training = tf.placeholder(tf.bool, shape=[], name='training')
            # 更新一次需要几个批次
            self.train_num_batch = tf.placeholder(tf.float32, shape=[], name='train_num_batch')
            # 优化器
            opt = tf.train.GradientDescentOptimizer(1)
            # 计步器
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)
            self.add_global_step = tf.assign_add(self.global_step, 1)

        with tf.variable_scope('APP'):  # 这句话很关键！
            self.sub_ts = []
            for i in range(cfg.gpu_num):
                first = False if i == 0 else True
                with tf.device(f'/gpu:{i}'):
                    print(f'GPU: {i}')
                    self.sub_ts.append(SubTensor(opt, self.training, first))

        with tf.device('/gpu:0'):
            self.vars = tf.trainable_variables()
            print('Merging grads...')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.grads1 = self.merge_grads(lambda ts: ts.grads1)
                # # [(grad, var), (grad, avr), ...]

            # 先得到相同形状的grads以待求和
            # 必须是变量形式，否则不能assign。非trainable
            self.grads = {gv[1]: tf.get_variable(name=f'sum_grads{ind}', shape=gv[0].shape,
                                                 initializer=tf.initializers.zeros(), trainable=False)
                          for ind, gv in enumerate(self.grads1)}

            # 分别对每个梯度进行初始化
            self.assign_zero_grads = tf.initialize_variables([g for v, g in self.grads.items()])
            # 赋值op的列表, 分别将梯度累加进去
            self.assign_grads = [tf.assign_add(self.grads[v], g)
                                 for g, v in self.grads1]

            # 求均值操作列表
            self.avg_grads_op = [tf.assign(g, g / self.train_num_batch)
                                 for v, g in self.grads.items()]
            # # 此时的sum_grads中的grads应该是均值的了

            # 利用均值进行优化
            self.train1 = opt.apply_gradients([(g, v) for v, g in self.grads.items()])

            print('Reduce_meaning loss1...')
            self.loss1 = tf.reduce_mean([_ts.loss1 for _ts in self.sub_ts])

    def merge_grads(self, f):
        """
        ts.grads [(grad, var), (grad, var), ...]
        :return: [(grad, var), (grad, var), ...]
        """
        var_grad = {}  # var: [grad1, grad2, ...]
        var_IndexedSlices = {}  # var: [IndexedSlices1, IndexedSlices2, ...]
        for ts in self.sub_ts:
            for grad, var in f(ts):
                if grad is None:
                    continue
                if isinstance(grad, tf.IndexedSlices):
                    if var not in var_IndexedSlices:
                        var_IndexedSlices[var] = []
                    var_IndexedSlices[var].append(grad)
                else:
                    if var not in var_grad:
                        var_grad[var] = []
                    var_grad[var].append(grad)

        # 返回用来求梯度的gv对
        # 普通var-grads直接求平均
        grad_var = [(tf.reduce_mean(var_grad[var], axis=0), var) for var in var_grad]
        # grad_var = [(var_grad[var][0], var) for var in var_grad]
        # 切片，则把不同GPU得到的切片值、索引，拼接起来，再形成新的切片
        for var in var_IndexedSlices:
            IndexedSlices = var_IndexedSlices[var]  # [IndexedSlices1, IndexedSlices2, ...]
            indices = tf.concat([i.indices for i in IndexedSlices], axis=0)
            values = tf.concat([i.values for i in IndexedSlices], axis=0)
            new_IndexedSlices = tf.IndexedSlices(values, indices)
            grad_var.append((new_IndexedSlices, var))
        return grad_var

    def assign_sum_grads(self, grad_vars, to_add_grad_vars):
        """
        得到累加的操作
        只适用于非tf.IndexedSlices类型的梯度
        :param grad_vars: 需要assign的
        :param _grad_vars: 添加进去的
        :return:
        """
        v_g_dict = {}
        res = []
        for _g_v in (to_add_grad_vars, grad_vars):
            for g, v in _g_v:
                if v in v_g_dict:
                    res.append(tf.assign(g, g + v_g_dict[v]))
                else:
                    v_g_dict[v] = g
        return res


class SubTensor:
    def __init__(self, opt, training, first):
        self._training = training
        self._first = first

        self.x = tf.placeholder(tf.float32, shape=[None, 16, 16, 3], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, ], name='y')
        y = tf.one_hot(self.y, depth=cfg.num_cls, dtype=tf.float32)

        x = self.conv(self.x, name='conv', reuse=first)
        x = tf.layers.flatten(x)
        x = self.dense(x, name='dense', reuse=first)

        loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x)
        self.loss1 = tf.reduce_mean(loss1)
        var1 = [v for v in tf.trainable_variables()]
        self.grads1 = opt.compute_gradients(self.loss1, var_list=var1)

    def conv(self, x, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            for ind in range(4):
                x = tf.layers.conv2d(x, 3, 3, 1, name=f'c3s1_{ind}', padding='same')
                x = tf.layers.batch_normalization(x, name=f'bn{ind}', training=self._training)
                x = tf.nn.relu(x)
                x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
        return x

    def dense(self, x, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            x = tf.layers.dense(x, 1, name='dense1')
            x = tf.layers.dense(x, 1, name='dense2')
            x = tf.layers.dense(x, cfg.num_cls, name='dense3')
        return x


class App:
    def __init__(self):
        self.inputs = [np.random.normal(size=[cfg.batch_size, 16, 16, 3]),
                       np.random.normal(size=[cfg.batch_size, 16, 16, 3]),
                       np.random.normal(size=[cfg.batch_size, 16, 16, 3]),
                       np.random.normal(size=[cfg.batch_size, 16, 16, 3])]
        self.labels = [np.random.randint(0, cfg.num_cls + 1, size=[cfg.batch_size, ]),
                       np.random.randint(0, cfg.num_cls + 1, size=[cfg.batch_size, ]),
                       np.random.randint(0, cfg.num_cls + 1, size=[cfg.batch_size, ]),
                       np.random.randint(0, cfg.num_cls + 1, size=[cfg.batch_size, ])]
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.ts = Tensor()
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.session = tf.Session(config=conf)
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            try:
                self.saver.restore(self.session, cfg.save_path)
                print(f'Restore model from f{cfg.save_path} succeed.')
            except:
                print(f'Restore model from f{cfg.save_path} failed.')

    def train(self):
        epochs = 10
        for epoch in range(epochs):
            feed_dict = {self.ts.training: True}
            # 先把求和grads初始化
            self.session.run(self.ts.assign_zero_grads)
            for i in range(4):  # train_num_batch = 4
                run_list = [self.ts.loss1, self.ts.grads1, self.ts.assign_grads]
                for ind_gpu in range(cfg.gpu_num):  # 喂入不同gpu中值
                    sub_ts = self.ts.sub_ts[ind_gpu]
                    feed_dict.update({sub_ts.x: self.inputs[i], sub_ts.y: self.labels[i]})
                loss1, grads1, _ = self.session.run(run_list, feed_dict)
                grads = self.session.run(self.ts.grads)
                print(f'EPOCH {epoch} BATCH {i} loss1={loss1:.3f}')
                print(f'grads1:\n{grads1}\ngrads:\n{grads}')
            # 求均值
            self.session.run(self.ts.avg_grads_op, {self.ts.train_num_batch: 4})
            print(self.session.run(self.ts.grads))
            # 优化梯度
            self.session.run(self.ts.train1)
            print('*' * 60)

    def close(self):
        self.session.close()


if __name__ == '__main__':
    app = App()

    app.train()
    app.close()
