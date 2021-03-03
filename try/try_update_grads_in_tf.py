"""
在tensorflow中实现多次训练的梯度平均后，再更新权重
失败放弃。
考虑到在tf中更新权重，需要再另建权重的梯度标量，对历史梯度进行保存，数量太大，估计内存承受不了。
"""
import tensorflow as tf
import numpy as np
import time


class Confg:
    def __init__(self):
        self.name = 'test'
        self.version = 'update_grads_after_times_in_tf'

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
            # 优化器
            opt = tf.train.AdamOptimizer()
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
            print('Merging grads...')
            self.grads1 = self.merge_grads(lambda ts: ts.grads1)
            # # [(grad, var), (grad, avr), ...]

            # 若是重新来了一次，则把梯度累加
            self.should_new_grads1 = tf.placeholder(tf.bool, [], 'should_new_grads1')
            self.update_grads1 = self.create_zero_grads_like(self.grads1)
            # 累加操作符的列表
            self.assgin_add_grads1 = self.op_add_grads(self.update_grads1, self.grads1, self.should_new_grads1)
            # 平均操作符的列表
            assign_avg_grads1 = self.op_avg_grads(self.update_grads1, 4)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS) + assign_avg_grads1):
                self.train1 = opt.apply_gradients(zip(self.update_grads1,
                                                      [gv[1] for gv in self.grads1]))

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
                    grad = tf.zeros_like(var, dtype=tf.float32)
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

    def create_zero_grads_like(self, grad_vars):
        new_grad_vars = []
        for ind, gv in enumerate(grad_vars):
            grad, var = gv
            new_grad = tf.get_variable(name=f'{ind}', shape=tf.shape(gv), dtype=tf.float32,
                                       trainable=False, initializer=tf.initializers.zeros())
            new_grad_vars.append((new_grad, var))
        return new_grad_vars

    def op_add_grads(self, update_grads, grads, should_new_grads):
        ops = []
        # 若should_new_grads，则将grad付给update_grad
        # 否则，将grad_update_grad付给update_grad
        for ind, gv in enumerate(grads):
            grad, var = gv
            op = tf.cond(should_new_grads,
                         lambda: tf.assign(update_grads[ind], grad),
                         lambda: tf.assign(update_grads[ind], update_grads[ind] + grad))
            ops.append(op)
        return ops

    def op_avg_grads(self, grads, num):
        return [tf.assign(grads[0][0], grads[0][0] / num) for i in range(len(grads))]


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
        epochs = 5
        run_list = [self.ts.global_step, self.ts.loss1, self.ts.grads1, self.ts.assgin_add_grads1]
        grads1, loss1 = None, None
        for epoch in range(epochs):
            feed_dict = {self.ts.training: True}
            # 多次喂入值，并更新累积的梯度
            for i in range(4):
                if i == 0:
                    feed_dict.update({self.ts.should_new_grads1: True})
                else:
                    feed_dict.update({self.ts.should_new_grads1: False})
                for ind_gpu in range(cfg.gpu_num):
                    sub_ts = self.ts.sub_ts[ind_gpu]
                    feed_dict.update({sub_ts.x: self.inputs[i], sub_ts.y: self.labels[i]})
                step, _loss1, _grads1, _ = self.session.run(run_list, feed_dict)
                if i == 0:
                    loss1 = _loss1
                else:
                    loss1 += _loss1
                print('_loss1:\n', _loss1)
                print('_grads1:\n', _grads1)
            _update_grads1 = self.session.run(self.ts.update_grads1)
            print('_update_grads:\n', _update_grads1)
            # 更新梯度
            _, avg_grads1 = self.session.run([self.ts.train1, self.ts.avg_grads1], feed_dict)
            print('loss1:\n', loss1)
            print('avg_grads1:\n', avg_grads1)
            print(f'EPOCH {epoch} step={step} loss1={loss1 / 4}')
            self.session.run(self.ts.add_global_step)

    def close(self):
        self.session.close()


if __name__ == '__main__':
    app = App()

    app.train()

    app.close()
