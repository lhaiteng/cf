"""
检测tensorflow的cpkt模型保存模式下各种行为的读取模型有效性。
"""
import tensorflow as tf
import numpy as np
import time


class Confg:
    def __init__(self):
        self.name = 'test'
        self.version = 'restore_ckpt'

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
            grads1 = self.merge_grads(lambda ts: ts.grads1)
            grads2 = self.merge_grads(lambda ts: ts.grads2)
            grads3 = self.merge_grads(lambda ts: ts.grads3)
            grads21 = self.merge_grads(lambda ts: ts.grads21)
            grads22 = self.merge_grads(lambda ts: ts.grads22)
            grads30 = self.merge_grads(lambda ts: ts.grads30)
            # # [(grad, var), (grad, avr), ...]
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train1 = opt.apply_gradients(grads1)
                self.train2 = opt.apply_gradients(grads2)
                self.train3 = opt.apply_gradients(grads3)
                self.train21 = opt.apply_gradients(grads21)
                self.train22 = opt.apply_gradients(grads22)
                self.train30 = opt.apply_gradients(grads30)

            print('Reduce_meaning loss1...')
            self.loss1 = tf.reduce_mean([_ts.loss1 for _ts in self.sub_ts])
            self.loss2 = tf.reduce_mean([_ts.loss2 for _ts in self.sub_ts])
            self.loss3 = tf.reduce_mean([_ts.loss3 for _ts in self.sub_ts])

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
        var1 = [v for v in tf.trainable_variables() if 'dense' in v.name]
        self.grads1 = opt.compute_gradients(self.loss1, var_list=var1)
        var3 = [v for v in tf.trainable_variables() if 'conv' in v.name]
        self.grads3 = opt.compute_gradients(self.loss1, var_list=var3)

        loss2 = tf.square(y - x)
        self.loss2 = tf.reduce_mean(loss2)
        self.grads2 = opt.compute_gradients(self.loss2)
        self.grads21 = opt.compute_gradients(self.loss2, var_list=var1)
        self.grads22 = opt.compute_gradients(self.loss2, var_list=var3)

        loss3 = tf.multiply(x, y)
        self.loss3 = tf.reduce_mean(loss3)
        self.grads30 = opt.compute_gradients(self.loss3)

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
        batch_size = cfg.batch_size
        epochs = 1000
        inputs = np.random.normal(size=[batch_size, 16, 16, 3])
        labels = np.random.randint(0, cfg.num_cls + 1, size=[batch_size, ])
        run_list = [self.ts.train1, self.ts.loss1, self.ts.global_step, self.ts.train2, self.ts.loss2, self.ts.train3,
                    self.ts.train21, self.ts.train22,
                    self.ts.loss3, self.ts.train30]
        feed_dict = {self.ts.training: True}
        for epoch in range(epochs):
            for ind_gpu in range(cfg.gpu_num):
                sub_ts = self.ts.sub_ts[ind_gpu]
                feed_dict.update({sub_ts.x: inputs, sub_ts.y: labels})
            self.session.run(self.ts.add_global_step)
            _, loss1, step, _, loss2, _, _, _, loss3, _ = self.session.run(run_list, feed_dict)
            print(f'\rE {epoch + 1}/{epochs} step={step} loss1={loss1:.3f} loss2={loss2:.3f} loss3={loss3:.3f}', end='')
        time.sleep(1)
        self.saver.save(self.session, cfg.save_path)
        print('\tSAVED.')

    def close(self):
        self.session.close()


if __name__ == '__main__':
    app = App()

    app.train()

    app.close()

    app = App()

    app.train()

    app.close()
