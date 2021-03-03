"""
检测tensorflow能否把grads导出来，人为把grads扩大。
"""
import tensorflow as tf
import numpy as np
import time


"""测试np对列表元组的运算"""
# import numpy as np
#
# g1 = [(np.ones([3, 3, 3]), (np.ones([3, 3, 3]) * 2)),
#       (np.ones([3, ]), (np.ones([3, ]) * 2)),
#       (np.ones([3, 1, 2, 3]), (np.ones([3, 3, 3, 3]) * 2)),
#       (np.ones([5, ]), (np.ones([5, ]) * 2))]
# g2 = [(np.ones([3, 3, 3]), (np.ones([3, 3, 3]) * 2)),
#       (np.ones([3, ]), (np.ones([3, ]) * 2)),
#       (np.ones([3, 1, 2, 3]), (np.ones([3, 3, 3, 3]) * 2)),
#       (np.ones([5, ]), (np.ones([5, ]) * 2))]
# g3 = [(np.ones([3, 3, 3]), (np.ones([3, 3, 3]) * 2)),
#       (np.ones([3, ]), (np.ones([3, ]) * 2)),
#       (np.ones([3, 1, 2, 3]), (np.ones([3, 3, 3, 3]) * 2)),
#       (np.ones([5, ]), (np.ones([5, ]) * 2))]
# g4 = [(np.ones([3, 3, 3]), (np.ones([3, 3, 3]) * 2)),
#       (np.ones([3, ]), (np.ones([3, ]) * 2)),
#       (np.ones([3, 1, 2, 3]), (np.ones([3, 3, 3, 3]) * 2)),
#       (np.ones([5, ]), (np.ones([5, ]) * 2))]
#
# gg1 = np.array([g1, g2, g3, g4])
# gg2 = np.array([np.array(g) for g in [g1, g2, g3, g4]])
# gg3 = [g1, g2, g3, g4]
# gg11 = np.mean(gg1, axis=0)
# gg22 = np.mean(gg2, axis=0)
# gg33 = np.mean(gg3, axis=0)
#
# gg4 = np.array(g1)+np.array(g2)+np.array(g3)+np.array(g4)
# gg44 = gg4/4
# """测试结束"""


class Confg:
    def __init__(self):
        self.name = 'test'
        self.version = 'feed_grads'

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
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train1 = opt.apply_gradients(self.grads1)

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
        self.inputs = np.random.normal(size=[cfg.batch_size, 16, 16, 3])
        self.labels = np.random.randint(0, cfg.num_cls + 1, size=[cfg.batch_size, ])
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

    def train1(self):
        epochs = 1000
        run_list = [self.ts.global_step, self.ts.loss1, self.ts.grads1]
        for epoch in range(epochs):
            feed_dict = {self.ts.training: True}
            grads1 = []
            for _ in range(4):
                for ind_gpu in range(cfg.gpu_num):
                    sub_ts = self.ts.sub_ts[ind_gpu]
                    feed_dict.update({sub_ts.x: self.inputs, sub_ts.y: self.labels})
                self.session.run(self.ts.add_global_step)
                step, loss1, _grads1 = self.session.run(run_list, feed_dict)
                grads1.append(_grads1)
            grads1 = np.mean(np.array(grads1), axis=0)
            feed_dict = {self.ts.training: False,
                         sub_ts.x: self.inputs
                         }
            for ind, grad_var in enumerate(self.ts.grads1):
                grad, var = grad_var
                val_grads = grads1[ind][0]
                feed_dict[grad] = val_grads
            print(f'\rE {epoch + 1}/{epochs} step={step} loss1={loss1:.3f}', end='')
            self.session.run(self.ts.train1, feed_dict)

    def train2(self):
        epochs = 1000
        run_list = [self.ts.global_step, self.ts.loss1, self.ts.train1]
        feed_dict = {self.ts.training: True}
        for epoch in range(epochs):
            for ind_gpu in range(cfg.gpu_num):
                sub_ts = self.ts.sub_ts[ind_gpu]
                feed_dict.update({sub_ts.x: self.inputs, sub_ts.y: self.labels})
            self.session.run(self.ts.add_global_step)
            step, loss1, _ = self.session.run(run_list, feed_dict)
            print(f'\rE {epoch + 1}/{epochs} step={step} loss1={loss1:.3f}', end='')
            self.session.run(self.ts.train1, feed_dict)
        # time.sleep(1)
        # self.saver_arcface.save(self.sess, arcface_cfg.save_dir)
        # print('\tSAVED.')

    def close(self):
        self.session.close()


if __name__ == '__main__':
    app = App()

    app.train1()
    print()
    app.train1()
    print()
    app.train2()

    app.close()
