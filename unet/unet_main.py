# -*- coding: utf-8 -*-
import tensorflow as tf
from unet.unet_config import unet_cfg as cfg
from unet.unet_samples import Samples
from unet.unet_tensors import Tensor
import numpy as np
import matplotlib.pyplot as plt
import cv2


class App:
    def __init__(self):
        cfg.check_all_dir_paths()
        graph = tf.Graph()
        self.sa = Samples()
        with graph.as_default():
            with tf.device('/gpu:0'):
                conf = tf.ConfigProto()
                conf.allow_soft_placement = True
                self.sess = tf.Session(config=conf)
                self.ts = Tensor()
                # 保存器
                self.saver_unet = tf.train.Saver(var_list=self.ts.vars_save, max_to_keep=5)
                self.saver_unet.export_meta_graph(cfg.save_path + '.meta')
                print(f'SUCCEED: export_meta_graph to {cfg.save_path}.meta')
                # 读取权重
                ts_paths = []
                # ts_paths.append((self.ts.vars_restore, cfg.save_path + f'-{cfg.start_epoch}'))
                ts_paths.append((self.ts.vars_restore, cfg.save_path + f'-37'))

                self.init_restore_model(ts_paths)

    def init_restore_model(self, ts_paths):
        # 先初始化所有变量
        self.sess.run(tf.global_variables_initializer())
        print('FINISH: init all vars.')

        for ts_list, save_path in ts_paths:
            print('-' * 50)
            # print(f'path: {path}\nf var_restore_list: {var_restore_list}')
            try:
                saver = tf.train.Saver(var_list=ts_list)
                saver.restore(self.sess, save_path)
                print(f'SUCCEED: restore model from {save_path}.')
            except:
                print(f'FAILED: restore model from {save_path}.')
        print('-' * 50)

    def train(self, batch_size=None):
        start_epoch = cfg.start_epoch
        num_epoch = cfg.epochs
        if batch_size is None:
            batch_size = cfg.train_batch_size
        # 每训练一次使用几个batch的平均梯度
        train_batch = cfg.train_batch
        # 一个epoch有多少batch
        num_batch = self.sa.train_num // (batch_size * cfg.gpu_num * train_batch)
        # 经过多少batch显示一次预测
        show_num = int(cfg.show_num // (batch_size * cfg.gpu_num * train_batch) * 1.1)
        # 每个epoch经过多少batch显示一次损失
        num_print_loss = cfg.num_batch_to_print_losses
        self.print_train_infos(batch_size, num_batch, num_epoch, num_print_loss, show_num, start_epoch, train_batch)

        # self.sess.run(tf.assign(self.ts.global_step, 0))
        # step = self.sess.run(self.ts.global_step)
        # 令步数从0开始计算
        step = 0
        run_loss = [self.ts.loss]
        feed_dict = {self.ts.keep_prob: cfg.keep_prob, self.ts.training: True}
        for epoch in range(start_epoch, num_epoch):
            lr = cfg.lr[epoch]
            for batch in range(num_batch):
                if step % show_num == 0:
                    self.predict_samples()

                # 训练train_batch次，得到损失和梯度
                losses = self._get_grads_losses(batch_size, train_batch, run_loss, feed_dict)
                # 更新梯度
                self.sess.run(self.ts.update_grads,
                              feed_dict={self.ts.lr: lr, self.ts.train_num_batch: train_batch})
                # 显示结果
                if num_print_loss == 0 or step % num_print_loss == 0:
                    losses /= train_batch
                    print(f'\rEPOCH {epoch}/{num_epoch} BATCH {batch}/{num_batch}'
                          f'\tTRAIN loss={losses[0]:.3f}', end='')
                # 更新步数
                self.sess.run(self.ts.add_global_step)
                step = self.sess.run(self.ts.global_step)

            # epoch进行完后，保存、预测
            self.predict_samples(show_pic=False, show_loss=True, save_pic=True, save_name=f'{cfg.version}_{epoch}')
            self.saver_unet.save(self.sess, cfg.save_path, global_step=epoch,
                                 write_meta_graph=False)
            print(f'Save into model {cfg.save_path}')

    def print_train_infos(self, batch_size, num_batch, num_epoch, num_print_loss, show_num, start_epoch, train_batch):
        print(f'Start training from epoch No.{start_epoch}...')
        print(f'Train pics {self.sa.train_num}.')
        print(f'Train num_epoch: {num_epoch}')
        print(f'Train num_batch: {num_batch}.')
        print(f'Train batch_size: {cfg.max_batch_size} -> train {batch_size} pics for {train_batch} times.')
        print(f'Show loss every {num_print_loss} batch / epoch.')
        print(f'Show predict every {show_num} batch, '
              f'after train {show_num * batch_size * cfg.gpu_num * train_batch} pics.')

    def _get_grads_losses(self, batch_size, train_batch, run_loss, feed_dict):
        # 每个batch的损失、梯度初始化
        losses = None
        self.sess.run(self.ts.assign_zero_grads)
        for _ in range(train_batch):
            # 喂入各gpu值
            self.update_feed_dict(feed_dict, batch_size, cate='train')
            # 计算损失、累加梯度
            infos = self.sess.run(run_loss + [self.ts.assign_grads], feed_dict=feed_dict)
            if losses is None:
                losses = np.array(infos[:1])
            else:
                losses += np.array(infos[:1])
        return losses

    def update_feed_dict(self, feed_dict, batch_size, cate='train'):
        for ind_gpu in range(cfg.gpu_num):
            sub_ts = self.ts.sub_ts[ind_gpu]
            feed_dict.update({sub_ts.imgs: self.sa.next_batch(batch_size, cate)})

    def predict_samples(self, batch_size=5, cate='test', show_pic=True,
                        show_loss=False, save_pic=False, save_name=None):
        if save_name is None:
            save_name = cfg.name + cfg.version
        imgs = self.sa.next_batch(batch_size=batch_size, cate=cate)
        sub_ts = self.ts.sub_ts[0]
        feed_dict = {self.ts.keep_prob: 1, self.ts.training: False,
                     sub_ts.imgs: imgs}
        re_imgs, loss = self.sess.run([sub_ts.re_imgs, sub_ts.loss], feed_dict=feed_dict)
        if show_loss:
            print(f'\tTEST loss={loss:.3f}')

        imgs = np.concatenate(imgs, axis=0)
        re_imgs = np.concatenate(re_imgs, axis=0)*255
        imgs = np.concatenate([imgs, re_imgs], axis=1).astype(np.uint8)

        if show_pic:
            plt.imshow(imgs[:, :, ::-1])
            plt.show()

        if save_pic:
            cv2.imwrite(cfg.gen_dir + f'{save_name}.png', imgs)

    def close(self):
        self.sess.close()
        self.sa.close()


if __name__ == '__main__':
    app = App()

    app.predict_samples(cate='train')
    # app.train()

    app.close()