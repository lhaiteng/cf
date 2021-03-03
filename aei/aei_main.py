# -*- coding: utf-8 -*-
from aei.aei_config import aei_cfg as cfg
from aei.aei_tensors import AEI
from aei.aei_samples import AEISamples as Samples
import tensorflow as tf
import numpy as np
import cv2, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


class App:
    def __init__(self):
        cfg.check_all_dir_paths()
        self.sa = Samples()
        graph = tf.Graph()
        with graph.as_default():
            self.ts = AEI()
            conf = tf.ConfigProto()
            conf.allow_soft_placement = True
            self.sess = tf.Session(config=conf)

            # 保存器和模型恢复内容
            ts_paths = []  # (ts, save_path)用来恢复模型指定变量的
            # arcface
            arcface_restore_list = [v for v in tf.global_variables() if 'arcface' in v.name]
            print(f'arcface_restore_list:\n{arcface_restore_list}')
            ts_paths.append((arcface_restore_list, cfg.arcface_save_path))
            # unet
            unet_restore_list = [v for v in tf.global_variables() if 'unet' in v.name]
            print(f'unet_restore_list:\n{unet_restore_list}')
            ts_paths.append((unet_restore_list, cfg.unet_save_path))
            # aei
            self.saver_aei = tf.train.Saver(var_list=self.ts.vars_save, max_to_keep=5)
            self.saver_aei.export_meta_graph(cfg.save_path + '.meta')
            print(f'SUCCEED: export_meta_graph to {cfg.save_path}.meta')
            ts_paths.append((self.ts.vars_restore, cfg.save_path + f'-{cfg.restore_epoch}'))

            # 模型初始化、恢复指定变量
            self.init_restore_model(ts_paths)

    def init_restore_model(self, ts_paths):
        # 先初始化所有变量
        print('START: init all vars.')
        self.sess.run(tf.global_variables_initializer())

        for var_restore_list, path in ts_paths:
            print('-' * 50)
            # print(f'path: {path}\nf var_restore_list: {var_restore_list}')
            try:
                saver = tf.train.Saver(var_list=var_restore_list)
                saver.restore(self.sess, path)
                print(f'SUCCEED: restore model from {path}.')
            except:
                print(f'FAILED: restore model from {path}.')
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
        self.print_train_finfos(batch_size, num_batch, num_epoch, num_print_loss, show_num, start_epoch, train_batch)

        # 令步数从0开始计算
        self.sess.run(tf.assign(self.ts.global_step, 0))
        step = 0
        run_loss = [self.ts.loss_adv_d,
                    self.ts.loss_adv_g, self.ts.loss_att, self.ts.loss_id, self.ts.loss_rec]
        feed_dict = {self.ts.keep_prob: cfg.aei_keep_prob, self.ts.training: True}
        for epoch in range(start_epoch, num_epoch):
            lr = cfg.aei_lr[epoch % 100]
            for batch in range(num_batch):
                # 整除时显示预测图片，但不保存
                if show_num and step % show_num == 0:
                    self.predict_samples(show_pic=True, save_pic=False)

                # 训练train_batch次，得到损失和梯度
                losses = self._get_grads_infos(step, batch_size, train_batch, run_loss, feed_dict)
                # 更新梯度
                self.sess.run([self.ts.train_d, self.ts.train_g],
                              {self.ts.lr: lr, self.ts.train_num_batch: num_batch})
                # 显示结果
                if not num_print_loss or step % num_print_loss == 0:
                    # 求各batch的损失信息均值
                    losses /= num_batch
                    print(f'\rEPOCH {epoch}/{num_epoch} BATCH {batch}/{num_batch} '
                          f'LOSS adv_d={losses[0]:.3f} adv_g={losses[1]:.3f} '
                          f'att={losses[2]:.3f} id={losses[3]:.3f} rec={losses[4]:.3f} ',
                          end='')
                # 更新步数
                self.sess.run(self.ts.add_global_step)
                step = self.sess.run(self.ts.global_step)

            # 每代结束后，进行预测并保存预测图片，但不显示
            self.predict_samples(show_pic=False, save_pic=True, save_name=f'{cfg.version}_E{epoch}')
            # 每代结束后，保存模型
            self.saver_aei.save(self.sess, cfg.save_path, write_meta_graph=False,
                                global_step=epoch)
            print(f'\tSave into model {cfg.save_path}')

    def print_train_finfos(self, batch_size, num_batch, num_epoch, num_print_loss, show_num, start_epoch, train_batch):
        print(f'Start training from epoch No.{start_epoch}...')
        print(f'Train pics {self.sa.train_num}.')
        print(f'Train num_epoch: {num_epoch}')
        print(f'Train num_batch: {num_batch}.')
        print(f'Train batch_size: {cfg.max_batch_size} -> train {batch_size} pics for {train_batch} times.')
        print(f'Show loss every {num_print_loss} batch / epoch.')
        print(f'Show predict every {show_num} batch, '
              f'after train {show_num * batch_size * cfg.gpu_num * train_batch} pics.')
        print(f'Train D every {cfg.num_d} times.')

    def _get_grads_infos(self, step, batch_size, train_batch, run_loss, feed_dict):
        # 把损失、grads初始化
        infos = None
        self.sess.run([self.ts.assign_zero_grads_d, self.ts.assign_zero_grads_g])
        # 计算多个batch的损失和grad_var
        for _ in range(train_batch):
            # 喂入各gpu值
            self.update_feed_dict(feed_dict, batch_size, cate='train')
            # 累加梯度，计算损失
            if step % cfg.num_d == 0:  # cfg.num_g个step训练d和g
                res = self.sess.run(run_loss + [self.ts.assign_grads_d, self.ts.assign_grads_g], feed_dict)
            else:  # 其余情况只训练d
                res = self.sess.run(run_loss + [self.ts.assign_grads_d], feed_dict)
            # # d, g, att, id, rec, _, _
            if infos is None:
                infos = np.array(res[:5])
            else:
                infos += np.array(res[:5])
        return infos

    def update_feed_dict(self, feed_dict, batch_size, cate='train'):
        for gpu_index in range(cfg.gpu_num):
            sub_ts = self.ts.sub_ts[gpu_index]
            srcs, targets, is_recs = self.sa.next_batch(batch_size, cate)
            feed_dict.update({sub_ts.src: srcs, sub_ts.target: targets, sub_ts.is_rec: is_recs})

    def predict_samples(self, show_pic=True, save_pic=False, save_name=None):
        if save_name is None:
            save_name = cfg.name
        samples = self.sa
        sub_ts = self.ts.sub_ts[0]

        run_list = sub_ts.st
        # 从训练集中抽取
        src, target, is_rec = samples.next_batch(1, 'train')
        img1 = self._predict_samples(sub_ts, run_list, src, target, is_rec)
        # 从测试集中抽取
        src, target, is_rec = samples.next_batch(1, 'test')
        img2 = self._predict_samples(sub_ts, run_list, src, target, is_rec)
        imgs = np.concatenate([img1, img2], axis=0).astype(np.uint8)
        # 显示图像
        if show_pic:
            plt.figure(figsize=[18, 18])
            plt.imshow(imgs[:, :, ::-1])
            plt.show()
        # 保存图像
        if save_pic:
            cv2.imwrite(cfg.gen_dir + f'{save_name}.png', imgs)

    def _predict_samples(self, sub_ts, run_list, src, target, is_rec):
        st = self.sess.run(run_list, {self.ts.keep_prob: 1, self.ts.training: False,
                                      sub_ts.src: src, sub_ts.target: target, sub_ts.is_rec: is_rec})
        img = np.concatenate([src[0], target[0], st[0] * 255], axis=1)
        return img

    def close(self):
        self.sess.close()
        self.sa.close()


if __name__ == '__main__':
    ## 设置属性防止中文乱码
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    app = App()

    # 预测图片看是否显示正常
    app.predict_samples(show_pic=True, save_pic=False)
    app.train()

    # app.predict_samples()

    app.close()
    print('Finished!')
