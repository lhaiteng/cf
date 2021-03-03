# -*- coding: utf-8 -*-
import numpy as np


class LR_decay:
    def __init__(self, start_lr=0.0001, lr=0.01, end_lr=0.001, warm_up_stage=0, cycle=True,
                 decay_steps=50, decay_rate=0.9, power=0.5,
                 epochs=100):
        self.warm_up_stage = warm_up_stage
        self.start_lr = start_lr
        self.lr = lr
        self.end_lr = end_lr
        self.cycle = cycle
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.power = power
        self.epochs = epochs - warm_up_stage

    def get_lr(self, step, cate='cosine_decay'):
        if step < self.warm_up_stage:
            return self.start_lr + (self.lr - self.start_lr) * step / self.warm_up_stage
        step -= self.warm_up_stage
        if cate == 'exponential_decay':
            return self.exponential_decay(step)
        if cate == 'natural_exponential_decay':
            return self.natural_exponential_decay(step)
        if cate == 'poly_decay':
            return self.poly_decay(step)
        if cate == 'cosine_decay':
            return self.cosine_decay(step)
        if cate == 'cosine_decay2':
            return self.cosine_decay2(step)

    def exponential_decay(self, step):
        return self.lr * self.decay_rate ** (step / self.decay_steps)

    def natural_exponential_decay(self, step):
        return self.lr * np.exp(-self.decay_rate * step / self.decay_steps)

    def poly_decay(self, step):
        if self.cycle:
            decay_steps = self.decay_steps * np.ceil(step / self.decay_steps)
            ratios = step / decay_steps
        else:
            steps = np.minimum(step, self.decay_steps)
            ratios = steps / self.decay_steps
        ans = (self.lr - self.end_lr) * (1 - ratios) ** self.power + self.end_lr
        return ans

    def cosine_decay(self, step):
        alpha = self.end_lr / self.lr
        if self.cycle:
            if step == 0:
                ratios = 0
            else:
                decay_steps = self.decay_steps * np.ceil(step / self.decay_steps)
                ratios = step / decay_steps
        else:
            steps = np.minimum(step, self.decay_steps)
            ratios = steps / self.decay_steps

        cos_decay = 0.5 * (1 + np.cos(np.pi * ratios))
        decayed = (1 - alpha) * cos_decay + alpha
        return self.lr * decayed

    def cosine_decay2(self, step):
        # 从开始到结束是一条余弦曲线
        return self.end_lr + 0.5 * (self.lr - self.end_lr) * (1 + np.cos(step / (self.epochs - 1) * np.pi))
