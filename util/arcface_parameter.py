# -*- coding: utf-8 -*-

class ArcPara:
    def __init__(self, scale_start=64, scale_end=64, scale_steps=50,
                 m1_start=1, m1_end=1.1, m1_steps=10,
                 m2_start=0.01, m2_end=0.15, m2_steps=10,
                 m3_start=0.1, m3_end=0.3, m3_steps=10):
        # arcface参数 标签项改为(cos(m1*thetas+m2) - m3)，其余不变
        self.scale_start, self.scale_end, self.scale_steps = scale_start, scale_end, scale_steps
        self.m1_start, self.m1_end, self.m1_steps = m1_start, m1_end, m1_steps
        self.m2_start, self.m2_end, self.m2_steps = m2_start, m2_end, m2_steps
        self.m3_start, self.m3_end, self.m3_steps = m3_start, m3_end, m3_steps

    def get_para(self, step):
        scale = self.scale_start + (self.scale_end - self.scale_start) * min(step / self.scale_steps, 1)
        m1 = self.m1_start + (self.m1_end - self.m1_start) * min(step / self.m1_steps, 1)
        m2 = self.m2_start + (self.m2_end - self.m2_start) * min(step / self.m2_steps, 1)
        m3 = self.m3_start + (self.m3_end - self.m3_start) * min(step / self.m3_steps, 1)
        return scale, m1, m2, m3

