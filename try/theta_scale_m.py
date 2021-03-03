# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""
带有scale m1 m2 m3 的交叉熵损失和对theta、untheta的导数
"""


class APP:
    def __init__(self):
        # # thetas, scale, m1, m2, m3
        # # x-thetas, y-m2
        # nlevel = 15
        # # x轴
        # thetas = np.tile(np.arange(315) / 100, (700, 1))
        # # y轴
        # y = np.tile(np.arange(0, 70, 0.1).reshape([-1, 1]), (1, 315))
        # loss1 = self.get_loss(thetas, scale=y, m1=1.15, m2=0.2, m3=0.8)
        # L_t, L_ut = self.get_L(thetas, scale=y, m1=1.15, m2=0.2, m3=0.8)
        # self.plot_contour(thetas, y, loss1, 'loss1', nlevel)
        # self.plot_contour(thetas, y, L_t, 'L_t', nlevel)
        # self.plot_contour(thetas, y, L_ut, 'L_ut', nlevel)

        # x轴
        theta = np.arange(315) / 100
        loss = self.get_loss(theta, scale=10, m1=1, m2=0.1, m3=0.5)
        L_t, L_ut = self.get_L(theta, scale=10, m1=1, m2=0.1, m3=0.5)
        for y in (L_t, L_ut):
            plt.plot(theta, y)
            plt.xlim([0, 3.15])
            plt.grid()
            plt.show()

        # x轴
        theta = np.arange(315) / 100
        loss = self.get_loss(theta, scale=10, m1=1.1, m2=0.1, m3=0.7)
        L_t, L_ut = self.get_L(theta, scale=10, m1=1.1, m2=0.1, m3=0.7)
        for y in (L_t, L_ut):
            plt.plot(theta, y)
            plt.xlim([0, 3.15])
            plt.grid()
            plt.show()

        # x轴
        theta = np.arange(315) / 100
        loss = self.get_loss(theta, scale=10, m1=1.15, m2=0.15, m3=0.9)
        L_t, L_ut = self.get_L(theta, scale=10, m1=1.15, m2=0.15, m3=0.9)
        for y in (L_t, L_ut):
            plt.plot(theta, y)
            plt.xlim([0, 3.15])
            plt.grid()
            plt.show()

    def get_softmax(self, theta, un_theta=1.57, n=234, scale=10, m1=1, m2=0.2, m3=0.5):
        fz = np.exp(scale * np.cos(m1 * theta + m3) - m2)
        fm = fz + (n - 1) * np.exp(scale * np.cos(un_theta))
        return fz / fm, np.exp(scale * np.cos(un_theta)) / fm

    def get_loss(self, theta, un_theta=1.57, n=234, scale=10, m1=1, m2=0.2, m3=0.5):
        si, sj = self.get_softmax(theta, un_theta, n, scale, m1, m2, m3)
        return -np.log(si)

    def get_L(self, theta, un_theta=1.57, n=234, scale=10, m1=1, m2=0.2, m3=0.5):
        # 向量s的softamx得到的第i项概率为si，对i求导是si(1-si)，对j求导是-si*sj
        si, sj = self.get_softmax(theta, un_theta, n, scale, m1, m2, m3)
        L_theta = scale * m1 * (1 - si) * np.sin(m1 * theta + m3)
        L_un_theta = -scale * sj * np.sin(un_theta)
        return L_theta, L_un_theta

    def plot_contour(self, x, y, h, title, levels=10, xname='thetas', yname='scale'):
        # 颜色集，6层颜色，默认的情况不用写颜色层数,
        cset = plt.contourf(x, y, h, levels, cmap=plt.cm.hot)  # or cmap='hot'
        # 画出8条线，并将颜色设置为黑色
        contour = plt.contour(x, y, h, levels)
        # 等高线上标明z（即高度）的值，字体大小是10，颜色分别是黑色和红色
        plt.clabel(contour, fontsize=10, fmt='%.1f')  # colors=('k','r')
        # 去掉坐标轴刻度
        # plt.xticks(())
        # plt.yticks(())
        # 设置颜色条，（显示在图片右边）
        plt.colorbar(cset)
        # 显示
        plt.title(f'{title} - untheta=1.57')
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    APP()
