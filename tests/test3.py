# -*- coding: utf-8 -*-
"""
iou和giou的对比
boxes 100x100，保持iou不变，改变预测框位置
"""

import turtle  # 引入turtle模块
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def draw_rec(a=100, b=100):
    turtle.fd(a)
    turtle.right(90)
    turtle.fd(b)
    turtle.right(90)
    turtle.fd(a)
    turtle.right(90)
    turtle.fd(b)
    turtle.right(90)


# 海龟设置
turtle.hideturtle()  # 隐藏箭头
turtle.speed(10)  # 设置速度
turtle.pu()

# 左移距离0~100
x = 80

# 向左平移
turtle.color('red', 'yellow')  # 设置绘制的颜色和填充颜色
turtle.goto(0, 200)
turtle.pd()
turtle.begin_fill()  # 开始填充位置
draw_rec()
turtle.end_fill()
turtle.pu()

turtle.color('grey', 'grey')  # 设置绘制的颜色和填充颜色
turtle.goto(x, 200)
turtle.pd()
draw_rec()
turtle.pu()
AB = (100 - x) * 100
A_B = 100 * 100 * 2 - AB
C = (100 + x) * 100
iou = AB / A_B
Giou = iou - (C - A_B) / C
print(f'平移时, iou={iou} Giou={Giou}')

# 向右下平移，保持iou不变
turtle.color('red', 'yellow')  # 设置绘制的颜色和填充颜色
turtle.goto(0, 0)
turtle.pd()
turtle.begin_fill()  # 开始填充位置
draw_rec()
turtle.end_fill()
turtle.pu()

turtle.color('grey', 'grey')  # 设置绘制的颜色和填充颜色
d = math.sqrt(AB)
turtle.goto(100 - d, -100 + d)
turtle.pd()
draw_rec()
turtle.pu()

AB = d * d
A_B = 100 * 100 * 2 - AB
C = (200 - d) ** 2
iou = AB / A_B
Giou = iou - (C - A_B) / C
print(f'向右下移动时, iou={iou} Giou={Giou}')

turtle.done()

ious, Gious = [], []
for x in range(101):
    AB = (100 - x) * 100
    A_B = 100 * 100 * 2 - AB
    ious.append(AB / A_B)

    d = math.sqrt(AB)
    AB = d * d
    A_B = 100 * 100 * 2 - AB
    C = (200 - d) ** 2
    iou = AB / A_B
    Gious.append(iou - (C - A_B) / C)
plt.plot(range(101), ious, label='向右平移')
plt.plot(range(101), Gious, label='右下平移')
plt.title('不同平移方式下，iou相同，而Giou的不同变化')
plt.legend()
plt.show()



