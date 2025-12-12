import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ==========================================
# 圆的基本信息
# 1.圆半径
r = np.hypot(3.1 / 2.0, 1.5 / 2.0)
# 2.圆心坐标
a, b = (0., 0.)
# ==========================================
# 方法一：参数方程
theta = np.arange(0, 2 * np.pi, 0.01)
x = a + r * np.cos(theta)
y = b + r * np.sin(theta)
fig = plt.figure()
# fig, ax = plt.subplots()
axes = fig.add_subplot(111)
axes.plot(x, y)
# axes.axis('equal')
rect = Rectangle((-0.75, -1.55), 1.5, 3.1, fill=None, linewidth=2)
axes.add_patch(rect)
# plt.title('圆形绘制1')
plt.show()
