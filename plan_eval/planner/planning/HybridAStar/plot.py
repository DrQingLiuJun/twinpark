import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 统一设置
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size":15,  # 设置字体大小
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)


# 生成数据
x1 = np.linspace(-40, 0, 400)
y1 = 35 * (x1 / -40)

x2 = np.linspace(0, 70, 400)
y2 = 70 - (x2 ** 2 / 70)

theta = np.linspace(0, 4 * np.pi, 400)
r = theta ** 2
x3 = r * np.cos(theta)
y3 = r * np.sin(theta)

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 第一张图
axes[0].plot(x1, y1)
axes[0].set_xlabel('x-axis [m]',fontsize=18,fontfamily='Times New Roman')
axes[0].set_ylabel('y-axis [m]',fontsize=18,fontfamily='Times New Roman')

# 第二张图
axes[1].plot(x2, y2)
axes[1].set_xlabel('x-axis [m]',fontsize=18,fontfamily='Times New Roman')
axes[1].set_ylabel('y-axis [m]',fontsize=18,fontfamily='Times New Roman')

# 第三张图
axes[2].plot(x3, y3)
axes[2].set_xlabel('x-axis [m]',fontsize=18,fontfamily='Times New Roman')
axes[2].set_ylabel('y-axis [m]',fontsize=18,fontfamily='Times New Roman')

# 保存图像
plt.tight_layout()
plt.savefig('resampled_plots.png', dpi=300)  # 设置更高的dpi以提高清晰度
plt.show()
