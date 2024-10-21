## 更改结果图颜色为蓝绿色，增加极值显示，美化结果图

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数定义
L = 10.0  # 梁的长度
T = 1.0  # 总时间
Nx = 100  # 空间网格数
Nt = 500  # 时间步数
dx = L / (Nx - 1)  # 空间步长
dt = T / Nt  # 时间步长
alpha = 2.546  # 弯曲刚度
K = 50   # 控制增益

# 外部激励函数
def external_force(i, n):
    """外部激励，作为时间和空间的函数"""
    return np.sin(np.pi * i * dx) * np.cos(2 * np.pi * n * dt)

# 初始化位移数组
x_values = np.linspace(0, L, Nx)
time_values = np.linspace(0, T, Nt)
omega_controlled = np.zeros((Nx, Nt))  # 有控制位移

# 设置初始条件：omega[:, 0] 为初始位移，omega[:, 1] 为初始速度导致的位移
omega_controlled[:, 0] = np.sin(np.pi * x_values)  # 初始位移为正弦分布
omega_controlled[:, 1] = omega_controlled[:, 0]  # 初始速度为0

# 计算有控制的情况
for n in range(1, Nt - 1):
    for i in range(2, Nx - 2):
        omega_xx = (omega_controlled[i - 2, n] - 4 * omega_controlled[i - 1, n] +
                     6 * omega_controlled[i, n] - 4 * omega_controlled[i + 1, n] +
                     omega_controlled[i + 2, n]) / dx ** 4
        control_force = -K * omega_controlled[i, n]  # 比例反馈控制力
        omega_controlled[i, n + 1] = (2 * omega_controlled[i, n] - omega_controlled[i, n - 1] +
                                       dt ** 2 * (-alpha * omega_xx + external_force(i, n) + control_force))

# 创建3D图
X, Y = np.meshgrid(time_values, x_values)  # X表示时间，Y表示梁上的位置
Z = omega_controlled  # Z表示位移

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲面图
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.5, antialiased=True)

# 添加颜色条
fig.colorbar(surface, shrink=0.5, aspect=10)

# 找到位移最大值的位置
max_value = np.max(omega_controlled)  # 找到位移的最大值
max_index = np.unravel_index(np.argmax(omega_controlled), omega_controlled.shape)  # 找到最大值的位置
max_x = time_values[max_index[1]]  # 最大值对应的时间
max_y = x_values[max_index[0]]  # 最大值对应的位置

# 在图中标注最大值
ax.scatter(max_x, max_y, max_value, color='r', s=50, alpha=1, edgecolor='black', label=f'Max Value: {max_value:.2f}')  # 设置尺寸和透明度

# 在图中添加文字标签
ax.text(max_x + 0.1, max_y, max_value + 0.1,
        f'Max: {max_value:.2f}', color='black', alpha=1, fontsize=12)

# 设置标签和标题
ax.set_xlabel('Time')
ax.set_ylabel('Position along beam')
ax.set_zlabel('Displacement')
ax.set_title(f'Controlled Vibration (K={K}) over Time and Position')

# 提高图像质量，设置视角
ax.view_init(elev=30, azim=120)  # 设置视角，仰角30度，方位角120度

# 显示图例
ax.legend()

plt.show()
