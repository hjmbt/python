# coding=utf-8
import exercise_numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 10)
# x = [1, 2, 3]
# 反转 reversed 返回的是一个 iterator 迭代器
# ? 为什么x.reverse() 可以输出但是没法用于画图 因为list.reverse()结果为null 他是直接作用在原有的list上的
y = list(reversed(x))
# 标题
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.title(u'测试画图')
# 设置X轴标签
plt.xlabel('X')
# 设置Y轴标签
plt.ylabel('Y')
# 点图
plt.scatter(x, y)
# 折线图
# plt.plot(x, y)

# 画图
plt.show()
