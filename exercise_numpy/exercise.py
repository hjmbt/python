import numpy as np

# 随机生产[2,4]的数组 ,均值1,7 方差0.1
arr = np.random.normal(1.7, 0.1, (2, 4))
# 数组重排列
arr = arr.reshape(4, 2)
# 定义一个权重数组
q = [0.1, 0.9]
# 使用权重方法
arr_dot = np.dot(arr, q)
# 定义一个全部为0的数组
arr_zero = np.zeros([4, 3])
# 定义一个全部为1的数组
arr_one = np.ones([4, 3])
# 将数组中所有的数都加5
arr_add_five = arr_one + 5
# 垂直拼接
arr_vstack = np.vstack((arr_one, arr_zero))
# 水平拼接
arr_hstack = np.hstack((arr_one, arr_zero))
# 均值
arr_mean = np.mean(arr)
# 轴最大值
arr_max = np.amax(arr, 1)
# 轴最小值
arr_min = np.amin(arr, 0)
# 条件判断
arr_big = arr > 1.7
# 三目运算
arr_3_big = np.where(arr > 1.7, 1, 0)

# print(arr)
# print(arr_dot)
# print(arr_zero)
# print(arr_one)
# print(arr_add_five)
# print(arr_vstack)
# print(arr_hstack)
# print(arr_mean)
# print(arr_max)
# print(arr_min)
# print(arr_big)
# print(arr_3_big)
