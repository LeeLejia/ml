"""
    pandas是numpy的升级版
    两者都是c语言实现，矩阵运算，运算速度快
"""

import numpy as np

array = np.array(
    [[1, 2, 3],
     [4, 5, 6]]
)
print("创建一个矩阵：", array)
print("shape：", array.shape)    # shape： (2, 3)  2行3列
print("dim：", array.ndim)       # dim:  维度
print("size：", array.size)      # 元素个数

"""
    矩阵运算
"""
a = np.array([10, 20, 30, 40])
b = np.arange(4)    # 一个序号序列
print(a, b)
c = b**2  # 求平方
print("b**2", c)
c = 10*np.sin(a)  # 求sin
print("sin", c)
print("a 中小于3的元素", a < 20)

d = b.reshape((2, 2))   # 重新组合数据行列
print("重置b的形状\n", d)


a = np.array([[1, 2], [3, 4]])
b = np.arange(4).reshape((2, 2))
e = a*b   # 数乘，按数乘 得到的是数组
f = np.dot(a, b)  # 矩阵乘法，得到的是矩阵
g = a.dot(b)    # 同上
print("a,b数乘：\n", e, "\na,b矩阵相乘：\n", f, "\na,b矩阵相乘：\n", g)
