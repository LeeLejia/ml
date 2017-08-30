#numpy模块，pandas模块
##为什么使用numpy
>
运算速度快：numpy 和 pandas 都是采用 C 语言编写, pandas 又是基于 numpy, 是 numpy 的升级版本。
消耗资源少：采用的是矩阵运算，会比 python 自带的字典或者列表快好多
>
##安装
>
sudo apt-get install python-numpy  
sudo apt-get install python-pandas
>
##numpy基础内容
####1.numpy属性
>
ndim：维度  
shape：行数和列数  
size：元素个数
>
<pre><code>
array = np.array([[1,2,3],[4,5,6]])  #列表转化为矩阵
print(array)
print('dim:',array.ndim)    # 维度 dim: 2
print('shape:',array.shape) # 行数和列数 shape:(2, 3)
print('size:',array.size)   # 元素个数 size: 6
</code></pre>
####2.numpy创建array
>
array：创建数组  
dtype：指定数据类型  
zeros：创建数据全为0  
ones：创建数据全为1  
empty：创建数据接近0  
arrange：按指定范围创建数据  
linspace：创建线段  
>
<pre><code>
# 一维列表
a = np.array([2,23,4])
a = np.array([2,23,4],dtype=np.int)
print(a.dtype) # int 64
a = np.array([2,23,4],dtype=np.int32)
print(a.dtype) # int32
# 二维矩阵
a = np.array([[2,23,4],[2,32,4]])  # 2行3列
a = np.zeros((3,4)) # 数据全为0，3行4列
a = np.ones((3,4),dtype = np.int)   # 数据为1，3行4列
a = np.empty((3,4)) # 数据为empty（每个值都是接近于零的数），3行4列
"""
array([[  0.00000000e+000,   4.94065646e-324,   9.88131292e-324,
          1.48219694e-323],
       [  1.97626258e-323,   2.47032823e-323,   2.96439388e-323,
          3.45845952e-323],
       [  3.95252517e-323,   4.44659081e-323,   4.94065646e-323,
          5.43472210e-323]])
"""
a = np.arange(10,20,2) # 10-19 的数据，2步长
a = np.arange(12).reshape((3,4))    # 3行4列，0到11
"""
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
"""
a = np.linspace(1,10,20)    # 开始端1，结束端10，且分割成20个数据，生成线段
"""
array([  1.        ,   1.47368421,   1.94736842,   2.42105263,
         2.89473684,   3.36842105,   3.84210526,   4.31578947,
         4.78947368,   5.26315789,   5.73684211,   6.21052632,
         6.68421053,   7.15789474,   7.63157895,   8.10526316,
         8.57894737,   9.05263158,   9.52631579,  10.        ])
"""
a = np.linspace(1,10,20).reshape((5,4)) # 更改shape
# 获取随机的二维矩阵
a=np.random.random((2,4))
# array([[ 0.94692159,  0.20821798,  0.35339414,  0.2805278 ],
#       [ 0.04836775,  0.04023552,  0.44091941,  0.21665268]])
</code></pre>
####3.numpy的基础运算
<pre><code>
# 定义np对象
a=np.array([10,20,30,40])
b=np.arange(4)
# 四则运算
c=a-b   # array([10, 19, 28, 37])
c=a+b   # array([10, 21, 32, 43])
c=a*b   # array([  0,  20,  60, 120])
# 平方
c=b**2  # array([0, 1, 4, 9])
</pre></code>
####4.numpy的矩阵运算
>dot 矩阵乘法
<pre><code>
a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape((2,2))
# 矩阵乘法
c_dot = np.dot(a,b)
# 或
c_dot_2 = a.dot(b)
</pre></code>
####5.numpy的比较
<pre><code>
b=np.arange(4)
print(b<3)  
# array([ True,  True,  True, False], dtype=bool)
</code></pre>
####6.numpy的元素函数
>sum,min,max求元素和及最值，axis声明行列运算（0列，1行）
<pre><code>
np.sum(a)   # 求元素之和
np.min(a)   # 求元素中的最小值
np.max(a)   # 求元素中的最大值
# axis属性：值为0，将以列作为查找单元
# 值为1的时候，将以行作为查找单元
# a = [[ 0.23651224  0.41900661  0.84869417  0.46456022]
     [ 0.60771087  0.9043845   0.36603285  0.55746074]]
print("sum =",np.sum(a,axis=1))
# sum = [ 1.96877324  2.43558896]
print("min =",np.min(a,axis=0))
# min = [ 0.23651224  0.41900661  0.36603285  0.46456022]
print("max =",np.max(a,axis=1))
# max = [ 0.84869417  0.9043845 ]
</code></pre>
####6.numpy的索引函数
>argmin,argmax分别求最小元素和最大元素的索引
<pre><code>
# A = array([[ 2, 3, 0, 5]
#        [ 6, 7, 8, 9]
#        [10,11,19,13]])
print(np.argmin(A))    # 2
print(np.argmax(A))    # 10
</code></pre>
####7.numpy的函数
>sin,cos,average,mean,median  
>cumsum,diff,nonzero,sort  
>transpose,xx.T,clip
<pre><code>
# 求正弦
c=10*np.sin(a) 
# array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])
#求均值
A = np.arange(2,14).reshape((3,4)) 
# array([[ 2, 3, 4, 5]
#        [ 6, 7, 8, 9]
#        [10,11,12,13]])
print(np.average(A))     # 7.5
print(np.mean(A))        # 7.5
print(A.mean())          # 7.5
print(A.median())        # 7.5 求解中位数
# 累加函数 (一维)
print(np.cumsum(A))        
# [2 5 9 14 20 27 35 44 54 65 77 90]
# 累差函数 (多维,原一维-1) 每一行中后一项与前一项之差
print(np.diff(A))        
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]]
# 获取非零元素的行列坐标（分别放在两个数组中）
print(np.nonzero(A))    
# (array([0,0,0,0,1,1,1,1,2,2,2,2]),array([0,1,2,3,0,1,2,3,0,1,2,3]))

A = np.arange(14,2, -1).reshape((3,4)) 
# array([[14, 13, 12, 11],
#       [10,  9,  8,  7],
#       [ 6,  5,  4,  3]])
# 按行排序
print(np.sort(A))    
# array([[11,12,13,14]
#        [ 7, 8, 9,10]
#        [ 3, 4, 5, 6]])
# 矩阵的转置
print(np.transpose(A))    
print(A.T)
# array([[14,10, 6]
#        [13, 9, 5]
#        [12, 8, 4]
#        [11, 7, 3]])
# 裁剪元素值,小于或大于指定值的元素分别设置为指定值
print(np.clip(A,5,9))    
# array([[ 9, 9, 9, 9]
#        [ 9, 9, 8, 7]
#        [ 6, 5, 5, 5]])
</pre></code>