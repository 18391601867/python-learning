import numpy as np
'''查看数组的形状'''
# a=np.array([1,2,3,4,5,6,7,8,9,10])
# print(a.shape)     # (10,)  一维数组
#
# b=np.array([[1,2,3,4],[5,6,7,8]])
# print(b)
# print(b.shape)     # (2, 4) 二维数组， 2行 4 列

# c=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[3,4,5],[5,6,7],[7,8,9]]])
# print(c)
# print(c.shape)    # (2, 3, 3)   指的是 2 大板块， 每块 3行3 列
#

'''对数组进行修改'''
# c=c.reshape(3,6)
# print(c)
# print(c.shape)    # (3, 6) 修改后

# c=c.reshape(18,)
# print(c)
# print(c.shape)   # (18,)

# c=c.flatten()
# print(c)      # 未知数据个数时， 使用该方法直接 变为一维

'''对数组进行计算'''
# x=np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
# x=x+2
# print(x)     # 加法

# x=x*2
# print(x)     # 乘法

# x=x-2
# print(x)       # 减法

# x=x/2
# print(x)       # 除法

'''相同的数组形状(维度)'''
# x2=np.arange(100,124).reshape(4,6)
# x1=np.arange(1,25).reshape(4,6)
# print(x1)
# print(x2)
# # print(x1+x2)    # 对应位置相加
# print(x2-x1)     # 对应位置相减
# print(x2*x1)

'''不同的数组形状(维度)'''
# 如果有 相同的 行 或 列 就可以 计算

'''创建数组'''
# t1=np.array([1,2,3])
# print(t1)
# print(type(t1))     # ndarray 类型
#
# t2=np.array(range(20))
# print(t2)

# t3=np.arange(12)    # 快速生成数组
# print(t3)
# print(t3.dtype)

# t4=np.arange(20).reshape(4,5)
# print(t4)
# print(t4.dtype)     # int32   数据类型

'''numpy 中的数据类型'''
# t5=np.array(range(1,4),dtype='i1')
# print(t5)
# print(t5.dtype)

# t6=np.array([1,0,0,1,0],dtype=bool)
# print(t6)
# print(t6.dtype)

'''调整数据类型'''
# t7=t6.astype('int8')
# print(t7)
# print(t7.dtype)

# import random
# '''numpy 中的小数'''
# t8=np.array([random.random() for i in range(10)])
# print(t8)
# print(np.round(t8,3))
# print(t8.dtype)


'''读取文本文件的数据'''
# input_file='data.csv'
# # skiprows=1 跳过第一行文本内容
# t1=np.loadtxt(input_file,delimiter=',',skiprows=1)
# print(t1.shape)
# 转置
# t2=t1.T
# print(t2)
# print(t2.shape)

'''二维数组中，转置的方法'''
# t3=np.arange(24).reshape(4,6)
# print(t3)
# #第一种
# t4=t3.transpose()
# print(t4)
# # 第二种
# t5=t3.T
# print(t5)
# # 第三种
# t6=t3.swapaxes(1,0)    # 交换轴
# print(t6)

'''取行'''
# input_file='data.csv'
# t7=np.loadtxt(input_file,delimiter=',',skiprows=1)
# print("取之前：\n",t7)
# 取行
# print("取第三行:\n",t7[2])
# # 取连续的行
# print("连续多行\n",t7[10:])
# # 取不连续的行
# print("不连续：\n",t7[[2,4,6,8,10]])

#取列
# print(t7[1,:])
# print(t7[2:,:])
# print(t7[[2,4,6,8,10]])
# print(t7[:,0])

# 取连续的多列
# print(t7[:,1:])

# 取不连续的多列
# print(t7[:,[0,2]])

# print(t7[2:6,[0,2]])

# print(t7[2,[0,2]])


'''数组的拼接'''
# t1=np.array(range(12)).reshape(3,4)
# t2=np.array(range(12,24)).reshape(3,4)
# print(t1)
# print("*"*50)
# print(t2)
'''第一种：竖直拼接'''
# print(np.vstack((t1,t2)))

'''第二种：水平拼接'''
# print(np.hstack((t1,t2)))

'''数组的行 列交换'''
# t2[[1,2],:]=t2[[2,1],:]    # 行交换
# print(t2)

# t2[:,[0,3]]=t2[:,[3,0]]
# print(t2)            # 列交换


'''对两组数据进行拼接，合为一组数据'''
# # 读取数据
# input_file1='uk_data.csv'
# input_file2='us_data.csv'
# uk_data=np.loadtxt(input_file1,delimiter=',',dtype=int)
# us_data=np.loadtxt(input_file2,delimiter=',',dtype=int)
#
# # 构造数据
# # 全为 0 的数据
# zeros_data=np.zeros((uk_data.shape[0],1)).astype('int')
# ones_data=np.ones((us_data.shape[0],1)).astype('int')
# # print(ones_data)
#
# # 分别添加一组全为0 或 1 的数据
# uk_data=np.hstack((uk_data,zeros_data)).astype('int')
# # print(uk_data)
# us_data=np.hstack((us_data,ones_data)).astype('int')
# # print(us_data)
#
# # 拼接两组数据
# data=np.vstack((uk_data,us_data)).astype('int')
# print(data)


'''numpy的 其他方法'''
# 2*3 全为1 的数组
# x1=np.ones((2,3)).astype('int')
# print(x1)

# 3*3 全为 0 的数组
# x0=np.zeros((3,3)).astype('int')
# print(x0)

# 创建一个 对角线 为 1 的正方形数组（方阵）
# xx=np.eye(5).astype('int')
# print(xx)

# 获取最大值、 最小值的位置
# x2=np.eye(4).astype('int')
# print(x2)
# x2_max=np.argmax(x2,axis=0)    # axis=0: 行方向 每一列
# print(x2_max)

import random
# x3=np.array([random.random() for i in range(12)]).reshape(3,4)
# print(x3)
# x3_max=np.argmax(x3,axis=1)    # axis=1 :列方向 每一行
# x3_min=np.argmin(x3,axis=0)
# # print(x3_max)
# # print(x3_min)

'''随机分布'''
# x4=np.array([np.random.rand() for i in range(24)]).reshape(4,6)
# print(x4)

'''标准正态分布'''
# x5=np.array([np.random.randn() for i in range(100)]).reshape(10,10).astype('float')
# print(x5)

# x6=np.random.randint(10,20,(4,5))
# print(x6)

'''随机数种子'''
# np.random.seed(10)     # 相当于 固定第一次运行的随机数
# t=np.random.randint(10,20,(3,4))
# print(t)


'''numpy中的 nan 注意点'''
# # 1.两个 nan 不相等
# # print(np.nan==np.nan)
# # # 用途
# x1=np.random.randint(10,20,(3,4))
# x1[:,1]=0
# # x1=np.count_nonzero(x1)     # 非 0 个数
# # print(x1)
# #
# print(np.sum(x1))


'''对 nan 值的处理'''
# def fill_ndarray(t1):
#     # 遍历每一列
#     for i in range(t1.shape[1]):
#         temp_col=t1[:,i]    # 当前的 一列
#         nan_num=np.count_nonzero(temp_col!=temp_col)
#         print(nan_num)
#         if nan_num!=0:     # 不为 0 ，说明 当前一列中 有 nan
#             temp_not_nan_col=temp_col[temp_col==temp_col]    # 当前一列 不为 nan 的 array
#             temp_not_nan_col.mean()
#             # 选中当前为nan 的 位置， 把 值 赋值为 不为 nan 的均值
#             temp_col[np.isnan(temp_col)]=temp_not_nan_col.mean()
#     return t1
# if __name__ == '__main__':
#     t1 = np.arange(12).reshape((3, 4)).astype('float')
#     t1[1, 2:] = np.nan
#     print(t1)
#     t1=fill_ndarray(t1)
#     print(t1)


'''小练习'''
# import matplotlib.pyplot as plt
# input_file1='uk_data.csv'
# input_file2='us_data.csv'
# uk_data=np.loadtxt(input_file1,delimiter=',',dtype='int')
#
# # 取评论的数据
# uk_id=uk_data[:,0]
# uk_comments=uk_data[:,-1]
# print(uk_comments.max(),uk_comments.min())
#
# # 绘图
# plt.figure(figsize=(35,20),dpi=80)
#
# plt.bar(uk_id,uk_comments,alpha=0.5)
# plt.xticks(uk_id,fontsize=20)
# plt.yticks(fontsize=20)
# plt.show()


