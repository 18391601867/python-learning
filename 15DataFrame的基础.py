

'''创建 DATaFrame'''
import pandas as pd
import numpy as np
# t=pd.DataFrame(np.arange(20,44).reshape(4,6))
# print(t)    # 二维数据， 既有行索引， 又有列索引
# # 自定义行列 索引
# t1=pd.DataFrame(np.arange(20,44).reshape(4,6),index=list('abcd'),columns=list('abcdef'))
# print(t1)    # 二维数据， 既有行索引， 又有列索引

# t2={'nae':['张三','李四'],'age':[18,22],'tel':['10010','10086']}
# d2=pd.DataFrame(t2)
# print(d2)

# t3=[{'name':'张三','age':22,'tel':'10010'},{'name':'李四','tel':'10086'},{'name':'马云','age':54}]
# d3=pd.DataFrame(t3)
# print(d3)

'''读取外部文本数据'''
# input_file='supplier_data.csv'
# data=pd.read_csv(input_file,delimiter=',')
# # 取出前五行
# print(data.head())
# # 取出后五行
# print(data.tail())
# # 打印 概述信息
# print(data.info())
# # 快速统计信息
# print(data.describe())

# # 找出消费最高的前五个
# data=data.sort_values(by='Cost',ascending=False)        # ascending=False倒序 ；by='Cost' 以“Cost”进行排序
# print(data.head())

# #获取某行 或 某列 数据
# # 前五行
# data=data[:5]
# print(data)
# # 获取第三列数据
# data=data['Part Number']
# print(data)
# # 获取第三列 前五行
# data=data[:5]['Part Number']
# print(data)

'''pandas中 loc 和 iloc'''
# t1=pd.DataFrame(np.arange(24).reshape(4,6),index=list('abcd'),columns=list('defghi'))
''' loc 通过 标签获取数据'''
# print(t1)
# # 获取 多行 e 列
# print(t1.loc[:,'e'])
# # 获取多行 e列 ，g 列
# print(t1.loc[:,['e','g']])

# # 获取 b 行
# print(t1.loc['b',:])
# # 获取 b行，d行
# print(t1.loc[['b','d']])
# # 获取 b行 多列
# print(t1.loc[['b'],:])
# # 获取 b行 ，d行， g 列
# print(t1.loc[['b','d'],'g'])
# # 获取 b行 ，d行， g 列 i 列
# print(t1.loc[['b','d'],['g','i']])

''' iloc 通过 位置获取数据'''
# # 获取第二行
# print(t1.iloc[1])
# # 获取 第一行、第三行数据
# print(t1.iloc[[0,2]])
# # 获取第 一行 第三列、 第五列数据
# print(t1.iloc[[0],[2,4]])
# # 获取 第一行、第三行 第二列、第三列、第六列数据
# print(t1.iloc[[0,2],[1,3,5]])

# # 获取第 二列 数据
# print(t1.iloc[:,1])
# # 获取 第二列 、第四列数据
# print(t1.iloc[:,[1,3]])

# 赋值更改数据
# t1.iloc[0:3,1]=30
# print(t1)


'''pandas 中字符串的使用方法'''
# input_file='supplier_data.csv'
# data=pd.read_csv(input_file,delimiter=',')
# data=pd.DataFrame(data)
# print(data)
# tolist() 将 一个 series 转化为 list
# data=data['Purchase Date'].str.split('/').tolist()
# data=data['Purchase Date'].str.replace('/','-')
# print(data)

# # 找出 Part Number 大于 7000 的 或者 小于 3000的
# data=data[(data['Part Number']<3000)|(data['Part Number']>7000)]
# print(data)

'''pandas 缺失数据的处理'''

t2=[{'name':'张三','age':18,'gender':'男','tel':'10010'},{'name':'李四','age':22,'tel':'10086'},
    {'name':'马云','gender':'女'},{'name':'艾琳','gender':'女','age':28}]
data_t2=pd.DataFrame(t2)
# 判断 数据是否 为空
# print(pd.isnull(data_t2))
# print(pd.notnull(data_t2))

# 选择 不为 空的 行; 删掉 里面含有 nan 的 数据
# print(data_t2.dropna(axis=0,how='any'))
# print(data_t2.dropna(axis=0))
# 原地修改
# data_t2=data_t2.dropna(axis=0,how='any',inplace=False)
# print(data_t2)

# 填充 nan
# print(data_t2.fillna(100))

# 按指定 对象填充
data_t2['age']=data_t2['age'].fillna(data_t2['age'].mean())
print(data_t2)
data_t2['gender']=data_t2['gender'].fillna('男')
print(data_t2)

