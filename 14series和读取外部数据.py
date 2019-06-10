'''pandas基础'''

'''Series创建'''
import pandas as pd
# # 最基本的 Series
# t1=pd.Series([11,21,31,41,51,61,71,81])
# print(t1)
# print(t1[t1>50])



# 自定义索引
# t2=pd.Series([11,21,31,41,51,61,71,81],index=list('abcdefgh'))
# print(t2)

# 以 字典的形式出现
# temp_dict={'name':'张三','age':80,'gender':'女','tel':10086}
# t3=pd.Series(temp_dict)
# print(t3)

# 字典推导式
# temp_dict1={i-1:i for i in range(10)}
# t4=pd.Series(temp_dict1).astype('float')
# print(t4)

# import string
# a_dict={string.ascii_uppercase[i]:i for i in range(20)}
# a=pd.Series(a_dict)
# print(a)

'''Series 切片和 索引'''
# temp_dict={'name':'张三','age':80,'gender':'女','tel':10086}
# t3=pd.Series(temp_dict)
# # 取出指定的 数据
# print(t3[['name','age']])
# print(t3.index)
# print(t3.values)

'''where 方法的使用'''
# a=pd.Series(range(10))
# print(a)
# # a=a.where(a>5)
# # 小于5 的 ，替换为 10
# a=a.where(a>5,10)
# print(a)

'''读取 外部文本文件'''
# input_file='data.csv'
# data=pd.read_csv(input_file)
# print(data)

# 读取 mongodb数据
# from pymongo import MongoClient
# client=MongoClient()
# collection=client['douban']['tv1']
# data=list(collection.find())
# t1=data[0]
# t1=pd.Series(t1)
# print(t1)







