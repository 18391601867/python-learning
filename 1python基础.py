'''
一、基本数据类型
'''
# 1.字符串
# print("Output:{0:s}".format('I\'m enjoying learning Python'))
# print("Output:{0:s}".format('''You can use triple single quotes for multi-line comment strings.'''))
# 1) 和数值类型一样，也有很多标准模块、内置函数和操作符可以用来管理字符串。常用的操 作符和函数包括 +、* 和 len。
# string1='This is a '
# string2="short string"
# sentence=string1+string2
# print("Output:{0:s}".format(sentence))
# print("Output:{0:s} {1:s} {2:s}".format("She is","very"*3,"beautiful."))
# length=len(sentence)
# print("Output:{0:d}".format(length))
# 2) 字符串函数方法
# string1="My deliverable is due in May"
# string1_list1=string1.split()
# string1_list2=string1.split(' ',2)   # 2:表示进行拆分的次数
# print("Output:{}",format(string1_list1))
# print("Output:{}".format(string1_list2))

# string2= "Your,deliverable,is,due,in,June"
# string2_list=string2.split(',')
# print("Output:{0}".format(string2_list))
# print("Output:{0} {1} {2} {3}".format(string2_list[0],string2_list[1],string2_list[3],string2_list[-1]))
#
# print("Output:{0}".format(','.join(string2_list)))
# print("Output:{0}".format(' '.join(string2_list)))

# string3=" Remove unwanted characters    from this string.\t\t    \n"
# print("Output:{0:s}".format(string3))      # 原格式
# string3_lstrip=string3.lstrip()       # 删除左边
# print("Output:{0:s}".format(string3_lstrip))
# string3_rstrip=string3.rstrip()       # 删除右边
# print("Output:{0:s}".format(string3_rstrip))
# string3_strip=string3.strip()      # 左右两端都删除
# print("Output:{0:s}".format(string3_strip))

# 附加参数
# string4="$$Here's another string that has unwanted characters.__---++"
# print("Output:{0:s}".format(string4))
# string4_strip=string4.strip('$_-+')
# print("Output:{0:s}".format(string4_strip))

# string5= "Let's replace the spaces in this sentence with other characters."
# string5_replace=string5.replace(" ",'-@')
# print("Output:{0:s}".format(string5_replace))
# string5_replace2=string5.replace(" ",",")
# print("Output:{0:s}".format(string5_replace2))

# string6="Here's WHAT Happens WHEN You Use lower."
# print("Output:{0:s}".format(string6))
# string6_lower=string6.lower()
# print("Output:{0:s}".format(string6_lower))
# string6_upper=string6.upper()
# print("Output:{0:s}".format(string6_upper))
#
# string7="here's what happens when you use lower."
# string7_capitalize=string7.capitalize()
# print("Output:{0:s}".format(string7_capitalize))    # 只有第一个单词 首字母为大写
# for word in string7.split():
#     words=word.capitalize()
#     print("Output:{0:s}".format(words))       # 所有单词 首字母均为大写





# 2.整数和浮点数


'''
二、数据容器
'''
# 1.变量
# 注：变量的数据类型不一定要是字符串或整数。变量可以保存各种不同的 Python 数据类型
# 2.列表
# 3.字典

'''
三、各种数据类型的用途
'''
# 1.字符串
# 大小写转换
# 删除字符串末尾的空格
# 分割字符串
# 2.整数和小数
# 加减运算
# 简单数学运算
# 3.列表
# 在列表中增加或删除元素
# 删除列表的最后一个元素
# 列表重新排列
# 列表排序
# 4.字典（注：它不会保存元素顺序，输出可能不同）
# 增加一个键/值对
# 将指定的键设置为新的值
# 利用键查找值

'''
四、python标准库中内置工具
'''
# 类型：type、 dir、 help
# 作用：确定变量的数据类型或对象类型
# 1.type:
# type 可以帮你确定你的对象属于哪种数据类型
# 2.dir:
# dir 会返回一个内置方法与属性的列表，帮你列出特定数据类型能做的所有事情。
# 注：暂时忽略返回的列表中开头的那些项（以双下划线开头的那些字符串）。这些是 Python 使 用的内部方法或私有方法。
# 3.help
# help 返回对象、方法或模块
# animals='cat,dog,horse'
# animals=animals.split()
# print(animals)

# print(help(animals.split))

'''
五、模块
'''
# 1、正则表达式
# 通过导入 re 模块 ，可以使用一大波函数和 元字符来创建和 搜索任意复杂的模式。
# 元字符是正则表达式中具有特殊意义的字符,。每个元字符都有特殊意义，它 们使正则表达式能够匹配特定的字符串。
#  常用的元字符包括  |、()、[]、 .、*、+、?、^、 $ 和 (?P<name>)
# # 1.计算 字符串中模式出现的次数
# import re
# string1= "The quick brown fox jumps over the lazy dog."
# string_list=string1.split()
# # re.compile() 函数将文本形式的模式 编译成为 编译后的 正则表达式
# # re.I 函数确保模式 不区分大小写.
# # 原始字符串 标志 r 可以确保 python不处理字符串中的 转义字符 (如 \、\t 或 \n)
# # 模式1
# pattern1=re.compile(r'The',re.I)
# count=0
# for word in string_list:
#     #  re.search 函数将列表中的每个单词与正则表 达式进行比较
#     if  pattern1.search(word):
#         count+=1
# print("Output:{0:d}".format(count))

# # 模式2
# import re
# string1= "The quick brown fox jumps over the lazy dog."
# string_list=string1.split()
# pattern2=re.compile(r'(?P<math_word>The)',re.I)
# # 第一个新代码片段是 (?P<name>)，这是一个出现在 re.compile 函数中的元字符。
# # 这个元字 符使匹配的字符串可以在后面的程序中通过组名符号 <name> 来引用
# print("Output:")
# for word in string_list:
#     if pattern2.search(word):
#         print("Output:{:s}".format(pattern2.search(word).group('math_word')))

# # 模式3:
# import re
# string1= "The quick brown fox jumps over the lazy dog."
# string_to_find=r'The'
# pattern3=re.compile(string_to_find,re.I)
# # 用 re.sub 函数来在文本中用一种模式替换另一种模式。(不区分大小写)
# print("Output:{:s}".format(pattern3.sub('a',string1)))      # 使用 a 替换字符串中的 The

# 2.日期
# Python 中包含了 datetime 模块，它提供了非常强大的功能来处理日期和时间。
# 要想在脚 本中使用 datetime 模块提供的功能，
# 需要在脚本上方加入 from datetime import date, time, datetime, timedelta
# from datetime import date,time,datetime,timedelta
# # 1）.打印 今天的 日期形式，以及 年月日
# today=date.today()
# print("Today:{}".format(today))
# print("年：{}".format(today.year))
# print("月：{}".format(today.month))
# print("日：{}".format(today.day))

# 2）.利用timedelta函数 对 date 对象进行时间的 加减操作
# from datetime import date
# import timedelta
# today=date.today()
# 输出昨天
# one_day=timedelta.Timedelta(days=-1)
# yesterday=today+one_day
# print("Yesterday:{}".format(yesterday))

# # 打印 未来10天
# future_day=timedelta.Timedelta(days=10)
# future=today+future_day
# print("未来10天：{}".format(future))

# # 打印 过去 8小时
# last_hours=timedelta.Timedelta(hours=-8)
# print("过去8 小时：{} {}".format(last_hours.days,last_hours.seconds))

# 未来两个星期
# future_week=timedelta.Timedelta(weeks=2)
# print("未来两个星期：{}".format(future_week.days))

# # 计算两个 日期之间的天数
# one_day=timedelta.Timedelta(days=-1)
# yesterday=today+one_day
# date_diff=today-yesterday
# print(date_diff)    # 返回 datetie 对象
# print("输出相差得天数：{}".format(str(date_diff).split()[0]))

# # 根据 一个日期对象创建具有 特定格式得字符 串
# print("Output:{}".format(today.strftime('%m/%d/%y')))
# print("Output:{}".format(today.strftime('%b %d,%Y')))
# print("Output:{}".format(today.strftime('%y-%m-%d')))
# print("Output:{}".format(today.strftime("%B %d,%Y")))

# # 根据 一个表示日期得字符串； 创建一个带有特殊格式得 datetime 对象
# import datetime
# date1=today.strftime('%m/%d/%Y')
# date2=today.strftime('%b %d, %Y')
# date3=today.strftime('%Y-%m-%d')
# print("Output1:{}".format(datetime.datetime.strptime(date1,'%m/%d/%Y')))
# print("Output2:{}".format(datetime.datetime.strptime(date2,'%b %d, %Y')))
# print("Output3:{}".format(datetime.datetime.date(datetime.datetime.strptime(date3,'%Y-%m-%d'))))

# 3.列表
# 用途：维护各种客户列表、产品列表、资产列表、销售量列 表，等等。但是 Python 中的列表（对象的可排序集合）更加灵活！
# 上面那些列表中包含的 都是相似的对象（例如：包含客户姓名的字符串或代表销售量的浮点数），但是 Python 中 的列表可不止这么简单。
# 它可以包含数值、字符串、其他列表、元组和字典 的任意组合。
# 因为列表在商业应用中使用广泛、灵活性高、作用突出
# # 1）创建列表
# # 使用方括号创建一个列表
# # 用len()计算列表中元素的数量
# # 用max()和min()找出最大值和最小值
# # 用count()计算出列表中某个值出现的次数
# another_list=['printer',5,['star','circle',9]]
# print("Output:{}".format(another_list))
# print("Output:{}".format(len(another_list)))
# print("Output:{}".format(another_list.count(5)))

# # 2）索引值
# # 作用：通过索引值， 访问列表中得特定元素
# another_list=['printer',5,['star','circle',9]]
# print("Output:{}".format(another_list[0]))
# print("Output:{}".format(another_list[1]))
# print("Output:{}".format(another_list[-1]))
# print("Output:{}".format(another_list[-2]))

# # 3) 列表切片
# # 使用列表切片访问列表元素的一个子集
# # 从开头开始切片，可以省略第1个索引值
# # 一直切片到末尾，可以省略第2个索引值
# another_list=['printer',5,['star','circle',9]]
# print(another_list[0:2])          # 左闭右开
# print(another_list[:2])
# print(another_list[1:3])
# print(another_list[:])

# # 4) 列表复制
# # 使用[:]复制一个列表
# another_list=['printer',5,['star','circle',9]]
# a_new_list=another_list[:]
# print("Output:{}".format(a_new_list))

# # 5)列表连接
# # 使用+将两个或更多个列表连接起来
# another_list = ['printer', 5, ['star', 'circle', 9]]
# another_list1=another_list+another_list
# print(another_list1)

# # 6) 使用 in 和 not in
# # 作用： 检查列表中 是否有特定元素
# another_list = ['printer', 5, ['star', 'circle', 9]]
# a= 5 in another_list
# print("Output:{}".format(a))
# if 5 in another_list:
#     print("Output:{}".format(another_list))
#
# if 6 not in another_list:
#     print("Output:{}".format(another_list))

# # 7) 追加、删除、和 弹出元素
# # 使用append()向列表末尾追加一个新元素
# # 使用remove()从列表中删除一个特定元素
# # 使用pop()从列表 末尾删除一个元素
# another_list = ['printer', 5, ['star', 'circle', 9]]
# another_list.append(5)
# another_list.append(8)
# print("Output:{}".format(another_list))
# another_list.remove(5)    # 默认为 第一次 出现的 元素
# print(another_list)
# another_list.pop()
# print(another_list)
# another_list.pop()
# print(another_list)

# 8) 列表反转
# 使用reverse()原地反转一个列表   会修改原列表
# 要想反转列表同时  又不修改原列表，可以先  复制列表,再 对列表副本 进行 reverse 操作
# another_list = ['printer', 5, ['star', 'circle', 9]]
# copy_new_list=another_list[:]
# print("原列表：{}".format(copy_new_list))
# copy_new_list.reverse()
# print("反转列表：{}".format(copy_new_list))

# 9）列表排序
# 使用sort()对列表进行原地排序会修改原列表
# 要想对列表进行排序同时又不修改原列表， 可以先复制列表 ，再对 副本进行排序
# unordered_list=[3,6,12,1,67,45,89,33,99,2,46]
# print("原列表：",unordered_list)
# list_copy=unordered_list[:]
# list_copy.sort()
# print("排序后的列表：",list_copy)
# print("原列表：",unordered_list)

# # 10）sorted 排序函数
# # 使用 sorted() 对一个列表集合按照列表中某个位置的元素进行排序
# my_list=[[1,2,3,4],[4,3,1,2],[3,4,2,1],[2,4,3,3]]
# sorted_by_index_3=sorted(my_list,key=lambda index: index[3])       # 按 子列表中索引为3 的元素进行 排序
# print(sorted_by_index_3)

# 注： sorted 函数的排序与 sort 函数的 in-place 原地排序方式不同，
# sort 函数 改变了原列表的元素顺序，
# sorted 函数则返回一个新的排好序的列表，并不改变原列表的 元素顺序。

# 4.元组
# # 特点： 不能被修改，其余跟列表 特别相似
# # 1）创建元组
# my_tuple=('x','y','z')
# print("Output:{}".format(my_tuple))
# print("Output:{}".format(len(my_tuple)))
# print("Output:{}".format(my_tuple[0]))
#
# longer_tuple=my_tuple+my_tuple
# print("Output:{}".format(longer_tuple))

# # 2) 元组解包
# # 使用赋值操作符左侧的变量对元组进行解包
# my_tuple=('x','y','z')
# one,two,three=my_tuple
# print("Output:{} {} {}".format(one,two,three))
# var1='李云龙'
# var2='马云'
# print("原格式：{} {}".format(var1,var2))
#
# # 用途：
# # 在变量之间 交换 彼此的值
# var1,var2=var2,var1
# print("Output:{} {}".format(var1,var2))

# 3）元组转换为列表（列表转换成元组）
# my_list=[1,2,3]
# my_tuple=('x','y','z')
# print("列表转化为元组:{}".format(tuple(my_list)))
# print("元组转化为列表:{}".format(list(my_tuple)))

# 5. 字典
# 用途：Python 中的字典本质上是包含各种带有唯一标识符的成对信息的列表。和列表一样，字 典也广泛应用于各种商业分析。
# 在商业分析中，可以用字典表示客户（以客户编码为键 值），也可以用字典表示产品（以序列号或产品编号为键值），
# 还可以用字典表示资产、 销售量等。

# 注：
# • 在列表中，你可以使用被称为索引或索引值的连续整数来引用某个列表值。
# 在字典中， 要引用一个字典值，则可以使用整数、字符串或其他 Python 对象，这些统称为字典键。
# 在唯一键值比连续整数更能反映出变量值含义的情况下，这个特点使字典比列表更实用。
# • 在列表中，列表值是隐式排序的，因为索引是连续整数。在字典中，字典值则没有排序， 因为索引不仅仅只是数值。你可以为字典中的项目定义排序操作，但是字典确实没有内 置排序。
# • 在列表中，为一个不存在的位置（索引）赋值是非法的。在字典中，则可以在必要的时 候创建新的位置（键）。
# • 因为没有排序，所以当你进行搜索或添加新值时，字典的响应时间更快（当你插入一个 新项目时，计算机不需要重新分配索引值）。
# 当处理的数据越来越多时，这是一个重要 的考虑因素。
# 1）创建字典
# 使用花括号创建字典
# 用冒号分隔键-值对
# 用len()计算出字典中键-值对的数量
# empty_dict={}
# a_dict={'one':1,'two':2,'three':3,'four':4}
# print("Output:{}".format(a_dict))
# print("Output:{}".format(len(a_dict)))
#
# another_dict={'x':'printer','y':5,'z':['star','circle',9]}
# print("Output:{}".format(another_dict))
# print("Output:{}".format(len(another_dict)))

# 2)引用字典中的值
# 使用键来引用字典中的特定值
# a_dict={'one':1,'two':2,'three':3,'four':4}
# another_dict={'x':'printer','y':5,'z':['star','circle',9]}
# print(a_dict['two'])
# print(another_dict['z'])

# # 3)复制
# # 使用 copy 复制一个字典
# a_dict={'one':1,'two':2,'three':3,'four':4}
# a_new_dict=a_dict.copy()
# print(a_new_dict)

# # 4) 键、值和项目
# # 使用 keys() ,values() 和 items()
# a_dict={'one':1,'two':2,'three':3,'four':4}
# print(a_dict.keys())
# a_dict_keys=a_dict.keys()
# print(a_dict_keys)
# print(a_dict.values())
# print(a_dict.items())

# # 5)使用 in 、not  in 和 get
# # 用途：测试字典中是否 存在 某个键值的 两种方法
# another_dict={'x':'printer','y':5,'z':['star','circle',9]}
# if 'y' in another_dict:
#     print(another_dict.keys())
# if 'c' not in another_dict:
#     print(another_dict.keys())
# print(another_dict.get('z'))
# print(another_dict.get('x'))
# print(another_dict.get('a','Not in dict'))     # 如果 第一个参数在 字典中，则返回 键值； 如果不存在，则返回 第二个参数

# 6）排序
# 使用sorted()对字典进行排序
# 要想对字典排序的同时不修改原字典
# 先复制字典

# a_dict={'one':1,'two':5,'three':3,'four':8}
# print("原字典：{}".format(a_dict))
# dict_copy=a_dict.copy()
# print(dict_copy.items())
# # 升序
# oreder_dict1=sorted(dict_copy.items(),key=lambda item:item[0])    # 按 键 进行排序
# print(oreder_dict1)
# oreder_dict2=sorted(dict_copy.items(),key=lambda item:item[1])    # 按值 进行排序
# print(oreder_dict2)
#
# oreder_dict3=sorted(dict_copy.items(),key=lambda item:item[0],reverse=True)    # 降序
# print(oreder_dict3)
# oreder_dict4=sorted(dict_copy.items(),key=lambda item:item[1],reverse=False)    # 升序
# print(oreder_dict4)

# 6.控制流
# 控制流元素非常重要，因为可以在程序中包含有意义的业务逻辑。很多商务处理和分析依 赖于业务逻辑，
# 例如“如果客户的花费超过一个具体值，那么就怎样怎样”或“如果销售 额属于 A 类，则编码为 X，如果销售额属于 B 类，则编码为 Y，否则编码为 Z。”
# 这些逻辑 语句在代码中可以用控制流元素来表示。
# Python 提供了若干种控制流元素，包括 if-elif-else 语句、for 循环、range 函数和 while 循环。

# 7.简化 for 循环
# # 1）列表生成式
# # 使用列表生成式选择特定的行
# my_data=[[1,2,3,4],[3,5,7,9],[2,4,6,8],[4,6,2,7]]
# rows_to_keep=[row for row in my_data if row[3]>6]       # row[i] 表示子列表中 如果第 i个下标元素 值大于6，则输出整个子列表
# print(rows_to_keep)
#
# # 2)集合生成式
# # 使用集合生成式在列表中选择出一组唯一的元组
# # 第一种：
# my_data=[(1,2,3),(4,5,6),(7,8,9),(4,7,9)]
# set_of_tuple1={x for x in my_data}
# print(set_of_tuple1)
# # 第二种：
# my_data=[(1,2,3),(4,5,6),(7,8,9),(4,7,9)]
# set_of_tuple2=set(my_data)      # 将列表 转化为 集合
# print(set_of_tuple2)

# # 3）字典生成式
# # 使用字典生成式选择特定的键-值对
# my_dict={'customer1': 7, 'customer2': 9, 'customer3': 11}
# result={key:value for key,value in my_dict.items() if value>8}
# print(result)

# # 8.异常
# # 1) try-except
# # 计算平均值
# def getMean(num):
#     return sum(num)/len(num)
# my_list=[]
# # print(getMean(my_list))
# try:
#     print("Output:{}".format(getMean(my_list)))
# except ZeroDivisionError as detail:
#     print("Output:Error:{}".format(float('nan')))
#     print("Output:Error:{}".format(detail))
#
# # 2) try-except-else-finally
# def getMean(num):
#     return sum(num)/len(num)
# my_list=[]
# try:
#     result=getMean(my_list)
# except ZeroDivisionError  as detail:
#     print("Output(Errors):{}".format(float('nan')))
#     print("Output(Error):{}".format(detail))
# else:
#     print("Output(Mean):{}".format(result))
# finally:
#     print(" The finally block is executed every time")

# # 9. 读取文本文件
# # argv[0] 就是脚本名称，argv[1] 是命令行中传递给脚本的第一个 附加参数
#
# # 1)读取一行文本
# # 1)第一种:
# import sys
# input_path=sys.argv[1]
# with open(input_path,'r',newline='') as f:
#     for row in f:
#         print("Output:{}".format(row.strip()))
#
# # 第二种：
# input_file='file2.txt'
# with open(input_file,'r',newline='')  as f:
#     for row in f:
#         print("Output:{}".format(row.strip()))

# # 2）读取多行文本
# import glob
# import os,sys
# print("读取多个文件：\n")
# input_path=sys.argv[1]
# for input_file in glob.glob(os.path.join(input_path,'*.txt')):
#     with open(input_file,'r',newline='')  as f:
#         for row in f:
#             print("Output:{}".format(row.strip()))

# # 10.写入文本文件
# # 1) 写入 txt文件
# my_txt=['one',1,'two',2,'three',3,'four',4]
# max_index=len(my_txt)
# file_txt='file2.txt'
# with open(file_txt,'w') as f:
#     for index_value in range(len(my_txt)):
#         if index_value<(max_index-1):
#             f.write(str(my_txt[index_value])+'\n')
#         else:
#             f.write(str(my_txt[index_value])+'\t')
# f.close()
# print("Output  written to file")

# # 2） 写入 csv 文件
# my_txt=['one',1,'two',2,'three',3,'four',4]
# max_index=len(my_txt)
# file_txt='file2.txt'
# with open(file_txt,'a') as f:
#     for index_value in range(len(my_txt)):
#         if index_value<(max_index-1):
#             f.write(str(my_txt[index_value])+',')
#         else:
#             f.write(str(my_txt[index_value])+'\n')
# f.close()
# print("Output  written to file")







'''
五、总结：
'''
# import sys
# import pprint
# pprint.pprint(sys.path)

# print(help(pprint.pprint))
# print(type(sys.path))















