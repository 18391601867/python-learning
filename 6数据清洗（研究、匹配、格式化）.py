'''
数据清洗的好处： 让数据更容易存储，搜索，复用
'''

'''
数据清洗步骤：
1、观察数据字段
'''

# 1、找出需要清洗的数据
# 1）根据需求，替换标题
# from csv import DictReader  # 每一行创建字典
# import csv
# # data_rdr=DictReader(open('surveys_catalogue.csv','rb'))
# data_csv=csv.DictReader(open('surveys_catalogue.csv','r'))
#
# info_data=[line for  line in data_csv]
# for data_dict in info_data:
#     for key,value in data_dict.items():
#         print("{}:{}".format(key,value))

# 2)合并问题与答案
# 修复标签问题的另一种方法是 Python 的 zip 方法：

# from  csv  import reader
# # 每一行创建 列表，
# # 要用的是 zip 方法，需要的是列表而不是字典，这样 我们就可以将标题列表与数据列表合并在一起。
#
# data_rdr=reader(open('surveys_catalogue.csv','r'))
# data_rows=[line for line in data_rdr]
# print(len(data_rows))


# 2、数据格式化
#数据清洗最常见的形式之一，就是将可读性很差或根本无法读取的数据和数据类型转换成 可读性较强的格式。
# 特别是如果需要用数据或可下载文件来撰写报告，你需要将其由机器 可读转换成人类可读。
# 如果你的数据需要与 API 一起使用，你可能需要特殊格式的数据 类型。

# 3、找出离群值和不良数据
# 你要做的是清洗数据，不是处理数据或修改数据，所以在需要删除离群值或 不良数据记录时，多花点时间思考如何处理这些数据。
# 如果你剔除离群值使 数据归一化，应该在最终结论中明确说明这一点。

# 4、找出重复值
# 如果你要处理的是同一调查数据的多个数据集，或者是可能包含重复值的原始数据，删除 重复数据是确保数据准确可用的重要步骤。
# 如果你的数据集有唯一标识符，你可以利用这 些 ID，确保没有误插入重复数据或获取重复数据。
# 如果你的数据集没有索引，你可能需要 找到判断数据唯一性的好方法（例如创建一个可索引的键）。

# list_dupes=[1, 5, 6, 2, 5, 6, 8, 3, 8, 3, 3, 7, 9]
# set_dupes=set(list_dupes)
# print(set_dupes)

'''
定义唯一性的方法：
'''
#第一种：  python中两个集合的 交集、并集、 差集、 以及是否包含

# first_set=set([1, 5, 6, 2, 6, 3, 6, 7, 3, 7, 9, 10, 321, 54, 654, 432])
# second_set=set([4, 6, 7, 432, 6, 7, 4, 9, 0])
# print(first_set.intersection(second_set))      # 集合的 intersection 方法返回两个集合的交集
# print(first_set.union(second_set))        # 集合的 union 方法将第一个 集合的值与 第二个集合的值合并在一起
# print(first_set.difference(second_set))   # difference 方法是 第一个和 第二个集合的差集
# print(second_set-first_set)       # 用一个集合去减 另一个 集合。（改变集合的顺序会改变结果）
# print(6 in second_set)        # in 判断元素是否包含在 集合中（速度很快）

# 第二种： numpy库
# 工作原理：
# import numpy as np
# list_dupes= [1, 5, 6, 2, 5, 6, 8, 3, 8, 3, 3, 7, 9]
# # numpy库的unique 方法会保存索引编号。 设置 return_index=True, 返回的是由数组组成的元组：
# # 第一个是 唯一值组成的数组，
# # 第二个是 由索引编号组成的  扁平化数组——只包含每一个数字第一次出现时的索引编号
# # print(np.unique(list_dupes,return_index=True))
#
# array_dupes=np.array([[1, 5, 7, 3, 9, 11, 23], [2, 4, 6, 8, 2, 8, 4]])
# # array_dupes代表 创建了一个 numpy 矩阵，矩阵是由（长度相同的）数组组成的数组
# # unique 将矩阵转换成 由唯一值 组成的集合
# print(np.unique(array_dupes))
#
# # 注：如果没有唯一键,可以编写函数创建 唯一集合，写法和（列表生成式一样简单）

# 5、模糊匹配
# 如果要处理 不止一个数据集，或者是 未标准化的脏数据，可以用 模糊匹配来寻找和 合并重复值；
# 模糊匹配可以判断 两个元素（通常是字符串） 是否“相同”；
# 模糊匹配的做法： 一个由 SeatGeek 开发的 python 库，使用很酷的内置方法来匹配多种场景的线上售票。
# 安装 pip install  fuzzywuzzy
#比如说你要处理一些脏数据。可能是输入时粗心，也可能是用户输入的，
# 导致数据中包含拼写错误和较小的语法错误或句法偏移。

# from fuzzywuzzy  import fuzz
# my_records=[{'favorite_book': 'Grapes of Wrath',
#              'favorite_movie': 'Free Willie',
#              'favorite_show': 'Two Broke Girls',
#              },
#             { 'favorite_book': 'The Grapes of Wrath',
#               'favorite_movie': 'Free Willy',
#               'favorite_show': '2 Broke Girls',
#               }]

# # fuzz 模块的 ratio 函数，接受两个字符串作比较。
# # 返回的是两个字符 串序列的相似程度（介于 1 和 100 之间的值）。
# print(fuzz.ratio(my_records[0].get('favorite_book'),my_records[1].get('favorite_book')))

# fuzz 模块的 partial_ratio 函数，接受两个字符串作比较。
# 返回的是 匹配程度 (最高) 的子字符串序列的相似程度（一个介于 1 和 100 之间的值）。
# print(fuzz.partial_ratio(my_records[0].get('favorite_book'), my_records[1].get('favorite_book')))

# print(fuzz.partial_ratio(my_records[0].get('favorite_book'),my_records[1].get('favorite_movie')))

# from fuzzywuzzy import fuzz
# my_records=[{
#     'favorite_food': 'cheeseburgers with bacon',
#      'favorite_drink': 'wine, beer, and tequila',
#     'favorite_dessert': 'cheese or cake',
#     },
#     {
#     'favorite_food': 'burgers with cheese and bacon',
#     'favorite_drink': 'beer, wine, and tequila',
#     'favorite_dessert': 'cheese cake',
#     }]
#  fuzz 模块的 token_sort_ratio 函数，在匹配字符串时不考虑单词顺 序。
#  对于格式不限的调查数据来说，这个方法是很好用的，比如“I like dogs and cats” 和“I like cats and dogs”的含义相同。
#  每个字符串都是先排序然后再比较，所以如果包 含相同的单词但顺序不同，也是可以匹配的。
# print(fuzz.token_sort_ratio(my_records[0].get('favorite_food'),my_records[1].get('favorite_food')))
# print(fuzz.token_sort_ratio(my_records[0].get('favorite_drink'),my_records[1].get('favorite_drink')))

#  fuzz 模块的 token_set_ratio 函数，
#  同样用的是标记方法，但比较的 是标记组成的集合，得出两个集合的交集和差集。
#  这个函数对排序后的标记尝试寻找最 佳匹配，返回这些标记相似的比例。
# print(fuzz.token_set_ratio(my_records[0].get('favorite_food'),my_records[1].get('favorite_food')))
# print(fuzz.token_set_ratio(my_records[0].get('favorite_drink'),my_records[1].get('favorite_drink')))


# from fuzzywuzzy import process
# choices=['yes','No','Maybe','N/A']
#  利用 FuzzyWuzzy 的 extract 方法，将字符串与可能匹配的列表依次比较。
#  函数返回的 是 choices 列表中两个可能的匹配。
# part=process.extract('ya',choices,limit=2)
# print(part)

#  利用 FuzzyWuzzy 的 extractOne 方法，返回 choices 列表中与我们的字符串对应的最佳 匹配。
# part2=process.extractOne('ya',choices)
# print(part2)

# part3=process.extract('nope',choices,limit=2)
# print(part3)

# part4=process.extractOne('nope',choices)
# print(part4)


# 6、正则表达式匹配
# 模糊匹配不一定总能满足你的需求，如果需要匹配字符串的一部分，如果只想匹配电话号码或电子邮件地址，
# 在抓取数据时，或 编译多个来源的原始数据时， 需要正则表达式 解决问题。

# import re
# # 定义一个普通 字符串的基本格式。 这个模式可以匹配包含字母和数字，但不包含空格和标点的字符串，
# # 这个模式会一直匹配，直到无法匹配为止 （+ 表示贪婪匹配）
# word='\w+'
# sentence='Here is my sentence.'
# # findall方法可以找出 这个模式在字符串中的 所有匹配，成功匹配了句子中的 每一个单词，但没有匹配句号
# find_result=re.findall(word,sentence)
# print(find_result)
# # search 方法可以在整个字符串中搜索匹配。 发现匹配后，返回匹配对象。
# search_result=re.search(word,sentence)
# print(search_result)
# # 匹配对象的 group 方法会返回 匹配的字符串
# print(search_result.group())
# # match  方法只从字符串开头 开始搜索。它的工作原理 与 search 不同。
# math_result=re.match(word,sentence)
# print(math_result)

# 还可以改变寻找匹配的方式。在 上面的例子中我们看到，findall 返回的是所有匹配组成的列表。比如说你只想提取长文 本中的网站。
# 你可以利用正则表达式模式找到链接，然后利用 findall 从文本中提取出所 有链接。
# 或者你可以查找电话号码或日期。如果你能够将想要寻找的内容转换成简单的模 式，
# 并将其应用到字符串数据上，你就可以使用 findall 方法。

# 还用到了 search 和 match 方法，在上面的例子中二者返回的结果相同——它们匹配的 都是句子中的第一个单词。
# 我们返回的是一个匹配对象，然后可以利用 group 方法获取数 据。group 方法还可以接受参数。
# import re
# # 定义一个数字模式。 加号表示 贪婪模式， 所以它会 尽可能匹配所有数字，
# # 直到遇到的一个 非数字字符串为 止。
# number='\d+'
# # 定义一个 大写单词的匹配。 这个模式使用 方括号来定义一个更长模式的一部分。
# # 方括号的意思是： 第一个字母是大写，后面紧跟着是一个 连续的单词。
# capitalized_word='[A-Z]\w+'
#
# sentence='I have 2 pets: Bear and Bunny.'
#
# # search_number=re.search(number,sentence)
# # print(search_number.group())         # 返回匹配对象
#
# match_number=re.match(number,sentence)
# print(match_number.group())     # 返回None,  而不是匹配对象
# 错误原因：只有一个模式组
# # search_capital=re.search(capitalized_word,sentence)
# # print(search_capital.group())
#
# # match_capital=re.match(capitalized_word,sentence)
# # print(match_capital.group())
#

# 改进法：
# import re
# # 用相同的大写单词语法， 用了两次，分别放在括号，括号的作用： 分组
# name_regex1='([A-Z]\w+) ([A-Z]\w+)'    # 注意：两个括号之间 须有 空格
# names= "Barack Obama,Ronald Reagan,Nancy Drew"
# # match 方法中包含多个正则表达式组。 如果找到匹配的话， 将返回多个匹配组
# name_match=re.match(name_regex1,names)
# print(name_match.group())
# # # 返回所有匹配组返回的 元组
# # print(name_match.groups())

# 格式命名使 代码更加清晰（此模式中， 第一个 first_name ; 第二个 last_name）
# name_regex2='(?P<first_name>[A-Z]\w+) (?P<last_name>[A-Z]\w+)'
# for name in re.finditer(name_regex2,names):
    # 返回元组形式
    # print(name.groups())
    # 返回对象
    # print(name.group())
    # print('{}'.format(name.group()))


# 7、处理重复记录
from csv import DictReader
data_rdr=DictReader(open('message.csv','r'))
data=[d for d in data_rdr]
# print(data)
def combine_data_dict(data_rows):
    data_dict={}
    for row in data_rows:
        key='%s-%s' %(row.get('学生'),row.get('班级名称'))
        if key in data_dict.keys():
            data_dict[key].append(row)
        else:
            data_dict[key]=[row]
    return data_dict
print(combine_data_dict(data))







