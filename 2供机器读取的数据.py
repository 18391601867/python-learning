'''
数据可以存储成许多不同的格式和文件类型。某些格式存储的数据很容易被机器处理，而 另一些格式存储的数据则容易被人工读取。
微软的 Word 文档属于后者，而 CSV、JSON 和 XML 文件则属于前者。
'''

'''
以易于机器理解的方式来存储数据的文件格式，通常被称作机器可读的 （machine readable）。常见的机器可读格式包括：
• 逗号分隔值（Comma-Separated Values，CSV） 
• JavaScript 对象符号（JavaScript Object Notation，JSON） 
• 可扩展标记语言（eXtensible Markup Language，XML）
'''
'''一、CSV数据 '''
# csv:指将数据列用逗号分隔的文件
# 另一种数据类型，叫作制表符分隔值（tab-separated values，TSV）数据，有时也与CSV 归为一类。
# TSV 与 CSV 唯一的不同之处在于，数据列之间的分隔符是制表符（tab），而不 是逗号。
# 文件的扩展名通常是 .tsv，但有时也用 .csv 作为扩展名。从本质上来看，.tsv 文 件与 .csv 文件在 Python 中的作用是相同的

# 注：如果文件的扩展名是 .tsv，那么里面包含的很可能是 TSV 数据。如果文件的 扩展名是.csv，那么里面包含的可能是CSV 数据，但也可能是TSV 数据。
# 一定要打开文件查看一下，这样你可以在导入数据之前就明确所处理的数据 类型。
# 1)导入csv 数据
# import csv
# csv_file=open('data.csv','r')
# # 以列表形式读取
# read_info=csv.reader(csv_file)
# # 以字典形式读取
# read_info=csv.DictReader(csv_file)
# for row in read_info:
#     print(row)

# import csv
# csv_file='supplier_data.csv'
# output_file='supplier_data3.csv'
# with open(csv_file,'r',newline='') as read_f:
#     with open(output_file,'w',newline='') as write_f:
#         file_read=csv.reader(read_f,delimiter=',')
#         # file_write=csv.writer(write_f,delimiter=',')
#         for row_list in file_read:
#             print(row_list)
#             if len(row_list)==5:
#                 write_f.writelines(row_list)
#                 write_f.writelines('\n')


# # 2) 不使用 csv 模块
# csv_file='supplier_data.csv'
# with open(csv_file,'r')  as f:
#     for row in f.readlines():     # 遍历每一行
#         # print(row.split(','))   # 读取每一行
#         row_list=row.split(',')
#         for cols in row_list:     # 遍历一行中所有元素
#             print(cols)

# # 3）利用pandas 处理数据
# import pandas  as pd
# import sys
# csv_file='supplier_data.csv'
# csv_file2='supplier_data1.csv'
# data_frame=pd.read_csv(csv_file)
# print(data_frame)
# # 将 一个文件数据导入 另一个文件
# data_frame.to_csv(csv_file2,index=False)

# # 4)筛选特定的行
# # • 行中的值满足某个条件
# # • 行中的值属于某个集合
# # • 行中的值匹配于某个模式（正则表达式

# • 行中的值满足某个条件
# 第一种：
# import csv
# csv_file='supplier_data.csv'
# csv_file2='supplier_data1.csv'
# with open(csv_file,'r',newline='')  as f_read:
#     with open(csv_file2,'w',newline='')  as f_write:
#         file_read=csv.reader(f_read)
#         for row_list in file_read:
#             # print(row_list)
#             supplier=str(row_list[0]).strip()
#             cost=str(row_list[3]).strip("$").replace(',','')
#             # print(cost)
#             if cost!='Cost' and len(row_list)==5:
#                 if supplier=='Supplier Z' or float(cost)>600.0:
#                     f_write.writelines(row_list)
#                     f_write.writelines('\n')
#                     print(row_list)

# 第二种：pandas
# 功能：pandas提供了一个 loc 函数，可以同时选择 特定的 行与列；
# 但是需要在 逗号前面设定 行筛选条件， 在 逗号 后面设定 列筛选的条件。
# import pandas as pd
# input_file='supplier_data.csv'
# out_file='supplier_data2.csv'
# data_fram=pd.read_csv(input_file)
# # 格式化 'Cost'列
# data_fram['Cost']=data_fram['Cost'].str.strip('$').astype(float)
# # print(data_fram)
# print(data_fram['Cost'])
# data_fram_value_meets_condition=data_fram.loc[(data_fram['Supplier Name'].str.contains('Z')) | (data_fram['Cost']>600.0),:]
# data_fram_value_meets_condition.to_csv(out_file,index=False)

# 行中的值属于某个集合
# 有些时候，当行中的值属于某个集合时，才需要保留这些行。例如，你可能会希望在
# CSV文件 ｜ 61
# 数据集中保留那些供应商名字属于集合 {Supplier X, Supplier Y} 的行（这里的花括号
# 表示集合，不是 Python 中的字典），或者希望保留所有购买日期属于集合 {'1/20/14',
# '1/30/14'} 的行。在这种情况下，你可以检验行中的值是否属于某个集合，然后筛选出具
# 有属于该集合的值的行。
# 第一种
# import csv
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# find_dates=['1/20/14','1/30/14']
# with open(input_file,'r',newline='') as f_csv:
#     with open(output_file,'w',newline='') as f_out_csv:
#         file_reader=csv.reader(f_csv)
#         file_write=csv.writer(f_out_csv)
#         header=next(file_reader)
#         file_write.writerow(header)
#         for row_list in file_reader:
#             a_data=row_list[4]
#             if a_data in find_dates:
#                 file_write.writerow(row_list)
#                 # print(row_list)

# # 第二种：
# # pandas ：
# import pandas as pd
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# important_data=['1/20/14','1/30/14']
# data_frame=pd.read_csv(input_file)
# data_frame_value_in_set=data_frame.loc[(data_frame['Purchase Date'].isin(important_data)),:]
# data_frame_value_in_set.to_csv(output_file)    # 保存到文件
# # print(data_frame_value_in_set)    # 直接输出

# • 行中的值匹配于某个模式（正则表达式
# 第一种：
# import re
# import csv
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# pattern=re.compile(r'(?P<my_pattern_group>^001-.*)',re.I)
# with open(input_file,'r',newline='') as f_csv:
#     with open(output_file,'w',newline='') as f_out_csv:
#         file_reader=csv.reader(f_csv)
#         file_write=csv.writer(f_out_csv)
#         header=next(file_reader)
#         file_write.writerow(header)
#         for row_list in file_reader:
#             # print(row_list)
#             Invoice_num=row_list[1]
#             if pattern.search(Invoice_num):
#                 file_write.writerow(row_list)
#                 print(row_list)

# # 第二种：
# # 使用 pandas 时，可以使用 startswith 函数来搜索数据，不需再使用笨重冗长的正则表达式
# import pandas as pd
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# data_frame=pd.read_csv(input_file)
# data_frame_value_matches_pattern=data_frame.loc[(data_frame['Invoice Number'].str.startswith('50-')),:]
# data_frame_value_matches_pattern.to_csv(output_file)
# print(data_frame_value_matches_pattern)

# 5）选取特定的列
# 有两种通用方法可以在 CSV 文件中选取特定的列
# ① 使用列索引值(下标)
# ② 使用列标题

# ① 使用列索引值
# 第一种:
# CSV文件中选取特定列的一种方法是使用你想保留的列的索引值
# import csv
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# my_columns=[0,3]
# with open(input_file,'r',newline='') as f_csv:
#     with open(output_file,'w',newline='') as f_out_csv:
#         file_reader=csv.reader(f_csv)
#         file_write=csv.writer(f_out_csv)
#         for row_list in file_reader:
#             row_list_output=[]
#             for index_value in my_columns:
#                 row_list_output.append(row_list[index_value])
#             file_write.writerow(row_list_output)
#             # print(row_list_output)

# # 第二种:
# # pandas:
# import pandas as pd
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# data_frame=pd.read_csv(input_file)
# data_frame_columns_index=data_frame.iloc[:,[0,3]]     # 多行两列  如果是数字， 则用 iloc
# data_frame_columns_index.to_csv(output_file,index=False)
# print(data_frame_columns_index)


# # ② 使用列标题
# # 第一种:
# import csv
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# my_columns=['Invoice Number','Purchase Date']
# my_columns_index=[]
# with open(input_file,'r',newline='') as f_csv:
#     with open(output_file,'w',newline='') as f_out_csv:
#         file_reader=csv.reader(f_csv)
#         file_write=csv.writer(f_out_csv)
#         header=next(file_reader)
#         # print(header)
#         for index in range(len(header)):
#             if header[index] in my_columns:
#                 my_columns_index.append(index)
#         file_write.writerow(my_columns)
#         for row_list in file_reader:
#             row_list_out=[]
#             for index in my_columns_index:
#                 row_list_out.append(row_list[index])
#             file_write.writerow(row_list_out)
#             print(row_list_out)

# # 第二种：
# # Pandas
# import pandas as pd
# input_file='supplier_data.csv'
# output_file='supplier_data2.csv'
# data_frame=pd.read_csv(input_file)
# data_frame_columns_name=data_frame.loc[:,['Invoice Number','Purchase Date']]     # 如果 是字符串， 则用 Loc()
# data_frame_columns_name.to_csv(output_file,index=False)
# # print(data_frame_columns_name)

# 6）选取连续的行
# 注：有些时候，在文件内容中，工作表头部和尾部都是你不想处理的。例如，文件头部可能是
# 标题和作者信息，文件尾部也可能会列出来源、假设、附加说明和注意事项。在很多情况
# 下，你不需要处理这些内容。
# 如果要在CSV 文件中选取连续的行，需要对 输入文件修改：
# (1) 在电子表格软件中打开 supplier_data.csv。
# (2) 在文件头部插入 3 行，就在列标题那行的上面。
# 在 A1:A3 单元格中随便写一些文字，比如“I don’t care about this line”。
# (3) 在文件尾部，也就是最后一行数据下面插入 3 行。
# 在最后一行数据下面 A 列的 3 个单元格中随便写一些文字，比如“I don’t want this
# line either”。
# (4) 将文件保存

# # 第一种：
# import csv
# input_file='supplier_data_unnecessary_header_footer.csv.csv'
# output_file='supplier_data2.csv'
# row_count=0      # 记录行号
# with open(input_file,'r',newline='') as f_csv:
#     with open(output_file,'w',newline='') as f_csv_out:
#         file_reader=csv.reader(f_csv)
#         file_write=csv.writer(f_csv_out)
#         for row_list in file_reader:
#             if row_count>=3 and row_count<=15:
#                 value=[value.strip() for value in row_list]
#                 print(value)
#                 file_write.writerow(value)
#             row_count+=1

# # 第二种：
# # pandas: 提供了一个 drop 函数根据行索引 或 列标题来丢弃行或列；
# # 提供的 iloc 功能强大，可以通过 这个函数根据 行索引选取一个 单独行作为 索引列
# # 使用 reindex 函数为数据框重新生成 索引
# import pandas as pd
# input_file='supplier_data_unnecessary_header_footer.csv.csv'
# output_file='supplier_data2.csv'
# data_frame=pd.read_csv(input_file,header=None,encoding='gbk')
# data_frame=data_frame.drop([0,1,2,16,17,18])
# data_frame.columns=data_frame.iloc[0]
# data_frame=data_frame.reindex(data_frame.index.drop(3))
# data_frame.to_csv(output_file,index=False)
# print(data_frame)

# 7)添加标题行
# 电子表格中没有标题行，但你确实希望所有列都有列标题。在这种情况下，可
# 以使用脚本添加列标题。
# 第一种：
# import csv
# input_file='supplier_data_no_header_row.csv.csv'
# output_file='supplier_data2.csv'
# with open(input_file,'r',newline='') as f_csv:
#     with open(output_file,'w',newline='') as f_csv_out:
#         file_reader=csv.reader(f_csv)
#         file_write=csv.writer(f_csv_out)
#         header_list=['Supplier Name','Invoice Number','Part Number','Cost','Purchase Date']
#         file_write.writerow(header_list)
#         for row_list in file_reader:
#             file_write.writerow(row_list)

# # 第二种：pandas
# # pandas 中的 read_csv 函数可以直接指定输入文件不包含标题行，并可以提供一个列标题列表。
# import pandas as pd
# input_file='supplier_data_no_header_row.csv.csv'
# output_file='supplier_data2.csv'
# header_list=['Supplier Name','Invoice Number','Part Number','Cost','Purchase']
# data_frame=pd.read_csv(input_file,header=None,names=header_list)
# data_frame.to_csv(output_file,index=False)

# 8) 读取多个 CSV文件


'''二、JSON数据 '''
# JSON 数据是数据传输最常用的格式之一。人们喜欢这一格式，是因为它结构清晰、易于 阅读且方便解析。
# 网站在向页面的 JavaScript 传输数据时，JSON 也是最常用的数据格式之 一。许多网站都提供了支持 JSON 的 API
# 1）导入JSON数据
# import json
# # 打开JSON文件
# json_file=open('city.json','rb').read()
# # 将JSON数据载入 python
# read_info=json.loads(json_file)
# for item in read_info:
#     print(item)


'''三、XML数据 '''
# XML 格式的数据既便于机器读取，也便于人工读取。但是对于本章的数据集来说，预 览并理解CSV 文件和JSON 文件要比XML 文件容易得多
# 注：如果文件的扩展名是.xml，那么它是XML 数据。如果文件扩展名是.html 或 .xhtml，有时也可以用 XML 解析器来解析

# 1）导入XML 数据

# from xml.etree import ElementTree as ET
# tree=ET.parse('xml_file')
# root=tree.getroot()
# data=root.find('Data')
# all_data=[]
# for observation  in data:
#     record=[]
#     for item in observation:
#         lookup_key=item.attrib.keys()[0]
#         if  lookup_key=='Numeric':
#             rec_key='NumERIC'
#             rec_value=item.attrib['Numeric']
#         else:
#             rec_key=item.attrib[lookup_key]
#             rec_value=item.attrib['Code']
#         record[rec_key]=rec_value
#     all_data.append(record)
# print(all_data)


'''四、读取CSV 数据的更多形式'''
# # 如果没有标题行， 而且数据间有 空格隔开的浮点数
# from numpy import loadtxt
# my_datarray=loadtxt('no_title.csv')
# print(my_datarray)

# #包含标题行， 而且数据 不全是 浮点数
# from numpy import dtype,loadtxt
# data_type=dtype([('name','S10'),('age',int),('color','S6'),('score',float)])
# my_datarray=loadtxt('has_title.txt',skiprows=1,dtype=data_type)
# print(my_datarray)






