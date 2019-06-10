
'''
在一个大文件集合中查找一组项目
'''

# import csv
# import glob
# import os
# from datetime import date
# # 入 datetime 模块的 date 方法和 xlrd 模块的 xldate_as_tuple 方法目的：确保我们从输入文件中提取的任何日期数据都 能以特定的形式保存到输出文件中。
# from xlrd import open_workbook,xldate_as_tuple
# item_number_file='item_numbers_to_ﬁnd.csv'        # 目标文件（要搜索的数值的项目）
# input_file='C:\\Users\王宝刚\Desktop\ﬁle_archive'
# output_file='item_data.csv'      # 将结果值输出
# item_numbers_to_find=[]
# with open(item_number_file,'r',newline='') as item_number_csv_file:
#     file_reader=csv.reader(item_number_csv_file)
#     for row_list in file_reader:
#         for row in row_list:
#             item_numbers_to_find.append(row)
# # print(item_numbers_to_find)    # 目标值
# file_write=csv.writer(open(output_file,'a',newline=''))
#
# file_counter=0      # 读入脚本的历史文件数量
# line_counter=0     # 在所有的输入文件和工作表中读出的行数，
# count_of_item_numbers=0    # 行中数值项目是要搜索的数值项目的行数
#
# for inputfile in glob.glob(os.path.join(input_file,'*.*')):
#     # 历史文件夹中所有输入文件中循环
#     # os.path.join() 函数和glob.glob() 函数来在ﬁle_archive 文件夹中找到所有匹配于一个 特定模式的文件
#     # os.path.join() 函数 将这个文件夹路径与 文件夹中所有文件的文件名 连接起来，
#     # 这些匹配于特定模式的文件名由 glob.glob() 函数进行扩展；
#     # 使用模 式 '*.*' 来匹配以任意扩展名结尾的任意文件名
#     # print(inputfile)
#     file_counter+=1      # 读入脚本的每一个 输入文件，都将 变量的值 加 1 ； 输入文件总数= 所有读入的脚本文件
#     if inputfile.split('.')[1]=='csv':
#         with open(inputfile,'r',newline='') as csv_in_file:
#             file_reader=csv.reader(csv_in_file)
#             header=next(file_reader)
#             # print(header)     # 读取标题行
#             for row_list in file_reader:
#                 row_of_output=[]         # 记录要搜索的 数值项目
#                 for cols in range(len(header)):
#                     # print(cols)
#                     if cols==3:
#                         # 先使用 lstrip() 方法剥离字符串左侧 的美元符号，
#                         # 然后使用 replace() 方法用空字符串替换掉字符串中的逗号（这样可以有效 地删除逗号），
#                         # 再使用 strip() 方法剥离字符串两端的空格、制表符和换行符。
#                         cell_value=str(row_list[cols]).lstrip('$').replace(',','').strip()
#                         # print(cell_value)
#                         row_of_output.append(cell_value)
#                     else:
#                         cell_value=str(row_list[cols]).strip()
#                         # print(cell_value)
#                         row_of_output.append(cell_value)
#                 # print(row_of_output)
#                 # 将输入文件的 基础文件名 追加到 列表 row_of_output
#                 row_of_output.append(os.path.basename(inputfile))
#                 # print(row_of_output)
#                 # print(row_list)
#                 if row_list[0] in item_numbers_to_find:
#                     file_write.writerow(row_of_output)
#                     count_of_item_numbers+=1    # 所有输入文件中找到的数值 项目的数量
#                 line_counter+=1           # 跟踪 在所有输入文件中 找到的 数据行的数量
#
#     # 寻找 Excel中的 数据
#     elif inputfile.split('.')[1]=='xlsx' or inputfile.split('.')[1]=='xls':
#         workbook=open_workbook(inputfile)
#         for worksheet in workbook.sheets():
#             header=worksheet.row_values(0)
#             # print(header)      # 标题
#             # print(worksheet.nrows)
#             for row in range(1,worksheet.nrows):
#                 row_of_output=[]
#                 for cols in range(len(header)):
#                     # print(worksheet.cell_value(row,4))
#                     # print(worksheet.cell_type(row,cols))
#
#                     # if 代码块处理单元格类型为 3 的列，也就是包含代 表日期的数值的列。
#                     # 这个代码块使用 xlrd 模块的 xldate_as_tuple() 方法和 datetime 模块 的 date() 方法来保证这个列中的日期值在输出文件中保持原来的格式。
#                     # 只要这个值被转换 为具有日期形式的文本字符串，就使用 strip() 方法剥离字符串两端的空格、制表符和换 行符，
#                     if worksheet.cell_type(row,cols)==3:
#                         cell_value=xldate_as_tuple(worksheet.cell(row,cols).value,workbook.datemode)
#                         cell_value=str(date(*cell_value[0:3])).strip()
#                         # print(cell_value)
#                         row_of_output.append(cell_value)
#                     else:
#                         cell_value=str(worksheet.cell_value(row,cols)).strip()
#                         row_of_output.append(cell_value)
#                 # print(row_of_output)
#                 row_of_output.append(os.path.basename(inputfile))     # 还将 基础文件名称追加到列表中
#                 row_of_output.append(worksheet.name)      # 将工作表名 也加入了 列表
#                 # print(row_of_output)
#                 # print(str(worksheet.cell(row,0).value).split('.')[0])
#                 if str(worksheet.cell(row,0).value).split('.')[0].strip() in item_numbers_to_find:
#                     # print(row_of_output)
#                     file_write.writerow(row_of_output)
#                     count_of_item_numbers+=1      # 跟踪在所有输入文件中找到的数值项目的数量
#                 line_counter+=1          # 跟踪在所有输入文件中找到的数据行的数量。
#
# print("Number of files(文件数):",file_counter)
# print("Number of lines(行数):",line_counter)   # 打印出在所有输入文件和工作表中读取的行数
# print("Number of item numbers:",count_of_item_numbers)    # 打印出带有 要搜索的数值项 目的行数，这个数值可能包含重复计数
#
#
#
#
#
# import csv
# import glob
# import os
# from datetime import date
# # 入 datetime 模块的 date 方法和 xlrd 模块的 xldate_as_tuple 方法目的：确保我们从输入文件中提取的任何日期数据都 能以特定的形式保存到输出文件中。
# from xlrd import open_workbook,xldate_as_tuple
# item_number_file='item_numbers_to_ﬁnd.csv'        # 目标文件（要搜索的数值的项目）
# input_file='C:\\Users\王宝刚\Desktop\ﬁle_archive'
# output_file='item_data.csv'      # 将结果值输出
# item_numbers_to_find=[]
# # 第一：先读取 目标文件 内容
# with open(item_number_file,'r',newline='') as item_numbers_csv_file:
#     filereader=csv.reader(item_numbers_csv_file)
#     for row_list in filereader:
#         for row in row_list:
#             item_numbers_to_find.append(row)
# # print(item_numbers_to_find)
# # 第二： 打开 CSV 输出文件， 并创建 一个 filewriter 对象，准备写入 数据 到输出文件
# filewriter=csv.writer(open(output_file,'a',newline=''))
#
# file_counter=0
# line_counter=0
# count_of_item_counter=0
#
# # 第三：循环遍历，从文件夹中 搜索 需要的文件
# for in_file in glob.glob(os.path.join(input_file,'*.*')):
#     # 记录文件数量
#     file_counter+=1
#
#     # 对Excel中 的  ' .csv'  中的数据进行处理
#     if in_file.split('.')[1]=='csv':
#         # print(in_file)
#
#         # 打开 搜索到的 文件，读取内容
#         with open(in_file,'r',newline='') as csv_in_file:
#             filereader=csv.reader(csv_in_file)
#             header=next(filereader)        # 访问 标题
#             for row_list in filereader:
#                 row_of_output=[]
#                 for cols in range(len(header)):
#                     if cols==3:
#                         cell_value=str(row_list[cols]).lstrip('$').replace(',','').strip()
#                         row_of_output.append(cell_value)
#                     else:
#                         cell_value=str(row_list[cols]).strip()
#                         row_of_output.append(cell_value)
#                 row_of_output.append(os.path.basename(in_file))
#                 if row_list[0] in item_numbers_to_find:
#                     filewriter.writerow(row_of_output)
#                     count_of_item_counter+=1
#                 line_counter+=1
#
#         # 对Excel中 的 ' .xlsx' 或 ' .xls' 中的数据进行处理
#     elif  in_file.split('.')[1]=='xlsx' or in_file.split('.')[1]=='xls':
#         # 创建 工作表对象
#         workbook=open_workbook(in_file)
#         for worksheet in workbook.sheets():
#             try:
#                 header=worksheet.row_values(0)         # 工作表中的第一行
#             except IndexError:
#                 pass
#             # print(worksheet.row_values)
#             for row in range(1,worksheet.nrows):
#                 row_of_out=[]
#                 for cols in range(len(header)):
#                     # print((worksheet.cell(row,cols)).value)
#                     if worksheet.cell_type(row,cols)==3:
#                         # print(worksheet.cell(row,4).value)
#                         cell_value=xldate_as_tuple(worksheet.cell_value(row,cols),workbook.datemode)
#                         cell_value=str(date(*cell_value[0:3])).strip()
#                         # print(cell_value)
#                         row_of_out.append(cell_value)
#                 # print(row_of_out)
#                     else:
#                         cell_value=str(worksheet.cell_value(row,cols)).strip()
#                         row_of_out.append(cell_value)
#                 # print(row_of_out)
#                 row_of_out.append(os.path.basename(in_file))
#                 row_of_out.append(worksheet.name)
#                 if str(worksheet.cell_value(row,0)).split('.')[0].strip() in item_numbers_to_find:
#                     filewriter.writerow(row_of_out)
#                     count_of_item_counter+=1
#                 line_counter+=1


'''
为CSV文件中数据的任意数目分类计算统计量
'''
# 用途：在很多商业分析中，需要为一个特定时间段内的未知数目的分类计算统计量。
# 举例来说， 假设我们销售 5 种不同种类的产品，你想计算一下在某一年中对于所有客户的按产品种类 分类的总销售额。
# 因为客户具有不同的品味和偏好，他们在一年中购买的产品也是不同 的。有些客户购买了所有 5 种产品，有些客户则只购买了一种产品。
# 在这种客户购买习惯 之下，与每个客户相关的产品分类数目都是不同的。

# 问题：计算出你的客户在他们购买的每个服务包类别上花费的总时间（以月计 算）。

# 例如，如果你的一个客户 Tony Shephard 在 2014 年 2 月 15 日购买了铜牌服务包，在 2014 年 6 月 15 日购买了银牌服务包，在 2014 年 9 月 15 日购买了金牌服务包，
# 那么关于 Tony Shephard 的计算结果就是：“铜牌服务包：4 个月”“银牌服务包：3 个月”“金牌服务 包：从 2014 年 9 月 15 日至今”。
# 如果另一个客户 Mollie Adler 只购买了银牌服务包和金牌 服务包，那么关于 Mollie Adler 的计算结果中就不会包含铜牌服务包的任何信息。

# customer_category_history.csv 进行了分析，这个数据集包括 4 列数据：Customer Name、Category、Price 和 Date。
# 还包 括 6 个客户：John Smith、Mary Yu、Wayne Thompson、Bruce Johnson、Annie Lee 和 Priya Patel。同时包括 3 个服务包分类：铜牌、银牌和金牌。数据是以先按照客户姓名，再按照 日期的形式升序排列的。
# 现在我们已经有了数据集，其中包括客户在过去一年中购买的服务包，还有服务包的购买 日期或更新日期。接下来要做的就是编写 Python 代码来执行计算。

# import csv
# from datetime import date,datetime
# def date_diff(date1,date2):
#     try:
#         diff=str(datetime.strptime(date1,'%m/%d/%Y')-datetime.strptime(date2,'%m/%d/%Y')).split()[0]
#         # print(diff)
#     except:
#         diff=0
#     if diff=='0:00:00':
#         diff=0
#     return diff
# input_file='customer_category_history.csv '
# output_file='customer_cost_time.csv'
# packages={}           # 保存需要的信息
# previous_name='N/A'   # 客户姓名
# previous_package='N/A'   # 服务包类别
# previous_package_date='N/A'       # 服务包日期
# first_row=True
# today=date.today().strftime('%m/%d/%Y')
# with open(input_file,'r',newline='') as input_csv_file:
#     filereader=csv.reader(input_csv_file)
#     header=next(filereader)
#     # print(header)      # 标题
#     for row_list in filereader:
#         # print(row_list)
#         current_name=row_list[0]
#         current_package=row_list[1]
#         current_package_date=row_list[3]
#         if current_name not in packages:
#             packages[current_name]={}
#         if current_package not in packages[current_name]:
#             packages[current_name][current_package]=0
#         if current_name!=previous_package:
#             if first_row:
#                 first_row=False
#             else:
#                 diff=date_diff(today,previous_package_date)
#                 # print(type(diff))
#                 if previous_package not in packages[previous_name]:
#                     packages[previous_name][previous_package]=int(diff)
#                     # print(packages[previous_name][previous_package])
#                 else:
#                     packages[previous_name][previous_package] += int(diff)
#         else:
#             diff=date_diff(current_package_date,previous_package_date)
#             packages[previous_name][previous_package]+=int(diff)
#         previous_name=current_name
#         previous_package=current_package
#         previous_package_date=current_package_date
# header=['Customer Name','Category','Total Time(int Days)']
# with open(output_file,'w',newline='') as output_csv_file:
#     filewrite=csv.writer(output_csv_file)
#     filewrite.writerow(header)
#     for customer_name,customer_name_value in packages.items():
#         for package_category,package_category_value in packages[customer_name].items():
#             row_of_output=[]
#             # print(customer_name,package_category,package_category_value)
#             row_of_output.append(customer_name)
#             row_of_output.append(package_category)
#             row_of_output.append(package_category_value)
#             filewrite.writerow(row_of_output)
#             # print(row_of_output)


























