'''
解析Excel 文件用的是xlrd 库
'''
'''
处理 Excel 文件主要有三个库。
• xlrd 读取 Excel 文件。
• xlwt 向 Excel 文件写入，并设置格式。
• xlutils 一组 Excel 高级操作工具（需要先安装 xlrd 和 xlwt）。
'''
import xlrd
book=xlrd.open_workbook('tuanyuan.xlsx')
'''
'sheet1' 为表中的子表， 一定要注意格式
'''
sheet=book.sheet_by_name('sheet1')
# 返回行数
# print(sheet.nrows)


# for i in range(sheet.nrows):
#     # 返回行数 （索引）
#     # print(i)
#     # 返回 行值
#     # print(sheet.row_values(i))
#     row=sheet.row_values(i)
#     '''
#      提取数据
#     '''
#     # 方法一：嵌套循环，返回元素
#     # for cell in row:
#     #     print(cell)

# # 方法二：计数器
# count=0
# for i in range(sheet.nrows):
#     if  count<20:
#         if i<5:
#             row = sheet.row_values(i)
#             print(i,row)
#         count+=1
# print("Count:",count)

# # 修改如下：（字典形式）
# count=0
# data={}
# for i in range(sheet.nrows):
#     if count<20:
#         if i<5:
#             row=sheet.row_values(i)
#             lyst=row[1]
#             data[lyst]={}
#             count+=1
# print(data)

# # 1.要确定工作簿中工作表的数量、名称和每个工作表中行列的数量，
# # 注：一个工作簿中有多个工作表
# from xlrd import open_workbook
# input_file='Customer_sales.xlsx'
# workbook=open_workbook(input_file)
# print("Number of worksheets(工作表数量):",workbook.nsheets)
# for worksheet in workbook.sheets():
#     print("Worksheet name:",worksheet.name,"\tRows(行数):",worksheet.nrows,'\tColumns(列数):',worksheet.ncols)

# 2.处理单个工作表
# 尽管 Excel 工作簿可以包含多个工作表，有些时候你也只是需要一个工作表中的数据。此
# 外，只要你知道如何分析一个工作表，就可以很容易地扩展到分析多个工作表。

# 1）读写 Excel 文件
'''
• xlrd 读取 Excel 文件。
• xlwt 向 Excel 文件写入，并设置格式。
• xlutils 一组 Excel 高级操作工具（需要先安装 xlrd 和 xlwt）。
'''
# 第一种：
# from xlrd import open_workbook
# from xlwt import Workbook
# input_file='Customer_sales.xlsx'
# output_file='Customer_sales1.xlsx'
# # 实例化工作簿
# output_workbook=Workbook()
# # 添加工作表
# output_worksheet=output_workbook.add_sheet('Customer4')
# with open_workbook(input_file) as workbook:
#     worksheet=workbook.sheet_by_name('Customer1')
#     for row_index in range(worksheet.nrows):
#         for columns_index in range(worksheet.ncols):
#             # 码使用 xlwt 的 write 函数和行与列的索引将每个单元格的值写入输出文件的工作表
#             output_worksheet.write(row_index,columns_index,worksheet.cell_value(row_index,columns_index))
# output_workbook.save(output_file)

# 注： Purchase Date 列（也就是第 E 列）中的日期显示为数值，不是日期。
# Excel 将日期和时间保存为浮点数，这个浮点数代表从 1900 年 1 月 0 日开始经过的日期
# 数，加上一个 24 小时的小数部分。例如，数值 1 代表 1900 年 1 月 1 日，因为从 1900 年 1
# 月 0 日过去了 1 天。因此，这一列中的数值代表日期，但是没有格式化为日期的形式。
# xlrd 扩展包提供了其他函数来格式化日期值。

# # 格式化日期数据：
# from datetime import date
# from xlrd import open_workbook,xldate_as_tuple
# from xlwt import Workbook
# input_file='Customer_sales.xlsx'
# output_file='Customer_sales2.xlsx'
# # 实例化工作簿
# output_workbook=Workbook()
# # 新建 工作表
# output_worksheet=output_workbook.add_sheet('Customer4')
# with open_workbook(input_file) as workbook:
#     # 获取 已存在的工作表
#     worksheet=workbook.sheet_by_name('Customer2')
#     for row_index in range(worksheet.nrows):
#         # row_list_output=[]
#         for cols_index in range(worksheet.ncols):
#             if cols_index==4 and row_index!=0:                 # 除去 第一行 ；
#                 date_cell=xldate_as_tuple(worksheet.cell_value(row_index,cols_index),workbook.datemode)
#                 # 日期格式转化
#                 date_cell=date(*date_cell[0:3]).strftime('%m/%d/%Y')
#                 # row_list_output.append(date_cell)
#                 output_worksheet.write(row_index,cols_index,date_cell)
#             else:
#                 non_date_cell=worksheet.cell_value(row_index,cols_index)
#                 # row_list_output.append(non_date_cell)
#                 output_worksheet.write(row_index,cols_index,non_date_cell)
# output_workbook.save(output_file)

# # 第二种： pandas
# import pandas as pd
# from xlwt import Workbook
# input_file='Customer_sales.xlsx'
# output_file='Customer_sales3.xlsx'
# output_workbook=Workbook()
# output_worksheet=output_workbook.add_sheet('Customer4')
# data_frame=pd.read_excel(input_file,sheet_name='Customer2')
# # print(data_frame)
# data_write=pd.ExcelWriter(output_file)
#
# data_frame.to_excel(output_worksheet,sheet_name='Customer4',index=False)
# print(data_frame)
# output_workbook.save(output_file)


# 3.筛选特定的行
# 1) 行中的值满足某个条件
# from datetime import date
# from xlrd import open_workbook,xldate_as_tuple
# from xlwt import Workbook
# input_file='Customer_sales.xlsx'
# output_file='Customer_sales2.xlsx'
# sale_index=3
# # 实例化工作簿
# output_workbook=Workbook()
# # 新建 工作表
# output_worksheet=output_workbook.add_sheet('Customer4')
# with open_workbook(input_file) as workbook:
#     # 获取 已存在的工作表
#     worksheet=workbook.sheet_by_name('Customer2')
#     for row_index in range(1,worksheet.nrows):
#         row_list_output = []
#         sale_amount = worksheet.cell_value(row_index, sale_index)
#         if sale_amount > 1400.0:
#             for cols_index in range(worksheet.ncols):
#                 if cols_index==4:                 # 除去 第一行 ；
#                     date_cell=xldate_as_tuple(worksheet.cell_value(row_index,cols_index),workbook.datemode)
#                     # 日期格式转化
#                     date_cell=date(*date_cell[0:3]).strftime('%m/%d/%Y')
#                     row_list_output.append(date_cell)
#                     output_worksheet.write(row_index,cols_index,date_cell)
#                 else:
#                     non_date_cell=worksheet.cell_value(row_index,cols_index)
#                     row_list_output.append(non_date_cell)
#                     output_worksheet.write(row_index,cols_index,non_date_cell)
#         for row_value in row_list_output:
#             if row_value!='':
#                 print(row_list_output)
# output_workbook.save(output_file)

# # 第二种：pandas
# from openpyxl.workbook import Workbook
# import pandas as pd
# input_file='Customer_sales.xlsx'
# output_file='../Customer_sales2.xlsx'
# data_frame=pd.read_excel(input_file,sheet_name='Customer1',index_col=None)
# # print(data_frame)
# # print(data_frame['Sale Amount'])
# data_frame_value_meets_condition=data_frame[data_frame['Sale Amount'].astype(float)>1400.0]
# # print(data_frame_value_meets_condition)
# writer=pd.ExcelWriter(output_file)
# data_frame_value_meets_condition.to_excel(writer,sheet_name='Customer4',index=False,encoding='gbk')
# # print(data_frame_value_meets_condition)

# 2)行中的值属于 某个集合
from datetime import date
from xlrd import open_workbook,xldate_as_tuple
from xlwt import Workbook
input_file='Customer_sales.xlsx'
output_file='Customer_sales2.xlsx'
output_workbook=Workbook()
output_worksheet=output_workbook.add_sheet('Sheet')
important_dates= ['01/24/2013', '01/31/2013']
date_column_index=4
with open_workbook(input_file) as workbook:
    worksheet=workbook.sheet_by_name('Customer1')
    data=[]
    header=worksheet.row_values(0)
    # print(header)
    data.append(header)
    # print(data)
    for row_index in range(1,worksheet.nrows):
        purchase_datetime=xldate_as_tuple(worksheet.cell_value(row_index,date_column_index),workbook.datemode)
        purchase_date=date(*purchase_datetime[0:3]).strftime('%m/%d/%Y')
        # print(purchase_date)
        row_list=[]
        if purchase_date in important_dates:
            for cols_index in range(worksheet.ncols):
                if cols_index==4:
                    date_cell=xldate_as_tuple(worksheet.cell_value(row_index,cols_index),workbook.datemode)

                    date_cell=date(*date_cell[0:3]).strftime('%m/%d/%Y')
                    row_list.append(date_cell)
                else:
                    non_date_cell=worksheet.cell(row_index,cols_index)
                    row_list.append(non_date_cell)
        if row_list:
            data.append(row_list)
    for list_index,output_list in enumerate(data):
        # print(list_index)
        # print(output_list)
        for element_index,element in enumerate(output_list):
            # print(element_index)
            print(element)
            output_worksheet.write(list_index,element_index,element)
        # # print(row_list)
output_workbook.save(output_file)







