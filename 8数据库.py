'''
关系数据库 和 关系数据库管理系统
SQL：表示结构化查询语言，是一组应用非常广泛的与数据库进行交互的命令
'''
'''
Python 内置的 sqlite3模块
'''
# # 1.基本操作：
# # 要创建数据库中的表、在表中插入数据，以及在输出中获取数据并对行进行计数
# import sqlite3
# # 创建 sqlite3 内存数据库
# # 创建带有 4个属性的 sales 表
#
# conn=sqlite3.connect(':memory:')
# query="""CREATE  TABLE sales
# 	(customer VARCHAR(20),
# 	product VARCHAR(40),
# 	amount FlOAT,
# 	date DATE); """
# # 执行 SQL 命令
# conn.execute(query)
# # 保存
# conn.commit()
#
# # 在表中插入 几行数据
# data=[('Richard Lucas', 'Notepad', 2.50, '2014-01-02'),
# 	('Jenny Kim', 'Binder', 4.15, '2014-01-15'),
# 	('Svetlana Crow', 'Printer', 155.75, '2014-02-03'),
# 	('Stephen Randolph', 'Computer', 679.40, '2014-02-20')
# ]
# # ？在这里的作用：占位符（SQL命令中 使用的值）
# statement="INSERT INTO sales VALUES(?, ?, ?, ?)"
#
# conn.executemany(statement,data)
# # 如果对数据库做出修改，则 要再次 保存
# conn.commit()
#
# # 查询 sales 表
# # 光标对象：有多种方法：execute、executemany、fetchone、fetchmany 和 fetchall
# cursor=conn.execute("SELECT * FROM sales")
# row_list=cursor.fetchall()    # 取出所有行
#
# # 计算 查询结果中 行的数量
# row_count=0
# for row in row_list:
#     print(row)
#     row_count+=1
# print("Number of rows:{}".format(row_count))

# # 2.向表中插入新记录
# # 使用 CSV 文件向表中添加数据和更新表中数据
# import csv
# import sqlite3
# input_file='supplier_data.csv'
# # 创建sqlite3 内存数据库
# # 创建带有 5个属性的 Supplier 表
# conn=sqlite3.connect('Suppliers.db')
# cur=conn.cursor()
# create_table="""CREATE TABLE IF NOT EXISTS Suppliers
# (   Supplier_Name VARCHAR(20),
# 	Invoice_Number VARCHAR(20),
# 	Part_Number VARCHAR(20),
# 	Cost FLOAT,
# 	Purchase_Date DATE);"""
# # 执行
# cur.execute(create_table)
# # 保存
# conn.commit()
#
# # 读取 CSV文件
# # 向表中 插入数据
# statement="INSERT INTO Suppliers VALUES(?,?,?,?,?);"
# file_reader=csv.reader(open(input_file,'r'),delimiter=',')
# header=next(file_reader,None)
# for row in file_reader:
#     data=[]
#     for cols_index in range(len(header)):
#         data.append(row[cols_index])
#     # print(data)
#     cur.execute(statement,data)
# conn.commit()
# # print('')
#
# # 查询 suppliers 表
# output=cur.execute("SELECT * FROM Suppliers")
# row_list=output.fetchall()
# print(len(row_list))
# for row in row_list:
#     output_data=[]
#     for cols_index in range(len(row)):
#         output_data.append(row[cols_index])
#     # print(output_data)
#
# #  计算行的数量
# supplier_count=0
# for row in row_list:
#     print(row)
#     supplier_count+=1
# print("行数：{}".format(supplier_count))

# # 3.更新表中的数据
# import csv
# import sqlite3
# input_file='supplier_data.csv'
# conn=sqlite3.connect(':memory:')
# query="""CREATE  TABLE  IF NOT EXISTS sales
# 	(customer VARCHAR(20),
# 	product VARCHAR(40),
# 	amount FlOAT,
# 	date DATE); """
# conn.execute(query)
# conn.commit()
#
# # 插入数据
# data=[('Richard Lucas', 'Notepad', 2.50, '2014-01-02'),
# 	('Jenny Kim', 'Binder', 4.15, '2014-01-15'),
# 	('Svetlana Crow', 'Printer', 155.75, '2014-02-03'),
# 	('Stephen Randolph', 'Computer', 679.40, '2014-02-20')
# ]
#
# for tuple in data:
#     print(tuple)
# sql_insert="INSERT INTO sales VALUES(?,?,?,?) "
# conn.executemany(sql_insert,data)
# conn.commit()
#
# # 读取 CSV文件 并更新特定的行
# file_header=csv.reader(open(input_file,'r'),delimiter=',')
# header=next(file_header)
# # print(header)
# for row_list in file_header:
#     data=[]
#     for cols_index in range(len(header)):
#         data.append(row_list[cols_index])
#     # print(data)
#     # 为 组特定的 customer 更新 amount 值和 date 值
#     conn.execute("UPDATE sales SET amount=?, date=? WHERE customer=?;", data)
# conn.commit()
#
# # 查询 sales 表
# cur=conn.execute("SELECT * FROM sales")
# rows=cur.fetchall()
# # print(rows)
# for row in rows:
#     output=[]
#     for cols_index in range(len(row)):
#         output.append(row[cols_index])
#     print(output)


'''
MySQL 数据库
'''
# 1.向表中插入数据
# import csv
# import time
# import pymysql
# from datetime import datetime,date
# #CSV 输入文件的路径和 文件名
# input_file='supplier_data.csv'
# # 连接MYSQL 数据库
# conn=pymysql.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='wbg123456')
# cursor=conn.cursor()
#
# # 向 Suppliers 表中插入数据库
# file_reader=csv.reader(open(input_file,'r',newline=''))
# header=next(file_reader)
# for row in file_reader:
#     data=[]
#     for col_index in range(len(header)):
#         if col_index<4:
#             data.append(str(row[col_index]).lstrip('$').replace(',','').strip())
#         # print(type(row[col_index]))
#         else:
#             data.append(row[col_index])
#             month, day, year = data[4].split('/')
#             day=int(day)
#             month=int(month)
#             year=int('20'+year)
#             date_cell = date(year, month, day).strftime('%Y-%m-%d')
#             data[4]=date_cell
#     cursor.execute("INSERT INTO suppliers VALUES(%s, %s, %s, %s, %s);",data)
# # conn.commit()
#
# # # 删除表中全部数据
# # sql_del='delete from suppliers where true '
# # cursor.execute(sql_del)
# # conn.commit()
#
# # # 查询表中数据
# cursor.execute("select * from suppliers")
# row_tuple=cursor.fetchall()
# for rows in row_tuple:
#     row_list_output=[]
#     for col_index in range(len(rows)):
#         row_list_output.append(str(rows[col_index]))
#     print(row_list_output)


# 2.查询 一个表 并将输出 写入CSV 文件
# 数据表中有了数据之后，最常见的下一个步骤就是使用查询从表中取出一组数据，用来进行分析或满足某种商业需求。

# 1）查找 Cost列中的值 大于 1000.00的所有记录
# import csv
# import pymysql
# output_file='supplier_data2.csv'
# # 链接 MySQL
# conn=pymysql.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='wbg123456')
# # 创建光标，执行SQL语句，并将 修改提交到数据库
# cursor=conn.cursor()
# # 创建写文件 对象， 并写入标题
# file_write=csv.writer(open(output_file,'w',newline=''),delimiter=',')
# header=['Supplier Name','Invoice Number','Part Number','Cost','Purchase Date']
# file_write.writerow(header)
#
# # 查询 suppliers 表，并将（找 Cost列中的值 大于 70.00的所有记录） 写入 CSV文件
# sql_find='select * from suppliers where Cost>700.00;'
# cursor.execute(sql_find)
# row_tuple=cursor.fetchall()
# for row in row_tuple:
#     file_write.writerow(row)
#     print(row)

# 3.更新表中的记录
import csv
import pymysql
input_file='supplier_data.csv'
# 链接 MySQL
conn=pymysql.connect(host='localhost',port=3306,db='my_suppliers',user='root',passwd='wbg123456')
# 创建光标，执行SQL 语句
cursor=conn.cursor()
# 读取CSV文件并更新 特定的行
file_reader=csv.reader(open(input_file,'r',newline=''),delimiter=',')
header=next(file_reader,None)
for row_list in file_reader:
    # print(row_list)
    data=[]
    for col_index in range(len(header)):
        data.append(str(row_list[col_index]).strip())
    # print(data)

#     # 更新 特定的行
#
#     sql_update="update suppliers set Cost='%s' where Supplier_Name='%s';"
#     # 执行SQL 语句
#     cursor.execute(sql_update,data)
# conn.commit()

# # 删除表中内容
# sql_del='delete from suppliers where true '
# cursor.execute(sql_del)
# conn.commit()

# # 查询suppliers 表
cursor.execute('Select * from suppliers;')
row_tuple=cursor.fetchall()
for rows in row_tuple:
    out_list=[]
    for col_index in range(len(rows)):
        out_list.append(str(rows[col_index]))
    print(out_list)



















