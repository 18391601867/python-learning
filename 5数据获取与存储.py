
# 对数据的处理
# 1、确定数据源的可靠性
# 2、真实性核查
#3、数据可读性、数据清洁度和数据寿命
#4.寻找数据
# 教育数据（http://datainventory.ed.gov/InventoryList）
# • 选举结果（http://www.fec.gov/pubrec/electionresults.shtml）
# • 人口普查数据（http://census.ire.org/）
# • 环境数据（https://www.epa.gov/enviro/about-data）
# • 劳工统计数据（http://www.bls.gov/）

# LexisNexis（http: // www.lexisnexis.com /）
# • 谷歌学术搜索（https: // scholar.google.com /）
# • 康奈尔大学arXiv项目（http: // arxiv.org /）
# • UCI机器学习数据集（http: // archive.ics.uci.edu / ml /）
# • 通用数据集倡议（http: // www.commondataset.org /）

#  开放科学数据云（https://www.opensciencedatacloud.org/publicdata/）
#  • 开放科学目录（http://www.opensciencedirectory.net/）
#  • 世界卫生组织数据（http://www.who.int/gho/database/en/）
# • Broad 研究所开放数据（http://www.broadinstitute.org/scientiﬁc-community/data）
# • 人类连接组项目（神经通路映射）（http://www.humanconnectomeproject.org/）
# • UNC 精神病基因组协会（http://www.med.unc.edu/pgc/）
# • 社会科学数据集（http://3stages.org/idata/）
# • CDC 医学数据（http://www.cdc.gov/nchs/fastats/

#5.数据存储

'''
数据库分为 关系型型数据库 和 非关系型数据库
'''
# 1、关系型数据库（Mysql和 PostgreSQL）
# 关系型数据库通常使用一系列唯一标识符来匹配数据集。
# 在 SQL 里我们一般把这些标识符 叫作 ID。这些 ID 可以被其他数据集所用，用来查询和匹配数据连接。
# 在这些连接好的数 据库中，我们可以进行 join 操作，在许多不同的数据库中同时访问连接的数据
# 1）如果你熟悉 MySQL（或正在学习 MySQL），想要使用 MySQL 数据库，那么用 Python 连 接 MySQL 是很容易的。
# 你需要做的只有两步。第一步，你必须安装 MySQL 驱动程序。
# 第二步，你应该用 Python 发送验证信息（用户名、密码、主机名、数据库名称

# 2）如果你熟悉PostgreSQL（或正在学习PostgreSQL），想要使用PostgreSQL 数据库，那么用 Python 连接 PostgreSQL 也是很容易的。
# 你也只需要做两步：安装驱动程序，用 Python 连接

# 2、非关系型数据库（NoSQL）
# 1）NoSQL 以及其他非关系型数据库将数据保存成平面格式（ﬂat format），通常是JSON 格 式
# 2）如果你的数据具有非关系型数据库结构，或者你希望在实践中学习，
# 那么用Python 连 接 NoSQL 数据库是非常简单的。
# 虽然有很多选择，但最流行的NoSQL 数据库框架之 一是MongoDB（http://mongodb.org/）。要使用MongoDB，你需要首先安装驱动程序 （http://docs.mongodb.org/ecosystem/drivers/python/）， 然 后 用Python 来 连 接。

# import pymysql
# db=conn=pymysql.connect(
#     host='localhost',
#     port=3306,
#     user='root',
#     passwd='wbg123456',
#     db='mydb',
#     charset='utf8'
# )
# # 构建游标对象
# cursor=db.cursor(pymysql.cursors.DictCursor)
# # 编辑 sql 语句(创建表)
# sql="create table stuinfo11(\
#     stu_id int auto_increment primary key comment '学号(主键)',\
#     sname varchar(255) comment '学生姓名',\
#     sex enum('男','女')  comment '性别',\
#    age  tinyint comment '年龄',\
#    city varchar(64) comment '所在城市'\
# )engine=innodb;"
# # 执行sql 语句
# # cursor.execute(sql)
#
# # 插入语句
# sql1="insert into stuinfo11(stu_id,sname,sex,age,city) values(null,'小张',1,18,'深圳'),(null,'小花',2,20,'上海'),(null,'李四',1,19,'北京'),(null,'马云',1,50,'不知道'),(null,'艾琳',2,28,'北京'),(null,'刘强东',1,48,'宿迁'),(null,'赵小叶',2,35,'上海'),(null,'徐磊',1,35,'深圳')"
#
# cursor.execute(sql1)
# info=cursor.fetchall()
# db.commit()
# cursor.close()
# db.close()
# print(info)



