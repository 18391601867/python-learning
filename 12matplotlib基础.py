
'''1.折线图'''
'''以折线的上升 或下降来表示统计 数量的增减变化的统计图'''
# import matplotlib.pyplot as plt
# x=[x for x in range(1,20,2)]
# # print(x)
# y=[15,34,23,20,14,20,35,27,19,28]
# # 设置 图片大小； 每英寸上 点的像素 为 80
# fig=plt.figure(figsize=(20,8),dpi=80)
# # 通过plot 绘制折线图
# plt.plot(x,y,'r')
#
# #设置 指定的x 轴刻度
# plt.xticks(x)
# plt.yticks(range(min(x),max(y)+1))
# # 保存
# # plt.savefig('')
# plt.show()

# # 例1：
# import matplotlib.pyplot as plt
# import random
# # 设置字体
# plt.rcParams['font.sans-serif']=['Simhei']
#
# t=[i for i in range(0,120)]
# T=[random.randint(20,35) for i in range(120)]
# # 设置图片大小
# fig=plt.figure(figsize=(40,20),dpi=50)
# plt.plot(t,T,'r')
# # 设置坐标轴格式
# _x=t[::3]
# # _xticks_labels=['hello,{}'.format(i) for i in _x]
# _xticks_labels=['10点{}分'.format(i) for i in range(60)]
# _xticks_labels+=['11点{}分'.format(i) for i in range(60)]
# # 取步长，数字和字符串一一对应， 数据长度一样
# # rotation=90 x 轴标签旋转90 度
# plt.xticks(_x,_xticks_labels[::3],rotation=45,fontsize=15)
#
# # 添加描述信息
# plt.xlabel('时间',fontsize=20)
# plt.ylabel('温度',fontsize=20)
# plt.title('10：00到12;00气温每分钟的变化情况',fontsize=20)
# plt.show()

# 例2
# import matplotlib.pyplot as plt
# x=[age for age in range(11,31)]
# y=[1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
# y1=[2,4,1,2,4,2,4,3,1,0,2,4,2,1,3,1,1,2,1,0]
# plt.rcParams['font.sans-serif']=['Simhei']
# # 设置图片大小
# fig=plt.figure(figsize=(35,20),dpi=80)
# # 画图
# plt.plot(x,y,'r',label='同桌',linestyle=':',linewidth=5)
# plt.plot(x,y1,'g',label='自己',linestyle='--')
# # 设置x 轴 刻度
# _xtick_labels=["{}岁".format(i) for i in x]
# plt.xticks(x,_xtick_labels,fontsize=18)
# plt.yticks(y,fontsize=18)
# plt.xlabel('年龄',fontsize=16)
# plt.ylabel('男/女朋友个数',fontsize=16)
# # 绘制网格
# plt.grid(alpha=0.5,linestyle='-.')
#
# # 添加图例
# plt.legend(fontsize=25,loc='upper left')    # 图例位置 设置
# plt.show()


'''2.直方图'''
'''由一系列高度不等的纵向条纹 或 线段表示数据分布的情况'''
'''一般 用横轴表示数据范围， 纵向表示分布 情况'''
# 特点： 绘制连续性的数据， 展示一组 或者多组数据的分布状况
'''未统计数据'''
# import matplotlib.pyplot as plt
# fig=plt.rcParams['font.sans-serif']=['Simhei']
# x=[1,9,3,10,2,4,3,10,3,4,4,5,6,5,4,3,3,1,8,9,2,6,1,2,8,2,4,3,7,6,2,4,2,1,3,1,9,2,1,7]
# # 组距为4
# d=1
# num=(max(x)-min(x))//d
# plt.hist(x,num)
# plt.xticks(range(min(x),max(x)+d,d))
# plt.grid(alpha=0.5)
# plt.show()

'''已经统计过的 数据'''
import matplotlib.pyplot as plt
# interval=[0,5,10,15,20,25,30,35,40,45,60,90]
# width=[5,5,5,5,5,5,5,5,5,15,30,60]
# quality=[800,300,567,764,539,239,400,687,519,520,500,580]
# # print(len(interval),len(width),len(quality))
# plt.bar(range(12),quality,width=1)
#
# # 设置 x 轴 刻度
# _x=[i-0.5 for i in range(13)]
# _xtick_labels=interval+[150]
# plt.xticks(_x,_xtick_labels)
# plt.grid(alpha=0.5)
# plt.show()


'''3.条形图'''
'''排列在工作表的列 或行中的数据 可以绘制到条形图中'''
# # 特点： 绘制连离散的数据，能够 一眼 看出 各个数据的大小， 比较数据之间的差别
'''正向图'''
# import matplotlib.pyplot as plt
# fig=plt.rcParams['font.sans-serif']=['Simhei']
# x=['ABC','BCD','CDE','DEF','EFG','FGH','GHI','HIJ']
# y=[50,60,20,30,10,70,40,60]
# plt.bar(x,y,width=0.5)
# plt.xticks(x)
# plt.xlabel('种类')
# plt.ylabel('数量')
# plt.show()


'''绘制横着的条形图'''
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['Simhei']
# x=['ABC','BCD','CDE','DEF','EFG','FGH','GHI','HIJ']
# y=[50,60,20,30,10,70,40,60]
# fig=plt.figure(figsize=(35,20),dpi=80)
# plt.barh(x,y,height=0.5)
# plt.xticks(y,fontsize=30)
# plt.yticks(x,fontsize=30)
# plt.grid(alpha=0.5)
# plt.xlabel('数量',fontsize=30)
# plt.ylabel('种类',fontsize=30)
# plt.show()


'''绘制 多次条形图'''
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['Simhei']
# x=['ABC','BCD','CDE','DEF','EFG','FGH','GHI','HIJ']
# y1=[50,60,20,30,10,70,40,60]
# y2=[100,30,60,20,50,70,40,120]
# y3=[200,180,190,300,600,200,70,100]
# plt.figure(figsize=(30,25),dpi=80)
#
# bar_width=0.3
# # 设置 x 轴
# x1=list(range(len(x)))
# x2=[i+bar_width for i in x1]
# x3=[i+bar_width*2 for i in x1]
#
# plt.bar(x1,y1,width=0.3,label='第一天')
# plt.bar(x2,y2,width=0.3,label='第二天')
# plt.bar(x3,y3,width=0.3,label='第三天')
# plt.xticks(x2,x,fontsize=20)
# plt.legend('upper right',fontsize=20)
#
# plt.show()


'''4.散点图（scatter）'''
''' 用两组数据构成多个 坐标点， 考察坐标点的分布， 判断两变量之间 是否存在 某种关系 或 总结坐标点的分布模式'''
# # 特点： 判断变量之间 是否存在数量关联 趋势， 展示离群点（分布规律）
# import matplotlib.pyplot as plt
# import random
# # 设置字体
# plt.rcParams['font.sans-serif']=['Simhei']
# # T1=[random.randint(20,40) for i in range(1,31)]
# # T2=[random.randint(20,40) for j in range(1,32)]
# T1=[15,18,20,15,25,28,22,30,25,24,20,28,18,20,22,24,19,18,17,23,26,27,30,25,28,29,23,20,25,24]
# T2=[26,26,28,19,21,17,16,19,20,20,19,22,23,17,20,21,20,22,15,11,15,5,13,17,10,11,10,9,6,8,3]
# t1=[month for month in range(1,31)]
# t2=[month for month in range(41,72)]
# # 设置图像大小
# fig=plt.figure(figsize=(35,20),dpi=80)
#
# # 画图
# # linewidths=5 点的粗细
# plt.scatter(t1,T1,color='g',linewidths=5,label='3月份')
# plt.scatter(t2,T2,label='11月份')
#
# # 调整 x 轴的刻度
# _x=list(t1)+list(t2)
# _xtick_labels=['3月{}日'.format(i) for i in t1]
# _xtick_labels+=['10月{}日'.format(i-40) for i in t2]
# plt.xticks(_x[::3],_xtick_labels[::3],fontsize=20,rotation=45)
#
# #添加图例
# plt.legend(loc='upper left',fontsize=20)
#
# # 添加描述信息
# plt.xlabel('时间',fontsize=20)
# plt.ylabel('温度',fontsize=20)
# plt.show()


