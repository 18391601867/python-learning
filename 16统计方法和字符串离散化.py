import pandas as pd
import matplotlib.pyplot as plt
input_file='zscoredata .csv'
air_data=pd.read_csv(input_file)
# print(air_data.head())
# print(air_data.info())

# L, runtime 分布 情况
# 选择图形， 直方图
# 准备数据
l_data=air_data['L'].values
# print(l_data)
max_l_data=l_data.max()
min_l_data=l_data.min()
print(max_l_data-min_l_data)
# 计算组数
num_bin=int((max_l_data-min_l_data)//7)
# print(num)
#设置图像大小
plt.figure(figsize=(35,20),dpi=80)
# 绘图
plt.hist(l_data,num_bin)
plt.xticks(range(int(min_l_data),int(max_l_data+1),7),fontsize=20)
# 添加网格
plt.grid(alpha=0.5)
plt.show()




