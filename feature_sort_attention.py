import numpy as np
import pandas as pd

# 加载权重矩阵
weight_attention = np.load('./weight/C-attention_weights.npy')

# 计算每一列的均值
column_means = np.mean(weight_attention, axis=1)

# print(column_means.shape)

# 读取 Excel 文件
excel_file = 'C-特征.xlsx'  # 替换为您的 Excel 文件路径
df = pd.read_excel(excel_file, sheet_name=0).iloc[:, :-1]  # sheet_name=0 表示第一个工作表

# 获取表头并将其转换为列表
headers = df.columns.tolist()

# print(len(headers))

# 创建一个字典，将每个均值放到对应位置的表头后
mean_dict = {header: mean for header, mean in zip(headers, column_means)}

# 按照均值排序字典
sorted_mean_dict = sorted(mean_dict.items(), key=lambda item: item[1], reverse=True)

for header, mean in sorted_mean_dict:
    print(f"{header}: {mean}")

# 指定要保存的文件名
output_file = './sort/C-sorted_feature.txt'

# 打开文件并写入排序后的结果
with open(output_file, 'w') as f:
    for header, mean in sorted_mean_dict:
        f.write(f"{header}: {mean}\n")

print(f"Results have been saved to {output_file}")