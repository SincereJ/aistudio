import numpy as np
import pandas as pd

# 读取CSV文件
csv_file = '../es/train_sft.csv'
data = pd.read_csv(csv_file)

# 将数据转换为NumPy数组
data_array = data.values

# 创建_idx.npz文件
np.savez(f'{csv_file.split(".")[0]}_idx.npz', data_array=data_array)