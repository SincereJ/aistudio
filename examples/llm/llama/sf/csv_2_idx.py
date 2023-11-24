import numpy as np
import pandas as pd
import csv

def csv_2_npz():

    # 读取CSV文件
    csv_file = '../es/train_sft.csv'
    data = pd.read_csv(csv_file)

    # 将数据转换为NumPy数组
    data_array = data.values

    # 创建_idx.npz文件
    np.savez(f'train_idx', data_array=data_array)

def csv_2_npy():
    csv_file = '../es/train_sft.csv'
    with open(csv_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

        # 将数据转换为NumPy数组
    numpy_data = np.array(data)

    # 保存为npy文件
    np.save('../data/train_ids.npy', numpy_data)


# mv llama_openwebtext_100k_ids.npy ./data
# mv llama_openwebtext_100k_idx.npz ./data

if __name__ == '__main__':
    csv_2_npy()