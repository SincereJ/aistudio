import numpy as np
import pandas as pd
import csv


def csv_2_npy_npz():
    csv_file = 'train_sft.csv'
    data = pd.read_csv(csv_file)
    numpy_data = data.values

    # 将NumPy数组保存为train_ids.npy文件
    np.save('train_ids.npy', numpy_data)

    # 将NumPy数组保存为train_idx.npz文件
    npz_data = {str(i): numpy_data[i] for i in range(len(numpy_data))}
    np.savez('train_idx.npz', **npz_data)


# mv llama_openwebtext_100k_ids.npy ./data
# mv llama_openwebtext_100k_idx.npz ./data

if __name__ == '__main__':
    csv_2_npy_npz()