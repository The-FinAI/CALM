import random

import numpy as np
import pandas as pd
import json

#####config
from sklearn.model_selection import train_test_split

name = "creditcard.csv"
feature_size = 29 + 1  # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = [f'V{i}' for i in range(1, feature_size - 1)]
mean_list.append('Amount')


#####function
def data_split(data):
    random.seed(10086)

    train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
    train_data = [data[i] for i in train_ind]

    index_left = list(set(list(range(len(data)))) - set(train_ind))
    dev__ind = random.sample(index_left, int(len(data) * dev_size))
    dev_data = [data[i] for i in dev__ind]

    index_left = list(set(index_left) - set(dev__ind))
    test_data = [data[i] for i in index_left]

    return train_data, dev_data, test_data


def process_table(data, mean_list):
    data_tmp = []
    prompt = "Detect the credit card fraud using the following financial table attributes. " \
             "Respond with only 'yes' or 'no', and do not provide any additional information. " \
             "Therein, the data contains 28 numerical input variables V1, V2, ..., and V28 which are the result of " \
             "a PCA transformation and 1 input variable Amount which has not been transformed with PCA. " \
             "The feature 'Amount' is the transaction Amount, " \
             "this feature can be used for example-dependant cost-sensitive learning. "
    from_text = "The client has attributes: V1: 0.144, V2: 0.358, V3: 1.220, V4: 0.331, V5: -0.273, V6: 0.429, " \
                "V7: -0.307, V8: -0.577, V9: 0.116, V10: -0.337, V11: 1.016, V12: 1.043, V13: -0.527, V14: 0.160, " \
                "V15: -0.951, V16: -0.452, V17: 0.166, V18: -0.446, V19: 0.036, V20: -0.275, V21: 0.768, " \
                "V22: -0.051, V23: -0.180, V24: 0.067, V25: 0.741, V26: 0.477, V27: 0.152, V28: 0.201, Amount: 6.990."
    prompt = prompt + f"For instance, '{from_text}' should be classified as 'no'. \nText: "
    for j in range(len(data)):
        text = 'The client has attributes: '
        for i in range(len(data[0]) - 1):
            sp = ', ' if i != len(data[0]) - 2 else '.'
            text = text + f'{mean_list[i]}: {data[j][i]:.3f}' + sp
        answer = 'no' if data[j][-1] == 0 else 'yes'
        # '0' is good  and '1' is bad
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["no", "yes"],
             "gold": int(data[j][-1]), 'text': text})
    return data_tmp


def json_save(data, dataname, mean_list=mean_list, out_jsonl=False):
    data_tmp = process_table(data, mean_list)
    if out_jsonl:
        with open('{}.jsonl'.format(dataname), 'w') as f:
            for i in data_tmp:
                json.dump(i, f)
                f.write('\n')
            print('-----------')
            print(f"{dataname}.jsonl write done")
        f.close()
    df = pd.DataFrame(data_tmp)
    # 保存为 Parquet 文件
    parquet_file_path = f'data/{dataname}.parquet'
    df.to_parquet(parquet_file_path, index=False)
    return data_tmp


def json_save_gpt4(data, dataname, mean_list=mean_list):
    data_tmp = process_table(data, mean_list)
    with open('gpt4-data/{}.jsonl'.format(dataname), 'w') as f:
        for i in data_tmp:
            json.dump(i, f)
            f.write('\n')
        print('-----------')
        print(f"{dataname}.jsonl write done")
    f.close()


def get_num(data):
    data_con = np.array(data)
    check = np.unique(data_con[:, -1])
    check1 = (data_con[:, -1] == check[0]).sum()
    check2 = (data_con[:, -1] == check[1]).sum()
    return check1, check2


def save_gpt4_data(test_data):
    tmp_data2 = [row for row in test_data if row[-1] == 0]
    tmp_data1 = [row for row in test_data if row[-1] == 1]
    gpt4_data = tmp_data1 + tmp_data2[:100 - len(tmp_data1)]

    s_data = pd.DataFrame(gpt4_data)
    np.random.seed(42)
    random_index = np.random.permutation(s_data.index)
    ss_data = s_data.reindex(random_index)
    # ss_data.to_csv('gpt4_rawdata.csv', index=False)
    json_save_gpt4(ss_data.values.tolist(), 'test_gpt4')


#####process

data = pd.read_csv(name, sep=',', header=0, names=[i for i in range(feature_size + 1)]).drop(0, axis=1)  # Time dropped
save_data, drop_data = train_test_split(data, test_size=0.96, stratify=data[30], random_state=100)
# data = data.sample(n=int(len(data) * 0.1), random_state=42).values.tolist()
data = save_data.values.tolist()

che = get_num(data)
data = data_split(data)
save_gpt4_data(data[2])

save_name = ['train', 'valid', 'test']
for i in range(len(data)):
    _ = json_save(data[i], save_name[i])
