import random

import numpy as np
import pandas as pd
import json

#####config
from sklearn.model_selection import train_test_split

name = "PortoSeguro.csv"
feature_size = 57 + 1  # Target_index = 0
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")


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
    prompt = "Identify whether or not to files a claim for the auto insurance policy holder using the following table attributes about individual financial profile. Respond with only \'yes\' or \'no\', and do not provide any additional information. "
    prompt = prompt + "And the table attributes that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). "
    prompt = prompt + "In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. "
    prompt = prompt + "Values of -1 indicate that the feature was missing from the observation. "
    from_text = "The client has attributes: ps_ind_01: 1.0, ps_ind_02_cat: 2.0, ps_ind_03: 0.0, ..., ps_calc_18_bin: 0.0, ps_calc_19_bin: 0.0, ps_calc_20_bin: 0.0."
    prompt = prompt + f"For instance, '{from_text}' should be classified as 'no'. \nText: "

    for j in range(len(data)):
        text = 'The client has attributes: '
        for i in range(1, len(data[0])):
            sd = str(data[j][i]) if len(str(data[j][i])) <= 4 else f'{data[j][i]:.2f}'
            sp = ', ' if i != len(data[0]) - 1 else '.'
            text = text + f'{mean_list[i - 1]}: ' + sd + sp
        answer = 'no' if data[j][0] == 0 else 'yes'
        gold = 0 if data[j][0] == 1 else 1
        # '0' is no  and '1'  is yes
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["yes", "no"],
             "gold": gold, 'text': text})
    return data_tmp


def json_save(data, dataname, mean_list, out_jsonl=False):
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


def json_save_gpt4(data, dataname, mean_list):
    data_tmp = process_table(data, mean_list)
    with open(f'gpt4-data/{dataname}.jsonl', 'w') as f:
        for i in data_tmp:
            json.dump(i, f)
            f.write('\n')
        print('-----------')
        print(f"{dataname}.jsonl write done")
    f.close()


def get_num(data):
    data_con = np.array(data)
    check = np.unique(data_con[:, 0])
    check1 = (data_con[:, 0] == check[0]).sum()
    check2 = (data_con[:, 0] == check[1]).sum()
    return check1, check2


def save_gpt4_data(test_data):
    tmp_data = [row[-1] for row in test_data]
    _, gpt4_data = train_test_split(test_data, test_size=500, stratify=tmp_data, random_state=100)

    s1_data = [row for row in gpt4_data if row[0] == 0]
    s2_data = [row for row in gpt4_data if row[0] == 1]
    s_data = s2_data + s1_data[:100 - len(s2_data)]
    s_data = pd.DataFrame(s_data)
    np.random.seed(42)
    random_index = np.random.permutation(s_data.index)
    ss_data = s_data.reindex(random_index)
    # ss_data.to_csv('gpt4_rawdata.csv', index=False)
    json_save_gpt4(ss_data.values.tolist(), 'test_gpt4', mean_list)


#####process
# column 1 is the target variable
data = pd.read_csv(name, sep=',', header=0).drop('id', axis=1)  # id dropped
mean_list = data.columns.values.tolist()
mean_list.pop(0)  # 'target' dropped
save_data, drop_data = train_test_split(data, test_size=0.98, stratify=data['target'], random_state=100)
# data = data.sample(n=int(len(data) * 0.1), random_state=42).values.tolist()  # all float
data = save_data.values.tolist()

che = get_num(data)
data = data_split(data)

test_data = data[2]

save_name = ['train', 'valid', 'test']
for i in range(len(data)):
    _ = json_save(data[i], save_name[i], mean_list)
