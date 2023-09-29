import random

import numpy as np
import pandas as pd
import json

#####config
from sklearn.model_selection import train_test_split

name = "travel insurance.csv"
feature_size = 10 + 1  # target_index = 4
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = [('Agency', 'Name of agency'), ('Agency Type', 'Type of travel insurance agencies'),
             ('Distribution Channel', 'Distribution channel of travel insurance agencies'),
             ('Product Name', 'Name of the travel insurance products'), ('Duration', 'Duration of travel'),
             ('Destination', 'Destination of travel'), ('Net Sales', 'Amount of sales of travel insurance policies'),
             ('Commission', 'Commission received for travel insurance agency'), ('Age', 'Age of insured')
             ]


#####function
def data_preparation(da):
    # Duration > 731 replaced by 731
    da = [[row[i] if i != 5 or row[i] <= 731 else 731 for i in range(len(row))] for row in da]
    # Average Duration
    dura_colu = [row[5] for row in da]
    dura_mean = int(sum(dura_colu) / len(dura_colu))
    # Duration < 1 replace by Average Duration and Age > 99 replaced by 99
    for row in da:
        row[10] = 99 if row[10] > 99 else row[10]
        row[5] = dura_mean if row[5] < 1 else row[5]
    # 删除 Attribute: Gender
    da = [row[:9] + row[9 + 1:] for row in da]
    return da


def process_table(data, mean_list):
    data_tmp = []
    prompt = 'Identify the claim status of insurance companies using the following table attributes for travel insurance status. Respond with only \'yes\' or \'no\', and do not provide any additional information. And the table attributes including 5 categorical attributes and 4 numerical attributes are as follows: \n'
    for i in range(len(data[0]) - 1):  # data[0] (del Gender): 9 + 1 (5)
        st = "(categorical). \n" if type(data[0][i]) == str else "(numerical). \n"
        prompt = prompt + f'{mean_list[i][0]}: ' + mean_list[i][1] + ' ' + st
    prompt = prompt + 'For instance: \'The insurance company has attributes: Agency: CBH, Agency Type: Travel Agency, Distribution Chanel: Offline, Product Name: Comprehensive Plan, Duration: 186, Destination: MALAYSIA, Net Sales: -29, Commision: 9.57, Age: 81.\', should be classified as \'no\'. \nText: '

    for j in range(len(data)):
        text = 'The insurance company has attributes:'
        for i in range(len(data[0])):
            # i = 4: Claim Status
            if i < 4:
                text = text + f' {mean_list[i][0]}: ' + str(data[j][i]) + ','
            if i > 4:
                sy = '.' if i == len(data[0]) - 1 else ','
                text = text + f' {mean_list[i - 1][0]}: ' + str(data[j][i]) + sy
        answer = 'yes' if data[j][4] == 'Yes' else 'no'
        gold = 0 if data[j][4] == 'Yes' else 1
        # 'No' 62399 and Yes' 927
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["yes", "no"],
             "gold": gold, 'text': text})
    return data_tmp


def json_save(data, dataname, mean_list=mean_list, out_jsonl=False):
    data_tmp = process_table(data, mean_list)
    if out_jsonl:
        with open('{}.jsonl'.format(dataname), 'w') as f:
            for i in data_tmp:
                json.dump(i, f)
                f.write('\n')
            print('-----------')
            print("write done")
        f.close()
    df = pd.DataFrame(data_tmp)
    # 保存为 Parquet 文件
    parquet_file_path = f'data/{dataname}.parquet'
    df.to_parquet(parquet_file_path, index=False)
    return data_tmp


def get_num(data):
    data_con = np.array(data)
    check = np.unique(data_con[:, 4])
    check1 = (data_con[:, 4] == check[0]).sum()
    check2 = (data_con[:, 4] == check[1]).sum()
    return check1, check2


def save_bias_data(feature_size, test_data, train_data):
    columns = [i for i in range(feature_size - 1)]
    ss_data = pd.DataFrame(test_data, columns=columns)
    ss_data.to_csv('bias_data/TraIn_test.csv', index=False)
    tt_data = pd.DataFrame(train_data, columns=columns)
    tt_data.to_csv('bias_data/TraIn_train.csv', index=False)


#####process
data = pd.read_csv(name, sep=',', header=0, names=[i for i in range(feature_size)])
save_data, drop_data = train_test_split(data, test_size=0.8, stratify=data[4], random_state=100)
# data preprocessing

che = get_num(save_data)
data = data_preparation(save_data.values.tolist())

random.seed(10086)
train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
train_data = [data[i] for i in train_ind]

index_left = list(filter(lambda x: x not in train_ind, [i for i in range(len(data))]))
dev__ind = random.sample(index_left, int(len(data) * dev_size))
dev_data = [data[i] for i in dev__ind]

index_left = list(filter(lambda x: x not in train_ind + dev__ind, [i for i in range(len(data))]))
test_data = [data[i] for i in index_left]

save_bias_data(feature_size, test_data, train_data)

test_prompt_data = json_save(test_data, 'test')
train_prompt_data = json_save(train_data, 'train')
dev_prompt_data = json_save(dev_data, 'valid')
