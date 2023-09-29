import random

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

#####config
name = "ccFraud.csv"
feature_size = 7 + 1   # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = ['gender', 'state', 'cardholder', 'balance', 'numTrans', 'numIntlTrans', 'creditLine']
mean_list_text = ['male or female', 'state number', 'number of cards', 'credit balance', 'number of transactions',
                  'number of international transactions', 'credit limit']


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
    prompt = "Detect the credit card fraud with the following financial profile. Respond with only \'good\' or \'bad\', and do not provide any additional information. "
    from_text = "The client is a female, the state number is 25, the number of cards is 1, the credit balance is 7000, the number of transactions is 16, the number of international transactions is 0, the credit limit is 6."
    prompt = prompt + f"For instance, '{from_text}' should be classified as 'good'. \nText: "
    for j in range(len(data)):
        text = 'The client '
        for i in range(len(data[0]) - 1):
            sp = ', ' if i != len(data[0]) - 2 else '.'
            if mean_list[i] == 'gender':
                subtext = f'is a male' if str(data[j][i]) == 1 else f'is a female'
                text = text + subtext + sp
            else:
                text = text + f'the {mean_list_text[i]} is {str(data[j][i])}' + sp
        answer = 'good' if data[j][-1] == 0 else 'bad'
        # '0' is good  and '1' is bad
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["good", "bad"],
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


def save_bias_data(data):
    columns = [i for i in range(8)]
    tt_data = pd.DataFrame(data[2], columns=columns)
    tt_data.to_csv('bias_data/ccfraud_test.csv', index=False)
    gg_data = pd.DataFrame(data[0], columns=columns)
    gg_data.to_csv('bias_data/ccfraud_train.csv', index=False)


def save_gpt4_data(test_data):
    tmp_data = [row[-1] for row in test_data]
    _, gpt4_data = train_test_split(test_data, test_size=500, stratify=tmp_data, random_state=100)
    s1_data = [row for row in gpt4_data if row[-1] == 0]
    s2_data = [row for row in gpt4_data if row[-1] == 1]
    s_data = s2_data + s1_data[:100 - len(s2_data)]

    s_data = pd.DataFrame(s_data, columns=['gender', 'state',
                                           'cardholder', 'balance', 'numTrans', 'numIntTrans',
                                           'creditLine', 'fraudRisk'])
    np.random.seed(42)
    random_index = np.random.permutation(s_data.index)
    ss_data = s_data.reindex(random_index)

    # ss_data.to_csv('gpt4_ccfraud_test.csv', index=False)
    json_save_gpt4(ss_data.values.tolist(), 'test_gpt4')
#####process

data = pd.read_csv(name, sep=',', header=0, names=[i for i in range(feature_size + 1)]).drop(0, axis=1)  # custID dropped
save_data, drop_data = train_test_split(data, test_size=0.99, stratify=data[8], random_state=100)
# data = data.sample(n=int(len(data) * 0.04), random_state=42).values.tolist()
data = save_data.values.tolist()

che = get_num(data)
data = data_split(data)

save_gpt4_data(data[2])
save_bias_data(data)

save_name = ['train', 'valid', 'test']
for i in range(len(data)):
    _ = json_save(data[i], save_name[i])
