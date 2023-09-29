import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json
import arff

#####config

name = "year.arff"
feature_size = 64 + 1  # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = ['net profit / total assets', 'total liabilities / total assets', 'working capital / total assets',
             'current assets / short-term liabilities',
             '[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365',
             'retained earnings / total assets', 'EBIT / total assets', 'book value of equity / total liabilities',
             'sales / total assets', 'equity / total assets',
             '(gross profit + extraordinary items + financial expenses) / total assets',
             'gross profit / short-term liabilities', '(gross profit + depreciation) / sales',
             '(gross profit + interest) / total assets',
             '(total liabilities * 365) / (gross profit + depreciation)',
             '(gross profit + depreciation) / total liabilities',
             'total assets / total liabilities', 'gross profit / total assets', 'gross profit / sales',
             '(inventory * 365) / sales',
             'sales (n) / sales (n-1)', 'profit on operating activities / total assets', 'net profit / sales',
             'gross profit (in 3 years) / total assets',
             '(equity - share capital) / total assets', '(net profit + depreciation) / total liabilities',
             'profit on operating activities / financial expenses', 'working capital / fixed assets',
             'logarithm of total assets',
             '(total liabilities - cash) / sales', '(total liabilities - cash) / sales',
             '(current liabilities * 365) / cost of products sold', 'operating expenses / short-term liabilities',
             'operating expenses / total liabilities', 'profit on sales / total assets', 'total sales / total assets',
             '(current assets - inventories) / long-term liabilities', 'constant capital / total assets',
             'profit on sales / sales',
             '(current assets - inventory - receivables) / short-term liabilities',
             'total liabilities / ((profit on operating activities + depreciation) * (12/365))',
             'profit on operating activities / sales',
             'rotation receivables + inventory turnover in days', '(receivables * 365) / sales',
             'net profit / inventory',
             '(current assets - inventory) / short-term liabilities', '(inventory * 365) / cost of products sold',
             'EBITDA (profit on operating activities - depreciation) / total assets',
             'EBITDA (profit on operating activities - depreciation) / sales', 'current assets / total liabilities',
             'short-term liabilities / total assets', '(short-term liabilities * 365) / cost of products sold)',
             'equity / fixed assets', 'constant capital / fixed assets', 'working capital',
             '(sales - cost of products sold) / sales',
             '(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)',
             'total costs /total sales', 'long-term liabilities / equity', 'sales / inventory', 'sales / receivables',
             '(short-term liabilities *365) / sales', 'sales / short-term liabilities', 'sales / fixed assets']


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
    prompt = "Predict whether the company will face bankruptcy based on the financial profile attributes provided in the following text. " \
             "Respond with only 'no' or 'yes', and do not provide any additional information. \n"
    from_text = "The client has attributes: net profit / total assets: -0.046186, ...," \
                " sales / short-term liabilities: 5.7063, sales / fixed assets: 1.3882."
    prompt = prompt + f"For instance, '{from_text}' should be classified as 'no'. \nText: "
    for j in range(len(data)):
        text = 'The client has attributes: '
        for i in range(len(data[0]) - 1):
            sp = ', ' if i != len(data[0]) - 2 else '.'
            text = text + f'{mean_list[i]}: {str(data[j][i])}' + sp
        answer = 'no' if data[j][-1] == '0' else 'yes'
        # '0' is good (41314) and '1' is bad
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
    with open(f'gpt4-data/{dataname}.jsonl', 'w') as f:
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
    tmp_data = [row[-1] for row in test_data]
    _, gpt4_data = train_test_split(test_data, test_size=500, stratify=tmp_data, random_state=100)
    s1_data = [row for row in gpt4_data if row[-1] == '0']
    s2_data = [row for row in gpt4_data if row[-1] == '1']
    s_data = s2_data + s1_data[:100 - len(s2_data)]
    s_data = pd.DataFrame(s_data)
    np.random.seed(42)
    random_index = np.random.permutation(s_data.index)
    ss_data = s_data.reindex(random_index)
    # ss_data.to_csv('gpt4_rawdata.csv', index=False)
    json_save_gpt4(ss_data.values.tolist(), 'test_gpt4')


#####process
# load data on five .arff files
data = []  # 7027 + 10173 + 10503 + 9792 + 5910
for i in range(1, 6):
    with open(f'{i}{name}', 'r') as f:
        dataset = arff.load(f)
        data_tmp = dataset['data']
        # 将数据集添加到列表中
        data.extend(data_tmp)
data = np.array(data)
save_data, drop_data = train_test_split(data, test_size=0.8, stratify=data[:, -1], random_state=100)
che = get_num(save_data)

data = data_split(save_data)

save_gpt4_data(data[2])

save_name = ['train', 'valid', 'test']
for i in range(len(data)):
    _ = json_save(data[i], save_name[i])

# print(f"The length of the first query for testdata is : {len(_[0]['query'].split())}")
