import random

import numpy as np
import pandas as pd
import json

#####function
from sklearn.model_selection import train_test_split

mean_list = ['Primary key of the record', 'Date when the declaration is reported',
             'Customs office that receives the declaration', 'Type of the declaration process',
             'Code for import type', 'Code for import use', 'Distinguish tariff payment type',
             'Nine modes of transport', 'Person who declares the item', 'Consumer who imports the item',
             'Overseas business partner which supplies goods to Korea', 'Delivery service provider',
             '6-digit product code', 'Country from which a shipment has or is scheduled to depart',
             'Country of manufacture, production or design, or where an article or product comes from',
             'Tax rate of the item', 'Tax types', 'Way of indicating the country of origin',
             'Mass without any packaging', 'Assessed value of an item']


def process_table(data, attributes):
    data_tmp = []
    prompt = "Identify the provided customs import declaration information to determine whether " \
             "it constitutes customs fraud that attempts to reduce customs duty or not. " \
             "The answer must be 'no' or 'yes', and do not provide any additional information. "
    prompt = prompt + "This Import Declaration consists of 20 data attributes, including Declaration ID, Date, " \
                      "Office ID, Process type, Import type, Import use, Payment type, Mode of transport, " \
                      "Declarant ID, Importer ID, Seller ID, Courier ID, HS6 code, Country of departure, " \
                      "Country of origin, Tax rate, Tax type, Country of origin indicator, Net mass and Item price. "
    from_text = "This customs import declaration has attributes: Declaration ID: 97061800, " \
                "Date: 2020-01-01, Office ID: 30, Process Type: B, ..., Item Price: 372254.4."
    prompt = prompt + f"For instance, '{from_text}' should be categorized as 'no'. \nText: "

    for j in range(len(data)):
        text = 'This customs import declaration has attributes: '
        for a in range(len(data[0]) - 1):
            sp = ', ' if a != len(data[0]) - 2 else '.'
            text = text + f'{attributes[a]}: {str(data[j][a])}' + sp
        if data[j][-1] == 0:
            answer = 'no'
            gold = 0
        else:
            answer = 'yes'
            gold = 1
        data_tmp.append({'id': j, "query": f"{prompt}'{text}' \nAnswer:", 'answer': answer,
                         "choices": ["no", "yes"],
                         "gold": gold, 'text': text})
    return data_tmp


def json_save(data, dataname, attributes, out_jsonl=False):
    data_tmp = process_table(data, attributes)
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

def json_save_gpt4(data, dataname, attributes):
    data_tmp = process_table(data, attributes)
    with open('gpt4-data/{}.jsonl'.format(dataname), 'w') as f:
        for i in data_tmp:
            json.dump(i, f)
            f.write('\n')
        print('-----------')
        print(f"{dataname}.jsonl write done")
    f.close()


#####process
def get_data(mode):
    data = []
    for i, m in enumerate(mode):
        file_name = f"df_syn_{m}_eng.csv"
        if m == 'test':
            da = pd.read_csv(file_name, sep=',', header=0).drop('Critical Fraud', axis=1)
            column_name = da.columns.values.tolist()
            da = da.values.tolist()
            random.seed(10086)
            da = random.sample(da, 2000)
        else:
            da = pd.read_csv(file_name, sep=',', header=0).drop('Critical Fraud', axis=1).values.tolist()
        data.append(da)
    return data, column_name


# mean_list
feature_size = 20 + 1
name = ['train', 'valid', 'test']
data, column_name = get_data(name)
for i in range(len(data)):
    _ = json_save(data[i], name[i], column_name)

test_data = data[2]
s1_data = [row for row in test_data if row[-1] != 2]
s2_data = [row for row in test_data if row[-1] == 2]
labels = [row[-1] for row in s1_data]
_, s3_data = train_test_split(s1_data, test_size=200 - len(s2_data), stratify=labels, random_state=100)
gpt4_data = pd.DataFrame(s2_data + s3_data)
random_index = np.random.permutation(gpt4_data.index)
gpt4_data = gpt4_data.reindex(random_index)
# gpt4_data.to_csv('gpt4_test_data.csv', index=False)
json_save_gpt4(gpt4_data.values.tolist(), 'test_gpt4', column_name)
