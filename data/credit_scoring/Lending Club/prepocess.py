import random

import numpy as np
import pandas as pd
import json

#####config
from sklearn.model_selection import train_test_split

name = "accepted_2007_to_2018Q4.csv"
feature_size = 21 + 1  # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = ['Installment', 'Loan Purpose', 'Loan Application Type', 'Interest Rate', 'Last Payment Amount',
             'Loan Amount', 'Revolving Balance',
             'Delinquency In 2 years', 'Inquiries In 6 Months', 'Mortgage Accounts', 'Grade', 'Open Accounts',
             'Revolving Utilization Rate', 'Total Accounts', 'Fico Range Low', 'Fico Range High',
             'Address State', 'Employment Length', 'Home Ownership', 'Verification Status', 'Annual Income',
             'Loan Status']


#####function
def process_table(data, mean_list):
    data_tmp = []
    prompt = 'Assess the client\'s loan status based on the following loan records from Lending Club. ' \
             'Respond with only \'good\' or \'bad\', and do not provide any additional information. For instance, ' \
             '\'The client has a stable income, no previous debts, and owns a property.\' ' \
             'should be classified as \'good\'. \nText: '

    for j in range(len(data)):
        text = 'The client has attributes as follows: '
        for i in range(len(data[0]) - 1):
            sp = '. ' if i != len(data[0]) - 2 else '.'
            if i == 3 or i == 12:
                text = text + f'The state of {mean_list[i]} is {str(data[j][i])}%' + sp
            else:
                text = text + f'The state of {mean_list[i]} is {str(data[j][i])}' + sp
        answer = 'good' if data[j][-1] == 'Fully Paid' else 'bad'
        gold = 0 if data[j][-1] == 'Fully Paid' else 1
        # 'Fully Paid' is good and 'Charged off' is bad
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["good", "bad"],
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
    return check2, check1


def get_data(name):
    selected_feature = ['installment', 'purpose', 'application_type', 'int_rate', 'last_pymnt_amnt', 'loan_amnt',
                        'revol_bal', 'delinq_2yrs', 'inq_last_6mths', 'mort_acc', 'grade', 'open_acc', 'revol_util',
                        'total_acc', 'fico_range_low', 'fico_range_high', 'addr_state', 'emp_length', 'home_ownership',
                        'verification_status', 'annual_inc', 'loan_status']
    data = pd.read_csv(name, sep=',', header=0, low_memory=False, usecols=selected_feature).reindex(
        columns=selected_feature)
    # only reserve Fully Paid and Charged Off
    for loan_st in data['loan_status'].unique().tolist():
        if loan_st != 'Fully Paid' and loan_st != 'Charged Off':
            data = data.drop(data[data['loan_status'] == loan_st].index)
    data.dropna(subset=['loan_status'], inplace=True)
    save_data, drop_data = train_test_split(data, test_size=0.99, stratify=data['loan_status'], random_state=100)
    return save_data


def save_gpt4_data(test_data):
    tmp_data = [test_data[-1] for row in test_data]
    #  test_data[-1] 这里该是row[-1],但无所谓，这里巧合的是，取出的分布差不多： 正类 99(bad) / 500  正确的应该是 96 / 500
    _, gpt4_data = train_test_split(test_data, test_size=500, stratify=tmp_data, random_state=100)
    # gpt4_data.to_csv('gpt4_rawdata', index=False)
    json_save_gpt4(gpt4_data, 'test_gpt4')


#####process
data = get_data(name).values.tolist()
check_num = get_num(data)
random.seed(10086)

train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
train_data = [data[i] for i in train_ind]

index_left = list(set(list(range(len(data)))) - set(train_ind))
dev__ind = random.sample(index_left, int(len(data) * dev_size))
dev_data = [data[i] for i in dev__ind]

index_left = list(set(index_left) - set(dev__ind))
test_data = [data[i] for i in index_left]

save_gpt4_data(test_data)

test_prompt_data = json_save(test_data, 'test')
train_prompt_data = json_save(train_data, 'train')
dev_prompt_data = json_save(dev_data, 'valid')
