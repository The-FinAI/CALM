import random

import numpy as np
import pandas as pd
import json

#####config
from sklearn.model_selection import train_test_split

name = "taiwan.csv"
feature_size = 95 + 1  # Target_index = 0
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

mean_list = ['ROA(C) before interest and depreciation before interest',
             'ROA(A) before interest and percentage after tax',
             'ROA(B) before interest and depreciation after tax', 'Operating Gross Margin',
             'Realized Sales Gross Margin', 'Operating Profit Rate',
             'Pre-tax net Interest Rate', 'After-tax net Interest Rate',
             'Non-industry income and expenditure', 'Continuous interest rate (after tax)',
             'Operating Expense Rate', 'Research and development expense rate',
             'Cash flow rate', 'Interest-bearing debt interest rate',
             'Tax rate (A)', 'Net Value Per Share (B)',
             'Net Value Per Share (A)', 'Net Value Per Share (C)',
             'Persistent EPS in the Last Four Seasons', 'Cash Flow Per Share',
             'Revenue Per Share (Yuan ¥)', 'Operating Profit Per Share (Yuan ¥)',
             'Per Share Net profit before tax (Yuan ¥)', 'Realized Sales Gross Profit Growth Rate',
             'Operating Profit Growth Rate', 'After-tax Net Profit Growth Rate',
             'Regular Net Profit Growth Rate', 'Continuous Net Profit Growth Rate',
             'Continuous Net Profit Growth Rate', 'Net Value Growth Rate',
             'Total Asset Return Growth Rate Ratio', 'Cash Reinvestment %',
             'Current Ratio', 'Quick Ratio',
             'Interest Expense Ratio', 'Total debt to Total net worth',
             'Debt ratio', 'Net worth to Assets',
             'Long-term fund suitability ratio (A)', 'Borrowing dependency',
             'Contingent liabilities to Net worth', 'Operating profit to Paid-in capital',
             'Net profit before tax to Paid-in capital', 'Inventory and accounts receivable to Net value',
             'Total Asset Turnover', 'Accounts Receivable Turnover',
             'Average Collection Days', 'Inventory Turnover Rate (times)',
             'Fixed Assets Turnover Frequency', 'Net Worth Turnover Rate (times)',
             'Revenue per person', 'Operating profit per person',
             'Operating profit per person', 'Operating profit per person',
             'Quick Assets to Total Assets', 'Quick Assets to Total Assets',
             'Quick Assets to Total Assets', 'Quick Assets to Current Liability',
             'Cash to Current Liability', 'Current Liability to Assets',
             'Operating Funds to Liability', 'Inventory to Working Capital',
             'Inventory to Current Liability', 'Current Liabilities to Liability',
             'Working Capital to Equity', 'Current Liabilities to Equity',
             'Long-term Liability to Current Assets', 'Retained Earnings to Total Assets',
             'Total income to Total expense', 'Total expense to Assets',
             'Current Asset Turnover Rate', 'Quick Asset Turnover Rate',
             'Working capital Turnover Rate', 'Cash Turnover Rate',
             'Cash Flow to Sales', 'Cash Flow to Sales',
             'Current Liability to Liability', 'Current Liability to Equity',
             'Equity to Long-term Liability', 'Cash Flow to Total Assets',
             'Cash Flow to Liability', 'CFO to Assets',
             'Cash Flow to Equity', 'Current Liability to Current Assets',
             'Liability-Assets Flag', 'Net Income to Total Assets',
             'Total assets to GNP price', 'Total assets to GNP price',
             'Gross Profit to Sales', 'Net Income to Stockholder\'s Equity',
             'Liability to Equity', 'Liability to Equity',
             'Interest Coverage Ratio', 'Net Income Flag',
             'Equity to Liability']


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
    # for i in range(1, len(data[0])):
    #     if i != len(data[0]):
    #         prompt = prompt + f'{mean_list[i-1]}, '
    #     if i == len(data[0])-1:
    #         prompt = prompt + f'{mean_list[i - 1]}. \n'
    from_text = "The client has attributes:  ROA(C) before interest and depreciation before interest: 0.499, " \
                "..., Net Income Flag: 1.000,  Equity to Liability: 0.044."
    prompt = prompt + f"For instance, '{from_text}' should be classified as 'no'. \nText: "
    for j in range(len(data)):
        text = 'The client has attributes: '
        for i in range(1, len(data[0])):
            sp = ', ' if i != len(data[0]) - 1 else '.'
            text = text + f'{mean_list[i - 1]}: {data[j][i]:.3f}' + sp
        answer = 'no' if data[j][0] == 0 else 'yes'
        # '0' is good (no bankruptcy) (6599) and '1' is bad
        data_tmp.append(
            {'id': j, "query": f"{prompt}'{text}'" + ' \nAnswer:', 'answer': answer, "choices": ["no", "yes"],
             "gold": int(data[j][0]), 'text': text})
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


def save_gpt4_data(test_data):
    tmp_data = [row[0] for row in test_data]
    _, gpt4_data = train_test_split(test_data, test_size=500, stratify=tmp_data, random_state=100)

    s1_data = [row for row in gpt4_data if row[0] == 0]
    s2_data = [row for row in gpt4_data if row[0] == 1]
    s_data = s2_data + s1_data[:100 - len(s2_data)]
    s_data = pd.DataFrame(s_data)
    np.random.seed(42)
    random_index = np.random.permutation(s_data.index)
    ss_data = s_data.reindex(random_index)
    # ss_data.to_csv('gpt4_rawdata.csv', index=False)
    json_save_gpt4(ss_data.values.tolist(), 'test_gpt4')


def get_num(data):
    data_con = np.array(data)
    check = np.unique(data_con[:, 0])
    check1 = (data_con[:, 0] == check[0]).sum()
    check2 = (data_con[:, 0] == check[1]).sum()
    return check1, check2


#####process
# column 1 is the target variable (float) on data
data = pd.read_csv(name, sep=',', header=0)
column_name = data.columns.values.tolist()
data = data.values.tolist()

che = get_num(data)
data = data_split(data)

save_gpt4_data(data[2])

save_name = ['train', 'valid', 'test']
for i in range(len(data)):
    _ = json_save(data[i], save_name[i], mean_list=column_name)
