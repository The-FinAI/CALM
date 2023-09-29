import random
import pandas as pd
import json

#####config
name = "australian.dat"
feature_size = 14 + 1   # Target_index = -1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")

#####function
def process_table(data):
    data_tmp = []
    prompt = 'Assess the creditworthiness of a customer using the following table attributes for financial status. ' \
             'Respond with either \'good\' or \'bad\'. And all the table attribute names including 8 categorical ' \
             'attributes and 6 numerical attributes and values have been changed to meaningless symbols to ' \
             'protect confidentiality of the data. For instance, \'The client has attributes: A1: 0, A2: 21.67, ' \
             'A3: 11.5, A4: 1, A5: 5, A6: 3, A7: 0, A8: 1, A9: 1, A10: 11, A11: 1, A12: 2, A13: 0, A14: 1.\', ' \
             'should be classified as \'good\'. \nText: '

    for j in range(len(data)):
        text = 'The client has attributes:'
        for i in range(len(data[0]) - 1):
            sy = '.' if i == len(data[0]) - 2 else ','
            text = text + f' A{i + 1}: ' + str(data[j][i]) + sy
        answer = 'good' if data[j][-1] == 1 else 'bad'
        gold = 0 if data[j][-1] == 1 else 1
        # '1' is good (307) and '0' is bad (383)
        data_tmp.append({'id': j, "query": prompt + text + ' \n Answer:', 'answer': answer, "choices": ["good", "bad"],
                         "gold": gold, 'text': text})
    return data_tmp


def json_save(data, dataname, out_jsonl=False):
    data_tmp = process_table(data)
    if out_jsonl:
        with open('{}.jsonl'.format(dataname), 'w') as f:
            for i in data_tmp:
                json.dump(i, f)
                f.write('\n')
            print('-----------')
            print(f"{dataname} write done")
        f.close()
    df = pd.DataFrame(data_tmp)
    # 保存为 Parquet 文件
    parquet_file_path = f'data/{dataname}.parquet'
    df.to_parquet(parquet_file_path, index=False)
    return data_tmp


#####process
data = pd.read_csv(name, sep=' ', names=[i for i in range(feature_size)]).values.tolist()

random.seed(10086)

train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
train_data = [data[i] for i in train_ind]

index_left = list(filter(lambda x: x not in train_ind, [i for i in range(len(data))]))
dev__ind = random.sample(index_left, int(len(data) * dev_size))
dev_data = [data[i] for i in dev__ind]

index_left = list(filter(lambda x: x not in train_ind + dev__ind, [i for i in range(len(data))]))
test_data = [data[i] for i in index_left]

columns = [i for i in range(feature_size)]
ss_data = pd.DataFrame(test_data, columns=columns)
ss_data.to_csv('australian_test.csv', index=False)

test_prompt_data = json_save(test_data, 'test')
train_prompt_data = json_save(train_data, 'train')
dev_prompt_data = json_save(dev_data, 'valid')
