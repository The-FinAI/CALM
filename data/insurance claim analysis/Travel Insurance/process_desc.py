import random
import pandas as pd
import json

#####config
name = "travel insurance.csv"
feature_size = 10 + 1  # Target_index = 4
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
    prompt = 'Identify the claim status of insurance companies using the records of travel insurance attributes. ' \
             'Respond with either \'Yes\' or \'No\' (Do not return \'Good\' or \'Bad\'). '
    from_text = "A policyholder aged 41 chosen product 'Rental Vehicle Excess Insurance' of the insurance company " \
                "'CWT' through sales channel 'Online to travel to destination 'ITALY'. The type of insurance is " \
                "'Travel Agency', with an effective period of 79, and the company recorded the net sales " \
                "and commission of the insurance as -19.8 and 11.88."
    prompt = prompt + f"For instance: '{from_text}', should be classified as \'No\'. \nText: "

    for j in range(len(data)):
        text = f"A policyholder aged {str(data[j][9])} chosen product '{str(data[j][3])}' of the insurance " \
               f"company '{str(data[j][0])}' through sales channel '{str(data[j][2])} to travel " \
               f"to destination '{str(data[j][6])}'. The type of insurance is '{str(data[j][1])}', " \
               f"with an effective period of {str(data[j][5])}, and the company recorded the net sales " \
               f"and commission of the insurance as {str(data[j][7])} and {str(data[j][8])}."
        answer = data[j][4]
        gold = 0 if data[j][4] == 'Yes' else 1
        # 'No' 62399 and Yes' 927
        data_tmp.append({'id': j, "query": prompt + text + ' \nAnswer:', 'answer': answer, "choices": ["Yes", "No"],
                         "gold": gold, 'text': text})
    return data_tmp


def json_save(data, dataname, mean_list=mean_list):
    data_tmp = process_table(data, mean_list)
    with open('{}.jsonl'.format(dataname), 'w') as f:
        for i in data_tmp:
            json.dump(i, f)
            f.write('\n')
        print('-----------')
        print("write done")
    f.close()
    return data_tmp


#####process
data = pd.read_csv(name, sep=',', header=0, names=[i for i in range(feature_size)]).values.tolist()
# data preprocessing
data = data_preparation(data)

random.seed(10086)

train_ind = random.sample([i for i in range(len(data))], int(len(data) * train_size))
train_data = [data[i] for i in train_ind]

index_left = list(filter(lambda x: x not in train_ind, [i for i in range(len(data))]))
dev__ind = random.sample(index_left, int(len(data) * dev_size))
dev_data = [data[i] for i in dev__ind]

index_left = list(filter(lambda x: x not in train_ind + dev__ind, [i for i in range(len(data))]))
test_data = [data[i] for i in index_left]

test_prompt_data = json_save(test_data, 'test_desc')
train_prompt_data = json_save(train_data, 'train_desc')
dev_prompt_data = json_save(dev_data, 'valid_desc')