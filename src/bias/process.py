from sklearn.preprocessing import LabelEncoder
import json

def predo(data):
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    pre_data = data.copy()
    label_encoder = LabelEncoder()
    for col in object_cols:
        pre_data[col] = label_encoder.fit_transform(data[col])

    # age 和gender需要进一步划分成二分类
    # 对于 german age划分以45岁为标准
    #todo 其他数据
    pre_data[12][pre_data[12] <= 45] = 0
    pre_data[12][pre_data[12] > 45]= 1

    pre_data[8][pre_data[8] == 2] = 0  # male
    pre_data[8][pre_data[8] == 3] = 0  # male
    pre_data[8][pre_data[8] == 5] = 1  # female

    return pre_data.values.tolist()

def predo_tra(data):
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    pre_data = data.copy()
    label_encoder = LabelEncoder()
    for col in object_cols:
        pre_data[col] = label_encoder.fit_transform(data[col])

    # age 和gender需要进一步划分成二分类
    # 对于 german age划分以45岁为标准
    #todo 其他数据
    pre_data[9][pre_data[9] <= 45] = 0
    pre_data[9][pre_data[9] > 45]= 1

    return pre_data.values.tolist()


def preres(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            if text['truth']=='good' and text['acc']=='1.0':
                res_data[i][-1] = 1
            elif text['truth']=='bad' and text['acc']=='0.0':
                res_data[i][-1] = 1
            else: res_data[i][-1] = 2
    return res_data

def preres_tra(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            if text['truth']=='no' and text['acc']=='1.0':
                res_data[i][0] = 0
            elif text['truth']=='yes' and text['acc']=='0.0':
                res_data[i][0] = 0
            else: res_data[i][0] = 1
    return res_data

def preres_cc(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        index = []
        for i, text in enumerate(file_json):
            if text['missing']=='1':
                index.append(i)
            elif text['truth']=='good' and text['acc']=='1.0':
                res_data[i][-1] = 0
            elif text['truth']=='bad' and text['acc']=='0.0':
                res_data[i][-1] = 0
            else: res_data[i][-1] = 1
    return res_data,index
