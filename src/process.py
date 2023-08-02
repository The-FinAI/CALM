from sklearn.preprocessing import LabelEncoder
import json

def predo(data):
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    pre_data = data.copy()
    label_encoder = LabelEncoder()
    for col in object_cols:
        pre_data[col] = label_encoder.fit_transform(data[col])
    return pre_data.values.tolist()

def preres(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            print()
            if text['truth']=='good' and text['acc']=='1.0':
                res_data[i][-1] = 1
            elif text['truth']=='bad' and text['acc']=='0.0':
                res_data[i][-1] = 1
            else: res_data[i][-1] = 2
    return res_data