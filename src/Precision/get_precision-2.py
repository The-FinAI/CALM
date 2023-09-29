import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

file_name = 'flare_customs2_desc_write_out_info.json'
model_name = ['gpt4', 'bloomz', 'vicuna', 'llama', 'llama2', 'llama2-chat', 'chatglm', 'llama', 'chatgpt', 'our']
i = -2

file_path = os.path.join(model_name[i], file_name)
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

logit = []
answers = []
l1, l2 = 0, 0
replacement_dict = {'no': [1], 'yes': [0]}
for item in data:
    # 只统计未缺失标签
    if 'no' in item['logit_0'].lower():
        logit.append('no')
        answers.append(item['truth'])
        l1 += 1
    elif 'yes' in item['logit_0'].lower():
        logit.append('yes')
        answers.append(item['truth'])
        l2 += 1
print(f"模型 {model_name[i]} 回答数据集 {file_name[6:14]} 样本数：l1-{l1}, l2-{l2}")
new1 = [replacement_dict[item] if item in replacement_dict else item for item in answers]
new2 = [replacement_dict[item] if item in replacement_dict else item for item in logit]
y_true = np.array(new1)
y_pred = np.array(new2)


print('binary precision', precision_score(y_true, y_pred, average='binary', zero_division=0))
print('binary f1-score', f1_score(y_true, y_pred, average='binary'))
print('Acc_score', accuracy_score(y_true, y_pred))
