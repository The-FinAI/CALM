import random
import pandas as pd
import json
import numpy as np
import os

#####config
name = "german.data"
feature_size = 20+1
train_size, dev_size, test_size = 0.7, 0.1, 0.2

if train_size + dev_size + test_size != 1:
    print("sample size wrong!!!")


mean_list = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
             'Credit amount', 'Savings account/bonds', 'Present employment since',
             'Installment rate in percentage of disposable income', 'Personal status and sex',
             ' Other debtors / guarantors', 'Present residence since', 'Property', 'Age in years',
             'Other installment plans', 'Housing', 'Number of existing credits at this bank' ,'Job'
             'Number of people being liable to provide maintenance for' , 'Telephone' , 'foreign worker'
             ]

dict = {'0':{'A11':'smaller than 0 DM', 'A12':'bigger than 0 DM but smaller than 200 DM',
             'A13':'bigger than 200 DM OR salary assignments for at least 1 year',
             'A14': 'no checking account'},
        '2':{'A30' : 'no credits taken/all credits paid back duly',
            'A31' : 'all credits at this bank paid back duly',
	        'A32' : 'existing credits paid back duly till now',
            'A33' : 'delay in paying off in the past',
	        'A34' : 'critical account/other credits existing (not at this bank)'},
        '3':{'A40' : 'car (new)',
	        'A41' : 'car (used)',
	        'A42' : 'furniture/equipment',
	        'A43' : 'radio/television',
	        'A44' : 'domestic appliances',
	        'A45' : 'repairs',
	        'A46' : 'education',
	        'A47' : '(vacation - does not exist?)',
	        'A48' : 'retraining',
	        'A49' : 'business',
	        'A410' : 'others'},
        '5':{'A61' :  'smaller than 100 DM',
	        'A62' :   'bigger than 100 smaller than  500 DM',
	        'A63' :   'bigger than 500 smaller than 1000 DM',
	        'A64' :   'bigger than 1000 DM',
            'A65' :   'unknown/ no savings account',},
        '6':{'A71' : 'unemployed',
	        'A72' : 'smaller than 1 year',
	        'A73' : 'bigger than 1  smaller than 4 years',
	        'A74' : 'bigger than 4  smaller than 7 years',
	        'A75' : 'bigger than 7 years'},
        '8':{'A91' : 'male   : divorced/separated',
	        'A92' : 'female : divorced/separated/married',
            'A93' : 'male   : single',
	        'A94' : 'male   : married/widowed',
	        'A95' : 'female : single'},
        '9':{'A101' : 'none',
	        'A102' : 'co-applicant',
	        'A103' : 'guarantor'},
        '11':{'A121' : 'real estate',
	        'A122' : 'building society savings agreement/life insurance',
            'A123' : 'car or other',
	        'A124' : 'unknown / no property'},
        '13':{'A141' : 'bank',
	        'A142' : 'stores',
	        'A143' : 'none'},
        '14':{'A151' : 'rent',
	        'A152' : 'own',
	        'A153': 'for free'},
        '16':{'A171' : 'unemployed/ unskilled  - non-resident',
	        'A172' : 'unskilled - resident',
	        'A173' : 'skilled employee / official',
	        'A174' : 'management/ self-employed/highly qualified employee/ officer'},
        '18':{'A191' : 'none',
	        'A192': 'yes, registered under the customers name'},
        '19':{'A201' : 'yes',
	       'A202' : 'no'},
        }

#####function
def process(data, mean_list, dict):
    data_tmp = []
    for j in range(len(data)):
        text = ''
        for i in range(len(data[0]) - 2):
            if str(i) not in list(dict.keys()):
                text = text+ '. '+ 'The state of ' + mean_list[i] + ' is ' + str(data[j][i])
            else:
                text = text+ '. '+ 'The state of ' + mean_list[i] + ' is ' + dict[str(i)][data[j][i]]
            label = 'good' if data[j][-1] == 1 else 'bad'
        data_tmp.append({'id':j ,'text':text, 'label':label})
    return data_tmp

def json_save(data, dataname, mean_list= mean_list, dict= dict):
    data_tmp = process(data, mean_list, dict)
    with open('{}.jsonl'.format(dataname),'w') as f:
        for i in data_tmp:
            json.dump(i, f)
            f.write('\n')
        print('-----------')
        print("write done")
    f.close()
    return None

#####process
data = pd.read_csv(name, sep=' ', names=[i for i in range(feature_size)]).values.tolist()

random.seed(10086)

train_ind = random.sample([i for i in range(len(data))],int(len(data)*train_size))
train_data = [data[i] for i in train_ind]

index_left = list(filter(lambda x: x not in train_ind, [i for i in range(len(data))]))
dev__ind = random.sample(index_left,int(len(data)*dev_size))
dev_data = [data[i] for i in dev__ind]

index_left = list(filter(lambda x: x not in train_ind + dev__ind, [i for i in range(len(data))]))
test_data = [data[i] for i in index_left]

json_save(test_data , 'test')
json_save(train_data , 'train')
json_save(dev_data , 'dev')