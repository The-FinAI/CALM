This document is about the credit scoring task using a dataset named "German".
Description-based: clear semantic information about the features

Original data description:
It classifies people described by a set of attributes as good or bad credit risks.
\url{http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data}

Basic data information:
It is a tabular data. 
The data have 20 features, including status of existing checking account, duration, credit history, purpose, credit amount, savings account/bonds, present employment since, installment rate in percentage of disposable income, personal status and sex, other debtors/guarantors, present residence since, property, age, other installment plans, housing, number of existing credits at this bank, job, number of people being liable to provide maintenance for, telephone and foreign worker. 
The number of the data is 1000.  We split 700 for train, 100 for dev, and 200 for test.

In this file, 
1. german.doc: It's a data description
2. german.data: It's the original data file.
3. german.data-numeric: It's the original numerical data file (discarded)
4. preprocess.py: It's our preprocessing code.

When using, please add the "prompt" at the beginning: 
"Evaluate the creditworthiness of a customer with the following financial profile. Respond with only either 'good' or 'bad'. For instance, 'The client has a stable income, no previous debts, and owns a property.' should be classified as 'good'. 
Text: "