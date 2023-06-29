本文档是credit scoring任务名为German的数据集。

原数据描述：
It classifies people described by a set of attributes as good or bad credit risks.
\url{http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data}

基本数据情况：
It is a tabular data. 
The data have 20 features, including status of existing checking account, duration, credit history, purpose, credit amount, savings account/bonds, present employment since, installment rate in percentage of disposable income, personal status and sex, other debtors/guarantors, present residence since, property, age, other installment plans, housing, number of existing credits at this bank, job, number of people being liable to provide maintenance for, telephone and foreign worker. 
The number of the data is 1000.  We split 700 for train, 100 for dev, and 200 for test.

在本文件中：
1. german.doc 为原数据描述文件
2. german.data 为原数据文件
3. german.data-numeric 为原数据文件全部预处理为数字后的（本文件弃用）
4. preprocess.py 为我们的预处理程序
5. xxx.jsonl 为生成的数据文件

在使用时：
需在最前面添加prompt:"Analyze the background of customers for credit scoring and determine good or bad credit risks for the customer. Please respond with either good or bad."