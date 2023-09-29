This document is about fraud detection task using a dataset named "Credit Card Fraud".
Table-based: meaningless symbols

Original data description:
Anonymized credit card transactions labeled as fraudulent or genuine. 
\url{https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}

Basic data description：
It is a tabular data.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
Thus, The data have 29 features, including V1, V2, ..., V28 and Amount.
The number of the data is 28,480/284,807 (10%).  We split 0.7  for train, 0.1  for dev, and 0.2  for test.

In this file：
1. creditcard.docx: It's a data description file.
2. creditcard.csv: It's the original data file. (Source: kaggle.com)
3. preprocess.py: It's our preprocessing code.

When using, please add the "prompt" at the beginning: 
"Detect the credit card fraud using the following financial table attributes. Respond with only 'yes' or 'no', and do not provide any additional information. Therein, the data contains 28 numerical input variables V1, V2, ..., and V28 which are the result of a PCA transformation and 1 input variable Amount which has not been transformed with PCA. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. For instance, 'The client has attributes: V1: 0.144, V2: 0.358, V3: 1.220, V4: 0.331, V5: -0.273, V6: 0.429, V7: -0.307, V8: -0.577, V9: 0.116, V10: -0.337, V11: 1.016, V12: 1.043, V13: -0.527, V14: 0.160, V15: -0.951, V16: -0.452, V17: 0.166, V18: -0.446, V19: 0.036, V20: -0.275, V21: 0.768, V22: -0.051, V23: -0.180, V24: 0.067, V25: 0.741, V26: 0.477, V27: 0.152, V28: 0.201, Amount: 6.990.' should be classified as 'no'. 
Text: "