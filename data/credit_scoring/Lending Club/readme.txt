This document is about credit scoring task using a dataset named "Lending Club".
Description_based: clear semantic information about the features

Original data description：
2007 through current Lending Club accepted and rejected loan data.
\url{https://www.kaggle.com/datasets/wordsforthewise/lending-club}

Basic data information：
It is a tabular data. 
After being processed in Lending Club.docx, The data have 21 features, including Loan Information: Installment, Loan Purpose, Loan Application Type, Interest Rate, Last Payment Amount, Loan Amount, Revolving Balance; History Information: Delinquency In 2 years, Inquiries In 6 Months, Mortgage Accounts, Grade, Open Accounts, Revolving Utilization Rate, Total Accounts, Fico Range Low, Fico Range High; Soft Information: Address State, Employment Length, Home Ownership, Verification Status, Annual Income.
The number of the data is 53,812/1,345,310.  We split 0.7 for train, 0.1  for dev, and 0.2  for test.

In this file, 
1. Lending Club.docx: It's a data description file.
2. accepted_2007_to_2018Q4.csv: It's the original data file.
3. preprocess.py: It's our preprocessing code.


When using, please add the "prompt" at the beginning: 
"Assess the client's loan status based on the following loan records from Lending Club. Respond with only 'good' or 'bad', and do not provide any additional information. For instance, 'The client has a stable income, no previous debts, and owns a property.' should be classified as 'good'. 
Text: "

