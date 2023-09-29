This document is about the bankruptcy prediction task using a dataset named "Taiwan"
Table-based: too many features

Original data description:
Bankruptcy data from the Taiwan Economic Journal for the years 1999–2009. The data were collected from the Taiwan Economic Journal for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange.
\url{https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction}

Basic data information:
It is a tabular data.
The data have 95 features, including X1-X95 as following prompt.
The number of the data is 6819. We split 0.7  for train, 0.1  for dev, and 0.2  for test.

In this file,
1. taiwan.docx: It's a data description file.
2. taiwan.csv: It's the original data file.
3. preprocess.py: It's our preprocessing code.

When using, please add the "prompt" at the beginning: 
"Predict whether the company will face bankruptcy based on the financial profile attributes provided in the following text. Respond with only 'no' or 'yes', and do not provide any additional information. 
For instance, 'The client has attributes:  ROA(C) before interest and depreciation before interest: 0.499, ..., Net Income Flag: 1.000,  Equity to Liability: 0.044.' should be classified as 'no'. 
Text: "