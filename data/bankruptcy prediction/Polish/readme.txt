This document is about the bankruptcy prediction task using a dataset named "Polish."
Table-based:  too many features

Original data description:
The dataset is about bankruptcy prediction of Polish companies.The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013.
\url{http://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data}

Basic data information: 
It is a tabular data.
The dataset is about bankruptcy prediction of Polish companies. The data was collected from Emerging Markets Information Service (EMIS, [Web Link]), which is a database containing information on emerging markets around the world. The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013. 
Basing on the collected data five classification cases were distinguished, that depends on the forecasting period: 
- 1stYear â€“ the data contains financial rates from 1st year of the forecasting period and corresponding class label that indicates bankruptcy status after 5 years. The data contains 7027 instances (financial statements), 271 represents bankrupted companies, 6756 firms that did not bankrupt in the forecasting period. 
- 2ndYear â€“ the data contains financial rates from 2nd year of the forecasting period and corresponding class label that indicates bankruptcy status after 4 years. The data contains 10173 instances (financial statements), 400 represents bankrupted companies, 9773 firms that did not bankrupt in the forecasting period. 
- 3rdYear â€“ the data contains financial rates from 3rd year of the forecasting period and corresponding class label that indicates bankruptcy status after 3 years. The data contains 10503 instances (financial statements), 495 represents bankrupted companies, 10008 firms that did not bankrupt in the forecasting period. 
- 4thYear â€“ the data contains financial rates from 4th year of the forecasting period and corresponding class label that indicates bankruptcy status after 2 years. The data contains 9792 instances (financial statements), 515 represents bankrupted companies, 9277 firms that did not bankrupt in the forecasting period. 
- 5thYear â€“ the data contains financial rates from 5th year of the forecasting period and corresponding class label that indicates bankruptcy status after 1 year. The data contains 5910 instances (financial statements), 410 represents bankrupted companies, 5500 firms that did not bankrupt in the forecasting period.

The data have 64 features, including X1-X64 as following prompt.
The number of the data is 43,405. We split 0.7  for train, 0.1  for dev, and 0.2  for test.

In this file, 
1. Polish.docx: It's a data description file.
2. *.arff: It's the original data file.
3. preprocess.py: It's our preprocessing code.


When using, please add the "prompt" at the beginning:
"Predict whether the company will face bankruptcy based on the financial profile attributes provided in the following text. Respond with only 'no' or 'yes', and do not provide any additional information. 
For instance, 'The client has attributes: net profit / total assets: -0.046186, ..., sales / short-term liabilities: 5.7063, sales / fixed assets: 1.3882.' should be classified as 'no'. 
Text: "



