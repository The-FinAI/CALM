This document is about the fraud detection task using a dataset named "ccFraud".
Description-based: clear semantic information about the features

Original data description: 
The ccFraud dataset has a high volume which cannot be processed on a single machine. Thus compelling to the use of distributed processing using Apache Spark. This dataset is a snapshot at a particular instant of time for processing, as there is non-availability of credit card dataset having real time inflow of transactions in the public domain.
\url{https://dl.acm.org/doi/abs/10.1145/2980258.2980319}

Basic data description: 
It is a tabular data.
The ccFraud dataset is a highly unbalanced dataset with only 5.96% of fraudulent transactions, rendering the veracity in the data. The “fraudRisk” is a binary feature having 1 and 0 as two discrete values. Here, 1 represents a fraudulent transaction and 0 is used for a non-fraudulent transaction. And We discarded the “custID”, since it contains unique values in all samples, which will disturb in the generalization of patterns.
The data have 7 features, including gender, state, cardholder, balance, numTrans, numIntlTrans and creditLine.

The number of the data is 41,943 / 1,048,575 (4%).  We split 0.7 for train, 0.1 for dev, and 0.2 for test.


In this file: 
1. ccfraud.docx: It's a data description file.
2. ccfraud.csv: It's the orignal data file (Source: \url{https://packages.revolutionanalytics.com/datasets/})
3. preprocess.py: It's our preprocessing code.

When using, please add the "prpmpt" at the beginning: 
"Detect the credit card fraud with the following financial profile. Respond with only 'good' or 'bad', and do not provide any additional information. For instance, 'The client is a female, the state number is 25, the number of cards is 1, the credit balance is 7000, the number of transactions is 16, the number of international transactions is 0, the credit limit is 6.' should be classified as 'good'. 
Text: "