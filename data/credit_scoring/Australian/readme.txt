This document is about the credit scoring task using a dataset named "Australian".
Table-based: meaningless symbols

Original data description: 
This file concerns credit card applications. This database exists elsewhere in the repository (Credit Screening Database) in a slightly different form.
\url{http://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval}

Basic data information: 
It is a tabular data. 
The data have 14 features, including 6 numerical and 8 categorical attributes. And all attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.
The number of the data is 690.  We split 482 for train, 69 for dev, and 139 for test.

In this file, 
1. australian.doc: It's a data description file.
2. data description: It's a concise data description file.
3. prepocess.py: It's our preprocessing code.

When using, please add the "prompt" at the beginning:
"Assess the creditworthiness of a customer using the following table attributes for financial status. Respond with either 'good' or 'bad'. And all the table attribute names including 8 categorical attributes and 6 numerical attributes and values have been changed to meaningless symbols to protect confidentiality of the data. For instance, 'The client has attributes: A1: 0, A2: 21.67, A3: 11.5, A4: 1, A5: 5, A6: 3, A7: 0, A8: 1, A9: 1, A10: 11, A11: 1, A12: 2, A13: 0, A14: 1.', should be classified as 'good'. 
Text: "