This document ia about insurance claim analysis task using a dataset named "Travel Insurance".

Orinal data description: 
Travel Insurance with the target attribute of: Claim Status (Yes or No). A third-party travel insurance servicing company that is based in Singapore.
\url{https://www.kaggle.com/datasets/mhdzahier/travel-insurance}

Basic data information: 
It is a tabular data. 
The data have 9 features, including Claim Status (target variable), Agency, Agency Type, Distribution Channel, Product Name, Duration, Destination, Net Sales, Commission, Gender (discarded) and Age.\
The number of the data is 63,326.  We split 0.7 (44,328) for train, 0.1 (6332) for dev, and 0.2 (12,666) for test.

In this file, 
1. travel insurance.docx: It's a data description and data-preprocessing description file.
2. travel insurance.csv: It's the original data file.
4. preprocess*: It's our preprocessing code. 

When using, please add the "prompt" at the beginning: 
"Identify the claim status of insurance companies using the following table attributes for travel insurance status. Respond with only 'yes' or 'no', and do not provide any additional information. And the table attributes including 5 categorical attributes and 4 numerical attributes are as follows: 
Agency: Name of agency (categorical). 
Agency Type: Type of travel insurance agencies (categorical). 
Distribution Channel: Distribution channel of travel insurance agencies (categorical). 
Product Name: Name of the travel insurance products (categorical). 
Duration: Duration of travel (categorical). 
Destination: Destination of travel (numerical). 
Net Sales: Amount of sales of travel insurance policies (categorical). 
Commission: Commission received for travel insurance agency (numerical). 
Age: Age of insured (numerical). 
For instance: 'The insurance company has attributes: Agency: CBH, Agency Type: Travel Agency, Distribution Chanel: Offline, Product Name: Comprehensive Plan, Duration: 186, Destination: MALAYSIA, Net Sales: -29, Commision: 9.57, Age: 81.', should be classified as 'no'. 
Text:"

Description-based:  the "prompt" shoule be:
"Identify the claim status of insurance companies using the records of travel insurance attributes. Respond with either 'Yes' or 'No' (Do not return 'Good' or 'Bad'). For instance: 'A policyholder aged 41 chosen product 'Rental Vehicle Excess Insurance' of the insurance company 'CWT' through sales channel 'Online to travel to destination 'ITALY'. The type of insurance is 'Travel Agency', with an effective period of 79, and the company recorded the net sales and commission of the insurance as -19.8 and 11.88.', should be classified as 'No'. 
Text:"
