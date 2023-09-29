Customs Declaration
\url{https://github.com/Seondong/Customs-Declaration-Datasets}

This dataset comprises customs
import declaration records and is intended to detect fraudulent that attempts to reduce customs duty or critical frauds
that can threaten public safety. It consists 54,000 artificially generated records created by CTGAN with 24.7 million
customs declarations reported from January 1, 2020, to June 30, 2021. The dataset encompasses 20 attributes
and includes two labels: ”fraud” and ”critical fraud”.

train/valid/test --> 37385/8134/8481(2000)

prompt:
"Identify the provided customs import declaration information to determine whether it constitutes customs fraud that attempts to reduce customs duty or not. The answer must be 'no' or 'yes', and do not provide any additional information. This Import Declaration consists of 20 data attributes, including Declaration ID, Date, Office ID, Process type, Import type, Import use, Payment type, Mode of transport, Declarant ID, Importer ID, Seller ID, Courier ID, HS6 code, Country of departure, Country of origin, Tax rate, Tax type, Country of origin indicator, Net mass and Item price. For instance, 'This customs import declaration has attributes: Declaration ID: 97061800, Date: 2020-01-01, Office ID: 30, Process Type: B, ..., Item Price: 372254.4.' should be categorized as 'no'.
Text: "
