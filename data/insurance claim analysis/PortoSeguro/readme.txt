This document ia about insurance claim analysis task using a dataset named "PortoSeguro".
Table-based: too many features

Original data description:
In this competition, you will predict the probability that an auto insurance policy holder files a claim.
\url{https://www.kaggle.com/datasets/alinecristini/atividade2portoseguro}
\url{https://www.mdpi.com/2227-9091/9/2/42}

Basic data information: 
It is a tabular data.
The data have 57 features, including ps_ind_01, ps_ind_02_cat, ps_ind_03, ps_ind_04_cat, ps_ind_05_cat, ps_ind_06_bin, ps_ind_07_bin, ps_ind_08_bin, ps_ind_09_bin, ps_ind_10_bin, ps_ind_11_bin, ps_ind_12_bin, ps_ind_13_bin, ps_ind_14, ps_ind_15, ps_ind_16_bin, ps_ind_17_bin, ps_ind_18_bin, ps_reg_01, ps_reg_02, ps_reg_03, ps_car_01_cat, ps_car_02_cat, ps_car_03_cat, ps_car_04_cat, ps_car_05_cat, ps_car_06_cat, ps_car_07_cat, ps_car_08_cat, ps_car_09_cat, ps_car_10_cat, ps_car_11_cat, ps_car_11, ps_car_12, ps_car_13, ps_car_14, ps_car_15, ps_calc_01, ps_calc_02, ps_calc_03, ps_calc_04, ps_calc_05, ps_calc_06, ps_calc_07, ps_calc_08, ps_calc_09, ps_calc_10, ps_calc_11, ps_calc_12, ps_calc_13, ps_calc_14, ps_calc_15_bin, ps_calc_16_bin, ps_calc_17_bin, ps_calc_18_bin, ps_calc_19_bin, and ps_calc_20_bin.
The number of the data is 59,521/595,212 (10%). We split 0.7  for train, 0.1  for dev, and 0.2  for test.

In this file, 
1. PortoSeguro.docx: It's a data description file.
2. PortoSeguro.csv: It's the original data file. (Source：kaggle.com,  train.csv)
3. preprocess.py: It's our preprocessing code.


When using, please add the "prompt" at the beginnig: 
"Identify whether or not to files a claim for the auto insurance policy holder using the following table attributes about individual financial profile. Respond with only 'yes' or 'no', and do not provide any additional information. And the table attributes that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. For instance, 'The client has attributes: ps_ind_01: 1.0, ps_ind_02_cat: 2.0, ps_ind_03: 0.0, ..., ps_calc_18_bin: 0.0, ps_calc_19_bin: 0.0, ps_calc_20_bin: 0.0.' should be classified as 'no'. 
Text:"