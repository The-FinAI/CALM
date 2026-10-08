<div align="center">

<h1>CALM</h1>

<p><b>Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models</b></p>

<p>
  <a href="https://arxiv.org/abs/2310.00566"><img src="https://img.shields.io/badge/arXiv-2310.00566-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/TheFinAI/calm-credit-scoring-and-risk-assessment-6ac7153781499dcd5001572e"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow?logo=huggingface" alt="Hugging Face Collection"></a>
  <a href="https://huggingface.co/daishen/CALM-7B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-CALM--7B-yellow?logo=huggingface" alt="CALM-7B model"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p>
  <a href="https://arxiv.org/abs/2310.00566">Paper</a> ·
  <a href="https://huggingface.co/collections/TheFinAI/calm-credit-scoring-and-risk-assessment-6ac7153781499dcd5001572e">Data</a> ·
  <a href="https://huggingface.co/daishen/CALM-7B">Models</a>
</p>

<p>
  Duanyu Feng<sup>1</sup>&emsp;
  Yongfu Dai<sup>1</sup>&emsp;
  <a href="https://jimin.chancefocus.com/">Jimin Huang</a><sup>2</sup>&emsp;
  Yifang Zhang<sup>1</sup>&emsp;
  Qianqian Xie<sup>3</sup>&emsp;
  Weiguang Han<sup>3</sup>&emsp;
  <a href="https://warrington.ufl.edu/directory/person/12693/">Alejandro Lopez-Lira</a><sup>4</sup>&emsp;
  Hao Wang*<sup>1</sup>
</p>

<p>
  <sup>1</sup>Sichuan University&emsp;
  <sup>2</sup>ChanceFocus AMC&emsp;
  <sup>3</sup>Wuhan University&emsp;
  <sup>4</sup>University of Florida
</p>

<p>
  <img src="https://i.postimg.cc/NjKhDkGY/DFAF986-CCD6529-E52-D7830-F180-D-C37-C7-DEE-4340.png" alt="Sichuan University Logo" height="60px">&emsp;
  <img src="https://i.postimg.cc/xTsgsrqN/logo11.png" alt="ChanceFocus AMC Logo" height="60px">&emsp;
  <img src="https://i.postimg.cc/CLtkBwz7/57-EDDD9-FB0-DF712-F3-AB627163-C2-1-EF15655-13-FCA.png" alt="Wuhan University Logo" height="60px">&emsp;
  <img src="https://i.postimg.cc/XY1s2RHD/University-of-Florida-Logo-1536x864.jpg" alt="University of Florida Logo" height="60px">
</p>

</div>

## Overview

Our project **CALM** aims to better utilize large language models (LLMs) to study issues related to credit and risk assessment in the financial industry, including data construction, model training, model evaluation, and bias analysis. We built a benchmark and instruction dataset with 70K samples drawn from 9 public datasets across 4 tasks, fine-tuned the **C**redit and Risk **A**ssessment **L**arge Language **M**odel (CALM-7B) on Llama2-chat, and analyzed the biases of LLMs on these tasks.

This project releases:

- [Credit and Risk Assessment Instruction Dataset](./data)
- [Credit and Risk Assessment Benchmark](#credit-and-risk-assessment-benchmark)
- [Credit and Risk Assessment LLM Bias Analysis](./src/bias)
- [**C**redit and Risk **A**ssessment **L**arge Language **M**odel (CALM-7B)](https://huggingface.co/daishen/CALM-7B)
- [Our paper "Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models"](https://arxiv.org/abs/2310.00566)

## News

- **[2023/10/15]** Our [Credit and Risk Assessment Large Language Model (CALM-7B)](https://huggingface.co/daishen/CALM-7B) has been officially released.
- **[2023/10/01]** CALM v1.0 has been officially released, open-sourcing the [Instruction Dataset](./data), [Benchmark](#credit-and-risk-assessment-benchmark), [Bias Analysis](./src/bias) and [paper](https://arxiv.org/abs/2310.00566).

## Contents

- [Introduction](#introduction)
- [Credit and Risk Assessment Instruction Dataset](#credit-and-risk-assessment-instruction-dataset)
- [Fine-tuning and Inference](#fine-tuning-and-inference)
- [Credit and Risk Assessment Benchmark](#credit-and-risk-assessment-benchmark)
- [Bias Analysis](#bias-analysis)
- [Resources on Hugging Face](#resources-on-hugging-face)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)
- [Citation](#citation)
- [License](#license)

## Introduction

Credit and risk assessment is a series of unique tasks in the financial industry, which play a significant role in various aspects of individuals and society. It evaluates the creditworthiness and risk situation of individuals, businesses, and even countries. The assessment includes tasks such as credit scoring, fraud detection, financial distress identification, and claim analysis. These tasks cover most aspects of our personal lives and social development, such as lending, insurance, and investment, which have a huge financial impact for the whole of society. Meanwhile, it is essential to note that data used for credit and risk assessment may collect some sensitive information such as gender and age. This may further lead to discrimination against certain groups. Therefore, in credit and risk assessment, potential biases should be taken into account to avoid negative social consequences.

Currently, LLMs' potential in financial natural language processing (NLP) tasks is significantly emphasized. Some studies explore the ability of the LLMs to improve quantitative stock trading strategies, predict stock market returns, and analyze investment context. This further raises the question of whether LLMs will have sufficient capabilities if used for credit and risk assessment or if it may bring some bias. 

We constructed the first comprehensive **benchmark** and **instruction datasets** with 70K data samples. We amalgamated an extensive array of 9 existing public datasets on 4 tasks, including credit scoring, fraud detection, financial distress identification, and claim analysis. It includes data that is meaningless, imbalanced, and high-dimensional. 

We fine-tune our Credit and Risk Assessment LLM (**CALM**) based on Llama2-chat with the instructions.

We also identify the inherent **biases** within the models.

## Credit and Risk Assessment Instruction Dataset

### Data Collection
In the data collection, we provide a comprehensive overview of the tasks related to credit and risk assessment, and gather nine well-known open-source datasets with detailed introductions. 

We classify it into four classes of tasks: credit scoring, fraud detection, financial distress identification, and claim analysis. All of them are binary classification problems and tabular-based.
We finally collect nine datasets for these tasks, including three credit scoring datasets, two fraud detection datasets, two financial distress identification datasets, and two claim analysis datasets. 
Among them, three original datasets (Australia, Credit Card Fraud, PortoSeguro) have been processed into meaningless symbols. 
For those datasets with a large amount, we sample them based on the label distribution for our data collection.
All of these original datasets are open-source and have been widely used and validated by the existing works.

![Image](./data.png)

**For more details, see our paper** ["Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models"](https://arxiv.org/abs/2310.00566).

### Data Construction
In the data construction, we detail how we construct various instruction-tuning forms tailored to different datasets.

<div align="center"> <img src="./prompt.png" width = 800 /> </div>

Due to the unique nature of some datasets (many features or meaningless symbols), we create two different forms of instructions, table-based and description-based, to evaluate the LLMs. 
Both forms of instruction first follow a similar template. For instance, [prompt] is a prompt created for each data, [input] expresses a single data from the original datasets, [output] is the corresponding correct label for the [input], [choices] are the available label options (such as ["Yes", "No"] or ["good", "bad"]), and [index] is the index of the correct label in the label set. Therefore, the differences between the two instructions come from the specific construction of [prompt] and [input].

**Table-based Instruction.**
This form of instruction is designed for data that contains too many features or meaningless symbols. As their features are too many or do not have any semantic information, it is hard to describe them in natural language with limited words. Therefore, in the [prompt], we explain the data is composed of meaningless symbols and provide the number of features; in the [input] section, we directly tell the values of each data.
It is concise and convenient to construct for highly structured data. 


**Description-based Instruction.**
This form of instruction is designed for the rest datasets that have clear semantic information about the features.
Here, we use natural language in [input] to re-explain the meaning of features and the corresponding numerical values for each data. For instance, in credit scoring, we transfer the features as "The purpose is car (new). The state of credit amount is 2366." This form makes LLMs easier to understand the data.

## Fine-tuning and Inference

We further build our **C**redit and Risk **A**ssessment **L**arge Language **M**odel (CALM-7B) by fine-tuning the Llama2-chat, with the above instruction dataset. Except for the Lending Club, Polish, and PortoSeguro datasets, we leave them to test the LLM's ability on similar tasks.

On some extremely imbalanced datasets, including Credit Card Fraud, ccFraud, Taiwan Economic Journal and Travel Insurance, we resample the minority class on the training set. After resampling, the ratio of majority class samples to minority class samples was 2:1.

The code of fine-tuning our LLM (CALM-7B) can be found in [CALM-train](https://github.com/Dai-shen/CALM-train).

## Credit and Risk Assessment Benchmark

We choose the latest and most popular LLMs as the baselines, including open resources and non-open resources. For the open resource LLMs, we use 6 LLMs, Bloomz, Vicuna, Llama1, Llama2, Llama2-chat, and Chatglm2. To ensure fairness and minimize computation costs, we use the around 7B-parameters version for all these LLMs. For the non-open resource LLMs, we use ChatGPT and GPT-4. In addition, we have also included a comparison of the results from the SOTA expert system models on various datasets.

### Evaluation Data

| Task | Dataset | Hugging Face (TheFinAI) | Original upload |
| --- | --- | --- | --- |
| Credit Scoring | German | — | [daishen/cra-german](https://huggingface.co/datasets/daishen/cra-german) |
| Credit Scoring | Australia | — | [daishen/cra-australian](https://huggingface.co/datasets/daishen/cra-australian) |
| Credit Scoring | Lending Club | [TheFinAI/en-lendingclub](https://huggingface.co/datasets/TheFinAI/en-lendingclub) | [daishen/cra-lendingclub](https://huggingface.co/datasets/daishen/cra-lendingclub) |
| Fraud Detection | Credit Card Fraud | [TheFinAI/en-ccf](https://huggingface.co/datasets/TheFinAI/en-ccf) | [daishen/cra-ccf](https://huggingface.co/datasets/daishen/cra-ccf) |
| Fraud Detection | ccFraud | [TheFinAI/en-ccfraud](https://huggingface.co/datasets/TheFinAI/en-ccfraud) | [daishen/cra-ccfraud](https://huggingface.co/datasets/daishen/cra-ccfraud) |
| Financial Distress Identification | Polish | [TheFinAI/en-polish](https://huggingface.co/datasets/TheFinAI/en-polish) | [daishen/cra-polish](https://huggingface.co/datasets/daishen/cra-polish) |
| Financial Distress Identification | Taiwan Economic Journal | [TheFinAI/en-taiwan](https://huggingface.co/datasets/TheFinAI/en-taiwan) | [daishen/cra-taiwan](https://huggingface.co/datasets/daishen/cra-taiwan) |
| Claim Analysis | PortoSeguro | [TheFinAI/en-portoseguro](https://huggingface.co/datasets/TheFinAI/en-portoseguro) | [daishen/cra-portoseguro](https://huggingface.co/datasets/daishen/cra-portoseguro) |
| Claim Analysis | Travel Insurance | [TheFinAI/en-travelinsurance](https://huggingface.co/datasets/TheFinAI/en-travelinsurance) | [daishen/cra-travelinsurace](https://huggingface.co/datasets/daishen/cra-travelinsurace) |

### Evaluation Results

![Image](./res.png)

## Bias Analysis

We explore bias in three datasets using three LLMs (ChatGPT, GPT-4, and our model) that have the processing ability of the tasks. We consider the impact of gender, age, and foreign status on German, the impact of gender on ccFraud, and the impact of age on Travel Insurance. We set the old, female and foreigner as the unprivileged groups for gender, age, and foreign status, respectively. Except for the 'foreigner' in German, the DI values are near 1, indicating that the original datasets are unbiased on these sensitive features.

<div align="center"> <img src="./bias1.png" width = 300 /> </div>

After the prediction of the LLMs, we compute the EOD and AOD on these features. 
For ChatGPT and GPT-4, it indicates that they have a bias in some special cases. For example, GPT-4 is more likely to give females wrong predictions (AOD is -0.273) on the ccFraud dataset and prefer foreign workers on the German dataset (EOD is 0.289), even though the original data is unbiased (DI close to 1); on the German dataset, ChatGPT prefers to lend money to older people (EOD is 0.137). It's also interesting to note that the potential biases that exist in both ChatGPT and GPT-4 are not completely consistent with each other ('gender' and 'age' in German, and 'gender' in ccFraud). 

<div align="center"> <img src="./bias2.png" width = 400 /> </div>

## Resources on Hugging Face

| Resource | Link |
| --- | --- |
| CALM credit scoring and risk assessment (collection) | [TheFinAI/calm-credit-scoring-and-risk-assessment](https://huggingface.co/collections/TheFinAI/calm-credit-scoring-and-risk-assessment-6ac7153781499dcd5001572e) |
| CALM-7B (model) | [daishen/CALM-7B](https://huggingface.co/daishen/CALM-7B) |
| Lending Club | [TheFinAI/en-lendingclub](https://huggingface.co/datasets/TheFinAI/en-lendingclub) |
| Credit Card Fraud | [TheFinAI/en-ccf](https://huggingface.co/datasets/TheFinAI/en-ccf) |
| ccFraud | [TheFinAI/en-ccfraud](https://huggingface.co/datasets/TheFinAI/en-ccfraud) |
| Polish | [TheFinAI/en-polish](https://huggingface.co/datasets/TheFinAI/en-polish) |
| Taiwan Economic Journal | [TheFinAI/en-taiwan](https://huggingface.co/datasets/TheFinAI/en-taiwan) |
| PortoSeguro | [TheFinAI/en-portoseguro](https://huggingface.co/datasets/TheFinAI/en-portoseguro) |
| Travel Insurance | [TheFinAI/en-travelinsurance](https://huggingface.co/datasets/TheFinAI/en-travelinsurance) |

## Acknowledgements

This project is built upon the following open-source projects, and we are really thankful for them:

- [**Llama 2**](https://github.com/facebookresearch/llama)
- [**PIXIU**](https://github.com/The-FinAI/PIXIU)
- [**LLMindCraft**](https://github.com/XplainMind/LLMindCraft)

## Disclaimer

This project is provided for **academic and educational purposes** only. We do not take responsibility for any issues, risks, or adverse consequences that may arise from the use of this project.

## Citation

If our project has been helpful for your work, please cite as follows:
```bibtex
@misc{feng2023empowering,
      title={Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models}, 
      author={Duanyu Feng and Yongfu Dai and Jimin Huang and Yifang Zhang and Qianqian Xie and Weiguang Han and Alejandro Lopez-Lira and Hao Wang},
      year={2023},
      eprint={2310.00566},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). Datasets and models on Hugging Face keep their own licenses, stated on each card.

---

<p align="center">Built by <a href="https://thefin.ai">The Fin AI</a> · <a href="https://huggingface.co/TheFinAI">Hugging Face</a> · <a href="https://github.com/The-FinAI">GitHub</a></p>
