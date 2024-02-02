# categorical-tabnet
[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

@ Code for Paper: categorical boosting based tabnet

## Abstract
Recent deep learning models perform well in image and natural language processing. 
However, in tabular data, there is a problem that good performance is not achieved due to data-level problems. 
Recently, TabNet, a model that overcomes these shortcomings, has been widely used for tabular data learning.
However, categorical variable data does not perform significantly in tabular data. 
To solve this problem, Catboost Encoding method is used to solve the problem. 
In the case of this model, the pre-processing of categorical variable data was well utilized to derive more performance than other models, and it showed better performance than other encoding techniques.

## Model Architecture
| Model Architecture |
|:-:|
| ![Model-Architecture](https://github.com/ds-wook/categorical-tabnet/assets/46340424/fd129cc1-390f-4b9c-af8d-d9efd6db71de) |
|The model was designed to train on distinct categorical and numerical datasets. Fig. illustrates the transformation of the categorical dataset into numerical data through the utilization of the CatBoost encoder. The transformed numerical data, alongside the batch-normalized numerical data, were then trained with the TabNet encoder to yield the final outcomes.|

## Experiments
| Results |
|:-:|
| ![image](https://github.com/ds-wook/categorical-tabnet/assets/46340424/7ece9f95-fe75-4f30-b1e7-d0c66cb13e23) |
| Categorical boosting demonstrates superior performance compared to TabNet and exhibits excellent results across diverse datasets.|

## Interpretability
| Mask |
|:-:|
| ![image](https://github.com/ds-wook/categorical-tabnet/assets/46340424/f2dff6c2-23f7-42d9-a096-1279df17b04a) |
|In TabNet, if attention is distributed equally among features, in this approach, attention to different features varies across layers.|