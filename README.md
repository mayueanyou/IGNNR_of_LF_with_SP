# IGNNR_of_LF_with_SP

This the repository for CIS700 Throem Proving Project.


## Table of Contents

- [Introduction](#Introduction)

- [Dataset](#Dataset)

- [Environment_Setup](#Environment_Setup)

- [Experiment](#Experiment)

- [Result](#Result)

## Introduction

Team Mumber:

Qinwei Huang, Xiao Yi, Yue Ma, Zifan Wang

## Dataset

The dataset we use for this project is HolStep. It could be download from the offcial [website](http://cl-informatik.uibk.ac.at/cek/holstep/)

To train or test our model, please download the HolStep.tgz file into the datasets folder and run the following code to unzip it.

```
tar -xvzf ./datasets/holstep.tgz -C ./datasets/
```

Then to prepocess the data, please run the command to transfer the data into the data folder.

```
python3 process_data.py
```
## Environment_Setup

Our model runs in python3 and needs the following packages.

- torch
- numpy
- networkx
- graphviz
- allennlp
- sentencepiece
- penman

To install the package, run

```
pip3 install -r requirements.txt
```

## Experiment

To train our model with the parameters setup we provide, simply run

```
python3 train.py seed num_rounds node_emb_dim
```

All of the parameters should be int data.

seed is to generate different trainning set.

num_rounds and node_emb_dim are two parameters for the model.

## Result

We have done 4 experiments. Each of them we test with three random seeds represented in three different color.

|num_rounds:0 node_emb_dim: 16 | num_rounds:3 node_emb_dim: 16 |
|-|-|
|<img src="https://github.com/mayueanyou/IGNNR_of_LF_with_SP/blob/main/experiments/0_16.png">|<img src="https://github.com/mayueanyou/IGNNR_of_LF_with_SP/blob/main/experiments/3_16.png">|

|num_rounds:0 node_emb_dim: 32|num_rounds:3 node_emb_dim: 32|
|-|-|
|<img src="https://github.com/mayueanyou/IGNNR_of_LF_with_SP/blob/main/experiments/0_32.png">|<img src="https://github.com/mayueanyou/IGNNR_of_LF_with_SP/blob/main/experiments/3_32.png">|



