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

TO install the package, run

```
pip3 install -r requirements.txt
```

## Experiment

To train our model with the parameters setup we provide, simply run

```
python3 train.py
```

## Result
