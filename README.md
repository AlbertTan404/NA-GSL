# NA-GSL

A PyTorch implementation of NA-GSL: Exploring attention mechanism for graph similarity learning

We propose a unified graph similarity learning framework, **N**ode-wise **A**ttention guided **G**raph **S**imilarity **L**earning, **NA-GSL**, involving i) a hybrid of graph convolution and graph self-attention for node embedding learning, ii) a cross-attention (GCA) module for graph interaction modeling, iii) similarity-wise self-attention (SSA) module for graph similarity matrix fusion and alignment and iv) graph similarity structure learning for predicting the similarity score.

## Requirements
* python==3.8
* pytorch==1.10.2
* torch_geometric==1.10
* tqdm
* scipy
* texttable

## Run
```
cd src
python main.py --dataset=LINUX
```
