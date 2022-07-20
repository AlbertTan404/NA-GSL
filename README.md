# NA-GSL

A PyTorch implemention of NA-GSL: Exploring attention mechanism for graph similarity learning
![](https://github.com/AlbertTan404/NA-GSL/blob/main/figure/model_overview.png)

We propose a unified graph similarity learning framework, *N*ode-wise *A*ttention guided *G*raph *S*imilarity *L*earning, *NA-GSL*, involving i) a hybrid of graph convolution and graph self-attention for node embedding learning, ii) a cross-attention (GCA) module for graph interaction modeling, iii) similarity-wise self-attention (SSA) module for graph similarity matrix fusion and alignment and iv) graph similarity structure learning for predict the similarity score.

## Requirements
* python3.8
* pytorch==1.10.2
* torch_geometric==1.10
* torch_scatter==2.0.9
* torch_sparse==0.6.12
* torch_cluster==1.5.9
* texttable==1.6.4

## Run
```
python main.py --dataset=LINUX
```
