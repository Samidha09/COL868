# Benchmarking

In this assignment, we have performed a benchmarking exercise on **GCN** and **GraphSAGE** models.

## Tasks:

1. Pairwise Node Classification - Proteins dataset
2. Link Prediction - PPI and Brightkite dataset
3. Multi-Class Node Classification - PPI dataset

## Report

Link to report: [Report](https://github.com/Samidha09/COL868/blob/master/Assignments/Benchmarking/Benchmarking.pdf)

## Setup

These are the python packages required to run the above codes in Google Colab:

```bash
!pip install torch

!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

!pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

!pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html

!pip install torch-geometric

!pip install tensorboardx

!pip install dgl
```

## Run (Google Colab):

1. **Pairwise Node Classification**

```bash
%cd PNC_LP_ppi/P-GNN-master/
```

- 2-layer GCN, protein dataset, pairwise node classification task

```bash
!python main.py --model GCN --layer_num 2 --dataset protein --task link_pair
```

- 3-layer GraphSAGE, protein dataset
  GraphSAGE, pairwise node classification task

```bash
!python main.py --model SAGE --layer_num 3 --dataset protein --task link_pair
```

2. **Link Prediction**

**PPI**

```bash
%cd PNC_LP_ppi/P-GNN-master/
!unzip data/ppi.zip
```

- 2-layer GCN, PPI dataset, link prediction task

```bash
!python main.py --model GCN --layer_num 2 --dataset ppi --task link
```

- 3-layer GraphSAGE, PPI dataset, link prediction task

```bash
!python main.py --model SAGE --layer_num 3 --dataset ppi --task link
```

**Brightkite**

```bash
%cd LinkPrediction_Brightkite
!unzip data/Brightkite.zip
```

- 2-layer GCN, Brightkite dataset, link prediction task

```bash
!python link_pred.py --model_type SAGE --num_layers 2
```

- 2-layer GraphSAGE, Brightkite dataset, link prediction task

```bash
!python link_pred.py --model_type SAGE --num_layers 2
```

_**Note**: Currently link prediction on Brightkite is only written for 2 layer models._

3. **Multi-Class Node Classification**

```bash
%cd NodeClassification
```

**Running custom experiments**

3-layer GraphSAGE, hidden dimensions 256, aggregator max, proteins dataset, multi-class node classification task

```bash
!python train.py --model_type=SAGE --num_layers=3 --hidden_dim=256 --aggregator=max
```

2-layer GCN, hidden dimensions 256,proteins dataset, multi-class node classification task

```bash
!python train.py --model_type=GCN --num_layers=3 --hidden_dim=256
```

_**Note**: Possible aggregators for GraphSAGE: [mean, max, add]_

**Experiment - 1**

```bash
!python run.py
```

**Experiment - 2**

```bash
!python exp2.py
```

**Experiment - 3**

```bash
!python exp3.py
```

_Read more about experiments in the report_

## References

Link Prediction PPI & Pairwise Node Classification Proteins: [P-GNN](https://github.com/JiaxuanYou/P-GNN)

Link Prediction Brightkite:
[dgl](https://github.com/dmlc/dgl)

Node Classification: [pytorch](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) and [GraphSAGE_model](https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py)
