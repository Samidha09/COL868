import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
import torch_geometric as tg
import torch_geometric.datasets
import time

import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
dataset_dir = 'data/Brightkite'
print("Loading data...")
G = nx.read_edgelist(
    open(dataset_dir + "/Brightkite_edges.txt", "rb"))

node_dict = {}
for id, node in enumerate(G.nodes()):
    node_dict[node] = id
feature = []
feature_id = []

fp = open(dataset_dir + "/Brightkite_totalCheckins.txt", "r")
lines = fp.readlines()
cnt = 0
for line in lines:
    line = line.strip()
    line = line.split()
    if (len(line) < 5):
        continue
    feature_id.append(int(line[0]))
    feature.append([time.mktime(time.strptime(
        line[1], '%Y-%m-%dT%H:%M:%SZ')), float(line[2]), float(line[3])])
    cnt += 1
    if (cnt == 10):
        break
    # print(line)

feature = np.array(feature)
feature[:, 0] = np.log(feature[:, 0] + 1.0)
feature[:, 1] = np.log(
    feature[:, 1] - min(np.min(feature[:, 1]), -1)+1)
feature[:, 2] = np.log(
    feature[:, 2] - min(np.min(feature[:, 2]), -1)+1)

feature_map = {}
for i in range(len(feature_id)):
    if (feature_id[i] not in feature_map):
        feature_map[feature_id[i]] = []
    feature_map[feature_id[i]].append(feature[i])
print(feature_map)
feature_actual_map = {}
for k in feature_map:
    feature_actual_map[k] = np.mean(feature_map[k], axis=0)
print(feature_actual_map)

comps = [comp for comp in nx.connected_components(G) if len(comp) > 10]
graphs = [G.subgraph(comp) for comp in comps]
id_all = []
features = []
count = 0
for comp in comps:
    id_temp = []
    feat_temp = []
    for node in comp:
        id = node_dict[node]
        id_temp.append(id)
        if (id not in feature_actual_map):
            feat_temp.append([0.0, 0.0, 0.0])
            count = count+1
        else:
            feat_temp.append(feature_actual_map[id])
        id_all.append(np.array(id_temp))
        features.append(np.array(feat_temp))

print("Not found features of %d nodes" % {count})
