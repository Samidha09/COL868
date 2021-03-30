'''
References: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
'''
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import dgl
import networkx as nx
import torch
import numpy as np
import pickle
import time
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from GraphSAGE import GraphSAGE
#from DotProduct import DotPredictor
from MLP import MLPPredictor
# Load Data

dataset_dir = 'data/Brightkite'
nxgraph = nx.read_edgelist(
    open(dataset_dir + "/Brightkite_edges.txt", "rb"))

device = 'cpu'  # change to cuda for gpu

fp = open(dataset_dir + '/updated_actual_features_dict.pickle', "rb")
feature_actual_map = pickle.load(fp)
fp.close()


# nodes = []
# for key in feature_actual_map.keys():
#     nodes.append(key)

# for node in nxgraph.nodes:
#     if (int(node) not in nodes):
#         feature_actual_map[int(node)] = [0.0, 0.0, 0.0]


# fp = open(dataset_dir + '/updated_actual_features_dict.pickle', "wb")
# pickle.dump(feature_actual_map, fp)
# fp.close()

print("Number of nodes: ", nxgraph.number_of_nodes())
print("Number of edges: ", nxgraph.number_of_edges())

# assigning attributes to nodes
for node in nxgraph.nodes:
    nxgraph.nodes[node]['feat'] = feature_actual_map[int(node)]  # [0]
    # nxgraph.nodes[node]['latitude'] = feature_actual_map[int(node)][1]
    # nxgraph.nodes[node]['longitude'] = feature_actual_map[int(node)][2]

# Make dgl graph
g = dgl.from_networkx(nxgraph, node_attrs=['feat'])

# Split edge set for training and testing
print("Split edge set for training and testing")
u, v = g.edges()

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
print("Find all negative edges and split them for training and testing")
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]
                               ], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]
                                 ], neg_v[neg_eids[test_size:]]

# remove test edges from graph
print("remove test edges from graph")
train_g = dgl.remove_edges(g, eids[:test_size])

# generate positive and negative graph
print("generate positive and negative graph")
train_pos_g = dgl.graph((train_pos_u, train_pos_v),
                        num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v),
                        num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# model configurations
print("model configurations")
model = GraphSAGE(train_g.ndata['feat'].shape[1], 32)
# You can replace DotPredictor with MLPPredictor.
#pred = MLPPredictor(16)
pred = MLPPredictor(32)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]),
                        torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(
    model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
print("training")
all_logits = []
for e in range(1):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))
