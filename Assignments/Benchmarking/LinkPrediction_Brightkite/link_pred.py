'''
References: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
https://github.com/dmlc/dgl/blob/99751d49605ca5c080343af3320508d3ba77fb85/tutorials/large/L2_large_link_prediction.py
'''
import sklearn.metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import tqdm
import dgl
import networkx as nx
import torch
import numpy as np
import pickle
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from GraphSAGE import GraphSAGE
#from DotProduct import DotPredictor
from MLP import MLPPredictor

# load arguments
parser = argparse.ArgumentParser(description='GNN arguments.')

parser.add_argument('--model_type', type=str,
                    help='Type of GNN model.')
parser.add_argument('--num_layers', type=int,
                    help='Number of graph conv layers')
# parser.add_argument('--hidden_dim', type=int,
#                     help='Training hidden size')
# parser.add_argument('--aggregator', type=str,
#                     help='Aggregator for GraphSAGE')
parser.add_argument('--epochs', type=int,
                    help='Number of training epochs')

parser.set_defaults(model_type='GCN',
                    num_layers=1,
                    hidden_dim=32,
                    aggregator='mean',
                    epochs=50)
args = parser.parse_args()
k_folds = 5
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

node_features = g.ndata['feat']

train_time_per_fold = []
roc_auc = []
# f1 = []
# prec = []
# recall = []
fp1 = open('./results/fold_results_' +
           args.model_type + str(args.epochs) + '.txt', 'w')
for fold in range(k_folds):
    print('--------- FOLD: {} ----------'.format(fold))
    fold_metrics = []
    # Split edge set for training and testing
    print("Split edge set for training and testing")
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    #print(len(eids[:test_size]), len(eids[test_size:]))

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Inefficiently sampling negative graphs
    # Find all negative edges and split them for training and testing
    # print("Find all negative edges and split them for training and testing")
    # adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    # adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    # neg_u, neg_v = np.where(adj_neg != 0)

    # neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    # test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]
    #                                ], neg_v[neg_eids[:test_size]]
    # train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]
    #                                  ], neg_v[neg_eids[test_size:]]

    # Defining Neighbor Sampler and Data Loader in DGL - (Efficient)
    # generates 5 negative examples per positive example
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    neighbours_in_each_layer = []
    for i in range(args.num_layers):
        neighbours_in_each_layer.append(6)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        neighbours_in_each_layer)

    train_dataloader = dgl.dataloading.EdgeDataLoader(
        # The following arguments are specific to NodeDataLoader.
        g,                                  # The graph
        eids[test_size:],  # The edges to iterate over
        sampler,                                # The neighbor sampler
        negative_sampler=negative_sampler,      # The negative sampler
        device=device,                          # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    test_dataloader = dgl.dataloading.EdgeDataLoader(
        # The following arguments are specific to NodeDataLoader.
        g,                                  # The graph
        eids[:test_size],  # The edges to iterate over
        sampler,                                # The neighbor sampler
        negative_sampler=negative_sampler,      # The negative sampler
        device=device,                          # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    # Checking values returned by loader
    input_nodes, pos_graph, neg_graph, mfgs = next(iter(test_dataloader))
    print('Number of input nodes:', len(input_nodes))
    print('Positive graph # nodes:', pos_graph.number_of_nodes(),
          '# edges:', pos_graph.number_of_edges())
    print('Negative graph # nodes:', neg_graph.number_of_nodes(),
          '# edges:', neg_graph.number_of_edges())
    print(mfgs)

    # # remove test edges from graph
    # print("remove test edges from graph")
    # train_g = dgl.remove_edges(g, eids[:test_size])

    # # generate positive and negative graph
    # print("generate positive and negative graph")
    # train_pos_g = dgl.graph((train_pos_u, train_pos_v),
    #                         num_nodes=g.number_of_nodes())
    # train_neg_g = dgl.graph((train_neg_u, train_neg_v),
    #                         num_nodes=g.number_of_nodes())

    # test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    # test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    # model configurations
    print("model configurations")
    if(args.model_type == 'SAGE'):
        model = GraphSAGE(g.ndata['feat'].shape[1],
                          32, 'mean', args.num_layers)
    else:
        model = GraphSAGE(g.ndata['feat'].shape[1], 32, 'gcn', args.num_layers)
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

    # def compute_f1(pos_score, neg_score):
    #     scores = torch.cat([pos_score, neg_score]).numpy()
    #     # print(score[:1000])
    #     scores = np.where(scores <= 0.5, scores, 0)
    #     scores = np.where(scores > 0.5, scores, 1)
    #     labels = torch.cat(
    #         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    #     return f1_score(labels, scores)

    # def compute_prec(pos_score, neg_score):
    #     scores = torch.cat([pos_score, neg_score]).numpy()
    #     scores = np.where(scores <= 0.5, scores, 0)
    #     scores = np.where(scores > 0.5, scores, 1)
    #     labels = torch.cat(
    #         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    #     return precision_score(labels, scores)

    # def compute_recall(pos_score, neg_score):
    #     scores = torch.cat([pos_score, neg_score]).numpy()
    #     scores = np.where(scores <= 0.5, scores, 0)
    #     scores = np.where(scores > 0.5, scores, 1)
    #     labels = torch.cat(
    #         [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    #     return recall_score(labels, scores)

    # ----------- 3. set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    optimizer = torch.optim.Adam(itertools.chain(
        model.parameters(), pred.parameters()), lr=0.01)

    # ----------- 4. training -------------------------------- #
    start_time = time.time()
    for epoch in range(args.epochs):
        print('--------- Epoch {} -----------'.format(epoch))
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']

                outputs = model(mfgs, inputs).float()
                pos_score = pred(pos_graph, outputs)
                neg_score = pred(neg_graph, outputs)

                score = torch.cat([pos_score, neg_score])
                label = torch.cat([torch.ones_like(pos_score),
                                   torch.zeros_like(neg_score)])
                loss = F.binary_cross_entropy_with_logits(score, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

                if (step % 5 == 0):
                    print('In step {}, loss: {}'.format(step, loss))
    end_time = time.time()
    train_time_per_fold.append(end_time - start_time)
    print("Fold {} : Training time in seconds {}".format(
        fold, end_time-start_time))
    # ----------- 5. check results - -----------------------
    batches = 0
    score_array = [0, 0, 0, 0]
    with torch.no_grad():
        with tqdm.tqdm(test_dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']

                outputs = model(mfgs, inputs).float()
                pos_score = pred(pos_graph, outputs)
                neg_score = pred(neg_graph, outputs)

                # print("Positive Score: ", pos_score[:100])
                # print("Negative Scor: ", neg_score[:100])

                batches += 1
                metrics = []
                metrics.append(compute_auc(pos_score, neg_score))
                # metrics.append(compute_f1(pos_score, neg_score))
                # metrics.append(compute_prec(pos_score, neg_score))
                # metrics.append(compute_recall(pos_score, neg_score))
                print('Step: ', step)
                print('ROC-AUC Score: ', metrics[0])
                # print('F1-Score: ', metrics[1])
                # print('Precision Score: ', metrics[2])
                # print('Recall Score: ', metrics[3])

                score_array[0] += metrics[0]
                # score_array[1] += metrics[1]
                # score_array[2] += metrics[2]
                # score_array[3] += metrics[3]

    roc_auc.append(score_array[0]/batches)
    # f1.append(score_array[1]/batches)
    # prec.append(score_array[2]/batches)
    # recall.append(score_array[3]/batches)

    # writing results in fp1
    fp1.write('--------- FOLD: {} ----------\n'.format(fold))
    fp1.write('ROC-AUC Score: {}\n'.format(score_array[0]/batches))
    # fp1.write('F1 Score: {}\n'.format(score_array[1]/batches))
    # fp1.write('Precision Score: {}\n'.format(score_array[2]/batches))
    # fp1.write('Recall Score: {}\n'.format(score_array[3]/batches))
    fp1.write('Average Training Time per Fold(sec): {}\n'.format(
        end_time - start_time))
    fp1.write('\n-------------------\n\n')

# calculate average result of 5 folds
mean_roc_auc = np.mean(np.array(roc_auc))
# mean_f1 = np.mean(np.array(f1))
# mean_prec = np.mean(np.array(prec))
# mean_recall = np.mean(np.array(recall))
mean_training_time_per_fold = np.mean(np.array(train_time_per_fold))

fp = open('./results/final_results_' +
          args.model_type + str(args.epochs) + '.txt', 'w')
fp.write('ROC-AUC Score: {}\n'.format(mean_roc_auc))
# fp.write('F1 Score: {}\n'.format(mean_f1))
# fp.write('Precision Score: {}\n'.format(mean_prec))
# fp.write('Recall Score: {}\n'.format(mean_recall))
fp.write('Average Training Time per Fold(sec): {}\n'.format(
    mean_training_time_per_fold))
fp.close()
fp1.close()
