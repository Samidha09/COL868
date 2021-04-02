'''
https://matplotlib.org/stable/tutorials/introductory/customizing.html
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# xls = pd.ExcelFile('./Final_Results/Stats/MLNC.xlsx')
xls = pd.ExcelFile('MLNC.xlsx')
# Graph_Sage time results
df1 = pd.read_excel(xls, 1)
# GCN time results
df2 = pd.read_excel(xls, 4)
# sage dimensions results
df3 = pd.read_excel(xls, 2)
# GCN embedding dimensions
df4 = pd.read_excel(xls, 5)


plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.marker"] = 'o'
plt.rcParams["axes.titlesize"] = 20
# plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.linewidth"] = 0.6


# plots for time vs quality - Experiment 2
# Graph_Sage time results
sage_time_taken = np.array(df1['avg_training_time_per_fold (in sec)'])
sage_time_f1 = np.array(df1['test_f1'])
sage_time_prec = np.array(df1['test_prec'])
sage_time_recall = np.array(df1['test_recall'])
sage_time_roc_auc = np.array(df1['test_roc'])

# plt.style.use('seaborn-poster')

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(sage_time_taken, sage_time_f1, color='g', label="F1_score")
plt.plot(sage_time_taken, sage_time_prec, color='r', label="Precision")
plt.plot(sage_time_taken, sage_time_recall, color='orange',label="Recall")
plt.plot(sage_time_taken, sage_time_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_fold(in sec)')
plt.ylabel('Performance')
plt.title('GraphSAGE ppi')
plt.show()

directory_name = "./images/"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
fig.savefig(directory_name + "GraphSAGE_mlnc_time" + ".png")

# GCN time results
gcn_time_taken = np.array(df2['avg_training_time_per_fold (in sec)'])
gcn_time_f1 = np.array(df2['test_f1'])
gcn_time_prec = np.array(df2['test_prec'])
gcn_time_recall = np.array(df2['test_recall'])
gcn_time_roc_auc = np.array(df2['test_roc'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_time_taken, gcn_time_f1, color='g', label="F1_score")
plt.plot(gcn_time_taken, gcn_time_prec, color='r', label="Precision")
plt.plot(gcn_time_taken, gcn_time_recall, color='orange', label="Recall")
plt.plot(gcn_time_taken, gcn_time_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_fold(in sec)')
plt.ylabel('Performance')
plt.title('GCN ppi')
plt.show()

fig.savefig(directory_name + "GCN_mlnc_time" + ".png")

# plots for embedding_dimension vs quality - Experiment 2
# sage dimensions results
sage_dim = np.array(df3['embedding_dim'])
sage_roc_auc = np.array(df3['test_roc'])
sage_f1 = np.array(df3['test_f1'])
sage_prec = np.array(df3['test_prec'])
sage_recall = np.array(df3['test_recall'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(sage_dim, sage_f1, color='g', label="F1_score")
plt.plot(sage_dim, sage_prec, color='r', label="Precision")
plt.plot(sage_dim, sage_recall, color='orange', label="Recall")
plt.plot(sage_dim, sage_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Embedding dimensions')
plt.ylabel('Performance')
plt.title('GraphSAGE ppi')
plt.show()

fig.savefig(directory_name + "GraphSAGE_mlnc_dim" + ".png")

# GCN embedding dimensions results
gcn_dim = np.array(df4['embedding_dim'])
gcn_roc_auc = np.array(df4['test_roc'])
gcn_f1 = np.array(df4['test_f1'])
gcn_prec = np.array(df4['test_prec'])
gcn_recall = np.array(df4['test_recall'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_dim, gcn_f1, color='g', label="F1_score")
plt.plot(gcn_dim, gcn_prec, color='r', label="Precision")
plt.plot(gcn_dim, gcn_recall, color='orange', label="Recall")
plt.plot(gcn_dim, gcn_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Embedding dimensions')
plt.ylabel('Performance')
plt.title('GCN ppi')
plt.show()

fig.savefig(directory_name + "GCN_mlnc_dim" + ".png")
