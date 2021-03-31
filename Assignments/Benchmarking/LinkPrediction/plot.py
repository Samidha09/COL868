import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

xls = pd.ExcelFile('./results/Stats/plot.xlsx')
df1 = pd.read_excel(xls, 2)
df2 = pd.read_excel(xls, 3)
df3 = pd.read_excel(xls, 4)
df4 = pd.read_excel(xls, 5)

# plots for time vs quality - Experiment 2
# Graph_Sage results
sage_time_taken = np.array(df1['Average_train_time_per_fold(in sec)'])
sage_time_f1 = np.array(df1['F1-Score'])
sage_time_prec = np.array(df1['Precision'])
sage_time_recall = np.array(df1['Recall'])
sage_time_roc_auc = np.array(df1['ROC-AUC Score'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(sage_time_taken, sage_time_f1, color='g',
         linewidth=5, label="F1_score")
plt.plot(sage_time_taken, sage_time_prec, color='r',
         linewidth=5, label="Precision")
plt.plot(sage_time_taken, sage_time_recall, color='orange',
         linewidth=5, label="Recall")
plt.plot(sage_time_taken, sage_time_roc_auc, color='blue',
         linewidth=5, label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_fold(in sec)', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.title(
    'GraphSAGE: Average_train_time_per_fold(in sec) vs performance quality', fontsize=18)
plt.show()

directory_name = "./images/"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
fig.savefig(directory_name + "GraphSAGE" + ".png")

# GCN results
gcn_time_taken = np.array(df2['Average_train_time_per_fold(in sec)'])
gcn_time_f1 = np.array(df2['F1-Score'])
gcn_time_prec = np.array(df2['Precision'])
gcn_time_recall = np.array(df2['Recall'])
gcn_time_roc_auc = np.array(df2['ROC-AUC Score'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_time_taken, gcn_time_f1, color='g',
         linewidth=5, label="F1_score")
plt.plot(gcn_time_taken, gcn_time_prec, color='r',
         linewidth=5, label="Precision")
plt.plot(gcn_time_taken, gcn_time_recall, color='orange',
         linewidth=5, label="Recall")
plt.plot(gcn_time_taken, gcn_time_roc_auc, color='blue',
         linewidth=5, label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_fold(in sec)', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.title(
    'GCN: Average_train_time_per_fold(in sec) vs performance quality', fontsize=18)
plt.show()

fig.savefig(directory_name + "GCN" + ".png")

# plots for embedding_dimension vs quality - Experiment 2
sage_dim = np.array(df3['embedding_dim'])
sage_f1 = np.array(df3['F1-Score'])
sage_prec = np.array(df3['Precision'])
sage_recall = np.array(df3['Recall'])
sage_roc_auc = np.array(df3['ROC-AUC Score'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(sage_dim, sage_f1, color='g',
         linewidth=5, label="F1_score")
plt.plot(sage_dim, sage_prec, color='r',
         linewidth=5, label="Precision")
plt.plot(sage_dim, sage_recall, color='orange',
         linewidth=5, label="Recall")
plt.plot(sage_dim, sage_roc_auc, color='blue',
         linewidth=5, label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Embedding dimensions', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.title(
    'GraphSAGE: Embedding dimensions vs performance quality', fontsize=18)
plt.show()

fig.savefig(directory_name + "GraphSAGE_dim" + ".png")

# GCN results
gcn_dim = np.array(df4['embedding_dim'])
gcn_f1 = np.array(df4['F1-Score'])
gcn_prec = np.array(df4['Precision'])
gcn_recall = np.array(df4['Recall'])
gcn_roc_auc = np.array(df4['ROC-AUC Score'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_dim, gcn_f1, color='g',
         linewidth=5, label="F1_score")
plt.plot(gcn_dim, gcn_prec, color='r',
         linewidth=5, label="Precision")
plt.plot(gcn_dim, gcn_recall, color='orange',
         linewidth=5, label="Recall")
plt.plot(gcn_dim, gcn_roc_auc, color='blue',
         linewidth=5, label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Embedding dimensions', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.title(
    'GCN: Embedding dimensions vs performance quality', fontsize=18)
plt.show()

fig.savefig(directory_name + "GCN_dim" + ".png")
