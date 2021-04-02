'''
https://matplotlib.org/stable/tutorials/introductory/customizing.html
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


xls = pd.ExcelFile('pnc_lp_Scores.xlsx')
# GCN lp ppi time results
df2 = pd.read_excel(xls, 2)
# Graph_Sage lp ppi time results
df3 = pd.read_excel(xls, 3)
# GCN lp bk time results
df4 = pd.read_excel(xls, 4)
# Graph_Sage lp bk time results
df5 = pd.read_excel(xls, 5)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.marker"] = 'o'
plt.rcParams["axes.titlesize"] = 20
# plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.linewidth"] = 0.6


# plots for time vs quality
# Graph_Sage ppi time results
sage_time_taken = np.array(df3['Time_min'])
sage_time_f1 = np.array(df3['F1'])
sage_time_prec = np.array(df3['Precision'])
sage_time_recall = np.array(df3['Recall'])
sage_time_roc_auc = np.array(df3['AUC'])

# plt.style.use('seaborn-poster')

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(sage_time_taken, sage_time_f1, color='g', label="F1_score")
plt.plot(sage_time_taken, sage_time_prec, color='r', label="Precision")
plt.plot(sage_time_taken, sage_time_recall, color='orange',label="Recall")
plt.plot(sage_time_taken, sage_time_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_run(in min)')
plt.ylabel('Performance')
plt.title('GraphSAGE PPI')
plt.show()

directory_name = "./images/"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
fig.savefig(directory_name + "GraphSAGE_lp_ppi_time" + ".png")

# GCN time results
gcn_time_taken = np.array(df2['Time_min'])
gcn_time_f1 = np.array(df2['F1'])
gcn_time_prec = np.array(df2['Precision'])
gcn_time_recall = np.array(df2['Recall'])
gcn_time_roc_auc = np.array(df2['AUC'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_time_taken, gcn_time_f1, color='g', label="F1_score")
plt.plot(gcn_time_taken, gcn_time_prec, color='r', label="Precision")
plt.plot(gcn_time_taken, gcn_time_recall, color='orange', label="Recall")
plt.plot(gcn_time_taken, gcn_time_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_run(in min)')
plt.ylabel('Performance')
plt.title('GCN PPI')
plt.show()

fig.savefig(directory_name + "GCN_lp_ppi_time" + ".png")

# Graph_Sage time results
sage_time_taken = np.array(df5['Time_min'])
sage_time_roc_auc = np.array(df5['AUC'])

# plt.style.use('seaborn-poster')

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(sage_time_taken, sage_time_roc_auc, color='b')

plt.xlabel('Average_train_time_per_fold(in min)')
plt.ylabel('ROC-AUC')
plt.title('GraphSAGE Brightkite')
plt.show()

directory_name = "./images/"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
fig.savefig(directory_name + "GraphSAGE_lp_bk_time" + ".png")

# GCN time results
gcn_time_taken = np.array(df4['Time_min'])
gcn_time_roc_auc = np.array(df4['AUC'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_time_taken, gcn_time_roc_auc, color='b')

plt.xlabel('Average_train_time_per_fold(in min)')
plt.ylabel('ROC-AUC')
plt.title('GCN Brightkite')
plt.show()

fig.savefig(directory_name + "GCN_lp_bk_time" + ".png")
