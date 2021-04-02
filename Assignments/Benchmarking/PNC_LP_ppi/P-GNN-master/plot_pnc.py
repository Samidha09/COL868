'''
https://matplotlib.org/stable/tutorials/introductory/customizing.html
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


xls = pd.ExcelFile('pnc_lp_Scores.xlsx')
# GCN pnc time results
df0 = pd.read_excel(xls, 0)
# Graph_Sage pnc time results
df1 = pd.read_excel(xls, 1)


plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.marker"] = 'o'
plt.rcParams["axes.titlesize"] = 20
# plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.linewidth"] = 0.6


# plots for time vs quality
# Graph_Sage time results
sage_time_taken = np.array(df1['Time_min'])
sage_time_f1 = np.array(df1['F1'])
sage_time_prec = np.array(df1['Precision'])
sage_time_recall = np.array(df1['Recall'])
sage_time_roc_auc = np.array(df1['AUC'])

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
plt.title('GraphSAGE Proteins')
plt.show()

directory_name = "./images/"
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
fig.savefig(directory_name + "GraphSAGE_pnc_time" + ".png")

# GCN time results
gcn_time_taken = np.array(df0['Time_min'])
gcn_time_f1 = np.array(df0['F1'])
gcn_time_prec = np.array(df0['Precision'])
gcn_time_recall = np.array(df0['Recall'])
gcn_time_roc_auc = np.array(df0['AUC'])

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(gcn_time_taken, gcn_time_f1, color='g', label="F1_score")
plt.plot(gcn_time_taken, gcn_time_prec, color='r', label="Precision")
plt.plot(gcn_time_taken, gcn_time_recall, color='orange', label="Recall")
plt.plot(gcn_time_taken, gcn_time_roc_auc, color='b', label="ROC-AUC")

plt.legend()
leg = ax.legend(prop={"size": 14})
plt.xlabel('Average_train_time_per_run(in min)')
plt.ylabel('Performance')
plt.title('GCN Proteins')
plt.show()

fig.savefig(directory_name + "GCN_pnc_time" + ".png")
