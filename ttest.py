from scipy.stats import ttest_ind
from scipy.stats import ranksums
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
#----參數宣告區
path = "D:/python/data/S05R01_4sec1olp.csv" # 欲讀取的資料
alpha = 0.05 # alpha值
output_path = "D:/python/data/ttest.xlsx" # 輸出資料目的地

# 此為ttest的程式, output為一excel檔，parametric是使用independent ttest母數檢定 
# non_parametric是使用Wilcoxon rank sums test無母數檢定，列出的feature是p-value < alpha的
# feature和p-value。即此feature在freez跟non freeze中有顯著的不同
#----


def(path, alpha, output_path):
    data = pd.read_csv(path)
    freez = np.array(data.loc[data["label"] == 1].iloc[:, :-1])
    nonfreez = np.array(data.loc[data["label"] == 0].iloc[:, :-1])
    result_param = ttest_ind(freez, nonfreez, equal_var = False)[1]
    result_nonparam = []
    for i in range(freez.shape[1]):
        result_nonparam.append(ranksums(freez[:, i], nonfreez[:, i])[1])

    col_names = data.columns[:-1]

    select_param = dict()
    select_nonparam = dict()

    for i in range(len(result_param)):
        if result_param[i] < alpha:
            select_param[col_names[i]] = result_param[i]

    for i in range(len(result_nonparam)):
        if result_nonparam[i] < alpha:
            select_nonparam[col_names[i]] = result_nonparam[i]


    writer = pd.ExcelWriter(output_path)
    pd.DataFrame(data = select_param, index = [0]).T.to_excel(writer, sheet_name = "parametric")
    pd.DataFrame(data = select_nonparam, index = [0]).T.to_excel(writer, sheet_name = "non_parametric")
    writer.close()


