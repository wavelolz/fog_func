import pandas as pd
import numpy as np

##--------
## 這是一個計算模型準確度的程式，輸入為一個dataframe
## 範例輸入:假設有3個模型
##      true_negative   false_positive   
## 1
## 2
## 3



##--------
def judge(df):

  result = []
  for i in range(len(df)):
    tn, fp, fn, tp = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]
    sen = round(tp / (tp + fn), 4)
    spec = round(tn / (fp + tn), 4)
    accu = round((tp + tn) / sum([tp, tn, fn, fp]), 4)
    pre = round(tp / (tp + fp), 4)
    f1 = round((2 * sen * pre) / (sen + pre), 4)
    result.append([sen, spec, accu, pre, f1])
  result = pd.DataFrame(result)

  result.columns = ["Sensitivity", "Specificity", "Accuracy", "Precision", "F1-score"]
  result.index = df.index
  return(result)
