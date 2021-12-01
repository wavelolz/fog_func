import pandas as pd
import numpy as np

##--------



##--------
def judge(df):

  result = []
  for i in range(len(df)):
    tn, fp, fn, tp = df[i, 0], df[i, 1], df[i, 2], df[i, 3]
    sen = round(tp / (tp + fn), 4)
    spec = round(tn / (fp + tn), 4)
    accu = round((tp + tn) / sum([tp, tn, fn, fp]), 4)
    pre = round(tp / (tp + fp), 4)
    f1 = round((2 * sen * pre) / (sen + pre), 4)
    result.append([sen, spec, accu, pre, f1])
  result = pd.DataFrame(result)

  result.columns = ["Sensitivity", "Specificity", "Accuracy", "Precision", "F1-score"]
  result.index = ["Model_" + str((i + 1)) for i in range(len(df))]
  return(result)
