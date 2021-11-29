import pandas as pd
import numpy as np

##--------
# 這是一個計算模型準確度的程式，輸入依序為
# 1. (actual is nonevent and predict is nonevent)
# 2. (actual is nonevent but predict is event)
# 3. (actual is event but predict is nonevent)
# 4. (actual is event and predict is event)

#      0    1
#   0  67   45
#
#   1  21   76
# 範例數入: 76 67 21 45


##--------
tn, fp, fn, tp = map(int, input().split())
sen = round(tp / (tp + fn), 4)
spec = round(tn / (fp + tn), 4)
accu = round((tp + tn) / sum([tp, tn, fn, fp]), 4)
pre = round(tp / (tp + fp), 4)
f1 = round((2 * sen * pre) / (sen + pre), 4)
arr = pd.DataFrame(np.array([[tn, fp],
                            [fn, tp]]))

arr.columns = ["Predict_is_nonevent", "Predict_is_event"]
arr.index = ["Real_is_nonevent", "Real_is_event"]

result = {
    "sensitivity" : sen,
    "specificity" : spec,
    "accuracy" : accu,
    "precision" : pre,
    "F1-score" : f1
}
print("---------------------")
print(result)
print("---------------------")
print("Confusion matrix:")
print(arr)
