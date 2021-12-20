data = pd.read_csv("D:/data/balanced_cut_all_1sec0.75olp.csv")
data = data[['y_u_max', 'z_u_max', 'x_u_max', 'y_a_fft_totalpower', 'y_u_mean',
       'y_a_std', 'y_a_fft_min', 'y_u_std', 'y_a_fft_max',
       'y_u_fft_totalpower', 'y_a_med', 'z_u_std', 'y_a_min', 'y_u_fft_max',
       'z_u_fft_totalpower', 'y_u_med', 'u_egv1', 'z_u_fft_min',
       'y_t_fft_totalpower', 'u_egv2', 'y_u_min', 'z_t_min', 'y_t_std',
       'z_a_fft_totalpower', 'y_u_fft_min', "label", "id"]]
print("資料裡是否有patient4跟patient10的資料[y/n]??")
I = input()
if I == "y":
    rounding_index = np.arange(1, 11, 1)
elif I == "n":
    rounding_index = [1, 2, 3, 5, 6, 7, 8, 9]


result = []

for i in rounding_index:
    # data selection
    non_freeze = [i, 4, 10]
    data_training = data.drop(data.loc[data["id"].isin(non_freeze)].index)
    data_testing = data.loc[data["id"] == i]

# 僅需修改這部分
#--------------------------------------------------------------------
    # train model                                                    #
    dt = tree.DecisionTreeClassifier()                               #
    dt.fit(data_training.iloc[:, :-2], data_training.iloc[:, -2])    #
                                                                     # 
    # predict                                                        #
    predict = dt.predict(data_testing.iloc[:, :-2])                  #
                                                                     #
#--------------------------------------------------------------------
    result.append(list(confusion_matrix(data_testing.iloc[:, -2], predict).ravel()))


row_name = [f"leave-P{i}-out" for i in rounding_index]
df = pd.DataFrame(result)
df.index = row_name

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


table = judge(df)
table.loc[table["Precision"] == 0, "Precision"] = np.NaN
avg_table = table.apply(lambda x: round(np.nanmean(x), 4), axis = 0)
avg_table.name = "Average without NaN"
table = table.append(avg_table)
print(table)
