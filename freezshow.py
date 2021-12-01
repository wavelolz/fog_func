import numpy as np
import matplotlib.pyplot as plt
import time

#-----參數宣告區
path = "D:/freezing of gait/dataset_fog_release/dataset/S01R01.txt"
interval = 10 # 每個視窗要包含的秒數
move = 3 # 每次移動要前進的秒數
start = 1000 # 起始的秒數
stay_time = 0.7 # 每個視窗停留的秒數
output_data = [2, 3, 4, 5, 6, 7, 8, 9] # 想要觀看的data的column index

# 這是一個將原始raw data加速度以移動視窗來呈現的程式
# 希望藉由資料視覺化能夠找出在raw data中有一些可以不需要
# 丟進model裡的trash data
# 在視窗中如果為黃底視窗，其label為0
# 白底視窗，其label為1
# 紅底視窗其label為2
 
#-----

data = np.loadtxt(path).transpose(1, 0)
index = np.array([int(start * 64), int(start * 64 + interval * 64)])
x = data[0][index[0] : index[1]] / 1000
y = []
for i in output_data:
    y.append(data[i][index[0] : index[1]])
y = np.array(y) / 1000
label = data[10][index[0] : index[1]]
 
fig, ax = plt.subplots(figsize = [20, 10], nrows= len(output_data))


for i in range(len(output_data) - 1):
    ax[i].tick_params(
        axis = "x",
        bottom = False,
        labelbottom = False
    )


while True:
    for i in range(len(output_data)):
        plotting(i)
    plt.pause(stay_time)
    for i in range(len(output_data)):
        ax[i].cla()
    index += int(move * 64)
    x = data[0][index[0] : index[1]] / 1000
    y = data[output_data, index[0] : index[1]] / 1000
    label = data[10][index[0] : index[1]]
plt.show()



