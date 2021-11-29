

import numpy as np
import matplotlib.pyplot as plt
import time
data = np.loadtxt("D:/freezing of gait/dataset_fog_release/dataset/S01R01.txt").transpose(1, 0)
interval = 10
move = 3
start = 1700
stay_time = 0.7
output_data = [2, 4, 6, 7, 8, 9]


index = np.array([int(start * 64), int(start * 64 + interval * 64)])
x = data[0][index[0] : index[1]] / 1000
y = []
for i in output_data:
    y.append(data[i][index[0] : index[1]])
y = np.array(y) / 1000
label = data[10][index[0] : index[1]]
 
fig, ax = plt.subplots(figsize = [20, 10], nrows= len(output_data))

def plotting(i):
    ax[i].plot(x, y[i])
    ax[i].fill_between(x, y1 = min(min(y[i]), -2), y2 = max(max(y[i]), 2), where = (label == 0), alpha = 0.3, color = "yellow")
    ax[i].fill_between(x, y1 = min(min(y[i]), -2), y2 = max(max(y[i]), 2), where = (label == 2), alpha = 0.3, color = "red")
    ax[i].set_ylim([min(min(y[i]), -2), max(max(y[i]), 2)])

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



