import os
import numpy as np

# 每个 clip 编码一次
single = np.load("D:/datasets/RepCount/tokens_single/train951_pose.npz")['arr_0']
print("data1 shape: ", single.shape)

# 多个 clip 组成 chunk，批量编码
batch = np.load("D:/datasets/RepCount/tokens_batch/train951_pose.npz")['arr_0']
print("data2 shape: ", batch.shape)

diff = np.abs(single - batch)
diff_max = np.max(diff)   # 数值上有微小差异， 最大偏差 e-5 量级算正常的结果
diff_avg = np.average(diff)

print("\n")
print("global diff_max: ", diff_max, ", diff_avg: ", diff_avg)
print("\n    each clip:")

for i in range(single.shape[0]):
    b1 = single[i,:]
    b2 = batch[i,:]
    d = np.abs(b1 - b2)
    d_max = np.max(d)   # 数值上有微小差异， 最大偏差 e-5 量级算正常的结果
    d_avg = np.average(d)

    print(f"d_max[{i}]: ", d_max, f", d_avg[{i}]: ", d_avg)