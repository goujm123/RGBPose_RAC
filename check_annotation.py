import os
import numpy as np

# 检查标注文件中存在的标注问题
anno_dir = "d:/datasets/RepCount/annotation"

# 原始标注
train_csv = os.path.join(anno_dir, "train.csv")
valid_csv = os.path.join(anno_dir, "valid.csv")
test_csv = os.path.join(anno_dir, "test.csv")

splits = [train_csv, valid_csv, test_csv]

durations = []

for split in splits:
    with open(split, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:  # 跳过表头行
                continue

            # id, type, video_name, count, start, end, start, end...
            columns = line.split(',')
            id = columns[0]
            type = columns[1]
            name = columns[2]
            if columns[3] is not None and len(columns[3]) > 0:
                count = int(columns[3])
            else:
                count = 0
                print(f"{split}, id: {id}, video: {name} count is missed")

            cycles = []
            for j in range(len(columns))[4:]:
                if columns[j] is not None and len(columns[j]) > 0:
                    index = int(columns[j])
                    cycles.append(index)
                else:
                    break
            start = cycles[0::2]
            stop = cycles[1::2]

            if len(start) != len(stop) or len(start) != count:
                print(f"{split}, id: {id}, video: {name} count error: {count}, {len(start)}, {len(stop)}")

            duration = np.array(stop) - np.array(start)
            durations.extend(duration.tolist())


total = np.array(durations)
avg = np.mean(total)
print("average action cycle length (frames): ", avg)
