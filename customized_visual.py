import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

# train data per one epoch size: 1094
# valid data per one epoch size: 106
size = [1094, 106]

# first part (step 6):
fp_step = 6
fp = [1458, 34]

# second part (step 100):
sp_step = 100
sp = [1858, 44]


if __name__ == "__main__":
    path = "model"

    with open(f"{path}/results.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    
    _, canvas = plt.subplots(
        ncols=len(data[list(data.keys())[0]].keys()), 
        nrows=len(data.keys()),
        figsize=(56, 12)
    )
    
    for i, group in enumerate(data.keys()):
        for j, metric in enumerate(data[group].keys()):
            values = data[group][metric]

            canvas[i][j].grid(axis='y', linestyle='--', alpha=0.5)

            X = list(range(0, fp[i], (size[i] // fp_step) + 1)) + list(range(fp[i], sp[i], (size[i] // sp_step) + 1)) + [sp[i] - 1]
            for x in X:
                canvas[i][j].axvline(x=x, color="gray")
            canvas[i][j].axvline(x=fp[i] + 0.5, color="red", linestyle="--")
            canvas[i][j].plot(list(range(1, fp[i] + 1)), values[:fp[i]], "c", label="First part")
            canvas[i][j].plot(list(range(fp[i] + 1, len(values) + 1)), values[fp[i]:], "g", label="Second part")
            X = np.array(X)
            values = np.array(values)[X]
            canvas[i][j].plot(X, values, "m-o", label="By epoch")
            canvas[i][j].set_title(f"{group}/{metric}")
            canvas[i][j].legend()
    
    plt.savefig(f"{path}/results.png", format="png", dpi=300)
