import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

path = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\Inputs"
chip = ["T87",
        "T88",
        "T89"
        ]
chipColor = ["r", "lime", "b"]
markers_map = {50:'o', 60:'P', 75:'^', 85:'s', 100:'*'}
marker_label = ['50V', '60V', '75V', '85V', '100V']
saveFolder = "results"

plt.figure()
for ic, c in enumerate(chip):
    df = pd.read_csv(os.path.join(path, c+"_out\mainProperties.csv"))
    df = df.sort_values(by="tension")
    for it, row in df.iterrows():
        plt.scatter(row["tension"], row["frequency"], marker='s', color=chipColor[ic], facecolors='none')
        plt.scatter(row["tension"], row["frequency2"], marker='^', color=chipColor[ic], facecolors='none')
        # plt.errorbar(row["tension"], row["frequency"], yerr=row["frequency_std"], ecolor=chipColor[ic], fmt='none', capsize=5)
    # plt.plot(df["departDiameter"], df["frequency"], ".-", )

# creation de la legende
marker_handles = [
    plt.Line2D([0], [0], marker=m, color='black', markerfacecolor='none',
               markersize=5, label=lab, linestyle='None')
    for m, lab in zip(['s', '^'], ["Growing time", "Detachment"])
]
color_handles = [
    plt.Line2D([0], [0], marker='.', color=c, markerfacecolor='none',
               markersize=8, label=lab)
    for c, lab in zip(chipColor, chip)
]
plt.legend(handles=color_handles+ marker_handles, ncol = 2)
plt.xlabel("Voltage [V]")
plt.ylabel("Mean frequency [Hz]")
plt.savefig(os.path.join(saveFolder,"tensionVSfrequeny.png"))
plt.show()

