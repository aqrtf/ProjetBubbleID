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
chipMarker = ["o", "^", "s"]
colors_map = {50:'r', 60:'g', 75:'b', 85:'c', 100:'m'}
color_label = ['50V', '60V', '75V', '85V', '100V']
saveFolder = "results"
def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='none', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to Axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='none', ecolor=facecolor)

    return artists



plt.figure()
for ic, c in enumerate(chip):
    df = pd.read_csv(os.path.join(path, c+"_out\mainProperties.csv"))
    df = df.sort_values(by="tension")
    for it, row in df.iterrows():
        plt.scatter(row["departDiameter"], row["frequency"], color=colors_map[row['tension']], marker=chipMarker[ic])
    # plt.plot(df["departDiameter"], df["frequency"], ".-", )

# creation de la legende
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=8, label=lab)
    for c, lab in zip(list(colors_map.values()), color_label)
]
marker_handles = [
    plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='black',
               markersize=8, label=lab)
    for m, lab in zip(chipMarker, chip)
]
plt.legend(handles=color_handles + marker_handles)
plt.xlabel("Departure diameter [mm]")
plt.ylabel("Frequency [Hz]")
plt.savefig(os.path.join(saveFolder,"diameterVSfrequeny.png"))
plt.show()

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


# Create figure and Axes
fig, ax = plt.subplots(1)
color = ['r', 'g', 'b']
for i, c in enumerate(chip):
    df = pd.read_csv(os.path.join(path, c+"_out\mainProperties.csv"))
    # Number of data points
    n = len(df)

    # Dummy data
    x = np.array(df["departDiameter"])
    y = np.array(df["frequency"])

    # Dummy errors (above and below)
    xerr = np.array(df["departDiameter_std"])
    yerr = np.array(df["frequency_std"])
    xerr = np.vstack([xerr, xerr])
    yerr = np.vstack([yerr, yerr])

    # Call function to create error boxes
    _ = make_error_boxes(ax, x, y, xerr, yerr, facecolor=color[i])
plt.legend(chip)
plt.xlabel("Departure diameter [mm]")
plt.ylabel("Frequency [Hz]")
plt.savefig(os.path.join(saveFolder,"diameterVSfrequenyError.png"))
plt.show()




plt.figure()

for c in chip:
    # Lecture des données
    df = pd.read_csv(os.path.join(path, c+"_out/mainProperties.csv"))
    
    # Scatter des points
    plt.plot(df["departDiameter"], df["frequency"], ".", label=f"{c} data")
    
    # Fit linéaire
    slope, intercept, r_value, p_value, std_err = linregress(df["departDiameter"], df["frequency"])
    
    # Droite ajustée
    x_fit = np.linspace(df["departDiameter"].min(), df["departDiameter"].max(), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, "-", label=f"{c} fit (R²={r_value**2:.2f})")

plt.xlabel("Departure diameter [mm]")
plt.ylabel("Frequency [Hz]")
plt.legend()
plt.savefig(os.path.join(saveFolder,"diameterVSfrequenyInterp.png"))
plt.show()
