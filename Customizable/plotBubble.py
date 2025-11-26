import pandas as pd
import matplotlib.pyplot as plt
import os
path = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\Inputs"
chip = ["T87", "T88", "T89"]


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
                          fmt='none', ecolor='k')

    return artists



plt.figure()
for c in chip:
    df = pd.read_csv(os.path.join(path, c+"_out\mainProperties.csv"))
    plt.plot(df["departDiameter"], 1/df["growingTime"], ".", )
plt.legend(chip)
plt.xlabel("Departure diameter [mm]")
plt.ylabel("Frequency [Hz]")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


# Create figure and Axes
fig, ax = plt.subplots(1)
for c in chip:
    df = pd.read_csv(os.path.join(path, c+"_out\mainProperties.csv"))
    # Number of data points
    n = len(df)

    # Dummy data
    x = np.array(df["departDiameter"])
    y = np.array(1/df["growingTime"])

    # Dummy errors (above and below)
    xerr = np.array(df["departDiameter_std"])
    yerr = np.array(1/df["growingTime_std"])
    xerr = np.vstack([xerr, xerr])
    yerr = np.vstack([yerr, yerr])

    # Call function to create error boxes
    _ = make_error_boxes(ax, x, y, xerr, yerr)
plt.legend(chip)
plt.show()
