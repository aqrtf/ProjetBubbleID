import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/resultsMean.csv")

chips = df['chip'].unique()
colors = ['r', 'g', 'b']

# 1. frequency vs departDiameter
plt.figure()
for chip, color in zip(chips, colors):
    sub = df[df['chip'] == chip]
    plt.errorbar(sub['departDiameter'], sub['frequency'],
                 xerr=sub['departDiameter_std'], yerr=sub['frequency_std'],
                 fmt='o', capsize=5, label=chip, color=color)
plt.xlabel('Diamètre de départ (mm)')
plt.ylabel('Fréquence (Hz)')
plt.title('Fréquence vs Diamètre de départ')
plt.legend()
plt.grid(True)
plt.savefig("results/frequency_vs_departDiameter2.png")

# 2. frequency vs tension
plt.figure()
for chip, color in zip(chips, colors):
    sub = df[df['chip'] == chip]
    plt.errorbar(sub['tension'], sub['frequency'],
                 yerr=sub['frequency_std'],
                 fmt='o', capsize=5, label=chip, color=color)
plt.xlabel('Tension (V)')
plt.ylabel('Fréquence (Hz)')
plt.title('Fréquence vs Tension')
plt.legend()
plt.grid(True)
plt.savefig("results/frequency_vs_tension2.png")

# 3. departDiameter vs tension
plt.figure()
for chip, color in zip(chips, colors):
    sub = df[df['chip'] == chip]
    plt.plot(sub['tension'], sub['frequency']*sub['departDiameter']**3, 'o-', label=chip, color=color)

plt.xlabel('Tension (V)')
plt.ylabel('fD3 (mm)')
plt.title('Diamètre de départ vs Tension')
plt.legend()
plt.grid(True)

plt.show()

# 3. departDiameter vs tension
plt.figure()
for chip, color in zip(chips, colors):
    sub = df[df['chip'] == chip]
    plt.errorbar(sub['tension'],*sub['departDiameter']^3,
                 yerr=sub['departDiameter_std'],
                 fmt='o', capsize=5, label=chip, color=color)
plt.xlabel('Tension (V)')
plt.ylabel('Diamètre de départ (mm)')
plt.title('Diamètre de départ vs Tension')
plt.legend()
plt.grid(True)
plt.savefig("results/departDiameter_vs_tension2.png")

plt.show()
#############################3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor,
                     edgecolor='none', alpha=0.5):
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)
    ax.add_collection(pc)
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='none', ecolor=facecolor)
    return artists

df = pd.read_csv("results/results2.csv")
chip = ["T87", "T88", "T89"]
chipColor = ["r", "g", "b"]

fig, ax = plt.subplots(1)
for i, c in enumerate(chip):
    sub = df[df['chip'] == c].sort_values(by="tension")
    x = np.array(sub["departDiameter"])
    y = np.array(sub["frequency"])
    xerr = np.array(sub["departDiameter_std"])
    yerr = np.array(sub["frequency_std"])
    xerr = np.vstack([xerr, xerr])
    yerr = np.vstack([yerr, yerr])
    _ = make_error_boxes(ax, x, y, xerr, yerr, facecolor=chipColor[i])
plt.legend(chip)
plt.xlabel("Departure diameter [mm]")
plt.ylabel("Frequency [Hz]")
plt.title("Frequency vs Departure diameter (avec erreurs)")
plt.grid(True)
plt.xlim(.5, 10.5)      # Limite l’axe x de 2 à 6
plt.ylim(10, 475)
plt.show()