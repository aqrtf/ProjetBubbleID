a = [1,2,3,4,5]
b = [1,2,3,5,9]
c = []
for x in a:
    if x not in b:
        c.append(x)

print(list(range(4-2,4+2)))

import json
dataFolder = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\My_output\SaveData3"
extension = "T113_2_40V_2"
contourFile = dataFolder + "/contours_" + extension +".json"  # Fichier des contours
with open(contourFile, 'r') as f:
    data = json.load(f)  # Charge tout le fichier JSON
print(data["49_2"])
print(data["49_3"])