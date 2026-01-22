import os
from parentBubble import my_detect_fusion, bulle_croissance_rapide, build_masks_and_index

dataFolder = r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet\ProjetBubbleID\Inputs\T87_out"
extension = "T87_75V1"
score_thres=0.7
OVERLAP_THRESH=0.1
MIN_OVERLAP_SAME=0.7
POST_FUSION_FRAMES=2
N_FRAMES_PREVIOUS_DISAPPEAR=3
N_FRAMES_POST_DISAPPEAR=2
IMAGE_SHAPE=(1024, 1024)
DILATE_ITERS=1
                                    
                                    
contourFile = os.path.join(dataFolder, f"contours_{extension}.json")
richFile = os.path.join(dataFolder, f"rich_{extension}.csv")
outputFileHistoryPath = os.path.join(dataFolder, f"fusionHistory_{extension}.txt")

# Construit l'index des masques par frame
data_by_frame = build_masks_and_index(contourFile, richFile, IMAGE_SHAPE, score_thres, DILATE_ITERS)

with open('NULL', 'w') as NULL:

    # ideal case
    
    fusionDict = my_detect_fusion(data_by_frame,
                                    NULL,
                                    N_FRAMES_PREVIOUS_DISAPPEAR=0,
                                    N_FRAMES_POST_DISAPPEAR=0,
                                    POST_FUSION_FRAMES=POST_FUSION_FRAMES,
                                    OVERLAP_THRESH=OVERLAP_THRESH,
                                    min_overlap_same=MIN_OVERLAP_SAME)
    Nideal = len(fusionDict)
    # modified case
    fusionDict = my_detect_fusion(data_by_frame,
                                NULL,
                                N_FRAMES_PREVIOUS_DISAPPEAR,
                                N_FRAMES_POST_DISAPPEAR,
                                POST_FUSION_FRAMES,
                                OVERLAP_THRESH,
                                MIN_OVERLAP_SAME)
    Nmodif = len(fusionDict)
    # croissance rapide
    fusionDict_cr = bulle_croissance_rapide(data_by_frame, richFile)
    Ncr = len(fusionDict_cr)
print(f"Ideal case: {Nideal} fusions détectées")
print(f"Modified case: {Nmodif} fusions détectées")
print(f"Croissance rapide case: {Ncr} fusions détectées")