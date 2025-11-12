import os, csv, cv2, numpy as np
import pandas as pd
import csteDef

min_attached_run=2
savefolder=r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\My_output\Test6"   # Define the folder you want the data to save in
extension="Test6" 
score_thres = 0.7

from parentBubble4 import findMerge
fusionDict, changeIDList = findMerge(savefolder, extension, score_thres=0.7, OVERLAP_THRESH=0.1,
                                    MIN_OVERLAP_SAME=0.7, POST_FUSION_FRAMES=2, N_FRAMES_PREVIOUS_DISAPPEAR=3, 
                                    N_FRAMES_POST_DISAPPEAR=2,
                                    IMAGE_SHAPE=(1024, 1024), DILATE_ITERS=1
                                    )




rich_path = os.path.join(savefolder, f"rich_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError("rich_ file not found")
rich_df = pd.read_csv(rich_path)

# Supprimer les lignes dont le score est inférieur à score_thres
df_filter = rich_df[rich_df['score'] >= score_thres]

# S'il reste des duplicata, tieni, per ogni (track_id, frame0), la detection con score max
df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
        .drop_duplicates(["track_id", "frame"], keep="first"))

df_score = df_filter[["track_id", "frame", "score", "class_id"]].copy()


def _replaceChangedID(rich_df, changeIDList):
    if "frame" not in rich_df.columns and "frame0" in rich_df.columns:
        rich_df["frame"] = rich_df["frame0"].astype(int) +1
        
    for frame, new_id, old_id in changeIDList:
        rich_df.loc[(rich_df['frame'] >= frame) & (rich_df['track_id'] == new_id), 'track_id'] = old_id
    return rich_df

df_score = _replaceChangedID(df_score, changeIDList)




