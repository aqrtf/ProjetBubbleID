def replaceChangedID(rich_df, changeIDList):
    if "frame" not in rich_df.columns and "frame0" in rich_df.columns:
        rich_df["frame"] = rich_df["frame0"].astype(int) +1
        
    for frame, new_id, old_id in changeIDList:
        rich_df.loc[(rich_df['frame'] >= frame) & (rich_df['track_id'] == new_id), 'track_id'] = old_id
    return rich_df
    # df.loc[...] sélectionne les lignes selon une condition.
    # (df['frame'] > x) & (df['tid'] == y) est la condition combinée.
    # 'status' est la colonne que tu modifies (remplace par celle que tu veux).
import parentBubble4  
dataFolder = "My_output/Test6"
extension = "Test6"

fusionDict, changeIDList = parentBubble4.findMerge(dataFolder, extension, score_thres=0.7, OVERLAP_THRESH=0.1,
                                    MIN_OVERLAP_SAME=0.7, POST_FUSION_FRAMES=2, N_FRAMES_PREVIOUS_DISAPPEAR=3, 
                                    N_FRAMES_POST_DISAPPEAR=2,
                                    IMAGE_SHAPE=(1024, 1024), DILATE_ITERS=1
                                    )

richFile = dataFolder + "/rich_" + extension +".csv"  # Fichier de tracking
import pandas as pd
rich_df = pd.read_csv(richFile)
print(rich_df)
print(replaceChangedID(rich_df, changeIDList))
