import os, csv, cv2, re, numpy as np
import pandas as pd
from csteDef import *

def analyze_dwell_time(savefolder, extension, score_thres = 0.7, n_frames_post_disappear=2, fps = 4000):
    """
    Analyze bubble dwell time and detachment events from tracking data.
    
    This function processes bubble tracking data to calculate attachment periods,
    detect detachment events, and generate evolution chains for bubbles.
    
    Args:
        savefolder (str): Path to the folder containing input CSV files
        extension (str): File extension identifier for input/output files
        score_thres (float): Minimum score threshold for filtering detections
        min_attached_run (int): Minimum number of frames for attached state (currently unused)
        n_frames_post_disappear (int): Number of frames to check for merges after disappearance
    
    Returns:
        None: Results are saved to {savefolder}/dwell6_{extension}.csv
    
    Raises:
        FileNotFoundError: If required input files are not found
    """
    
    # Load input data files
    rich_path = os.path.join(savefolder, f"rich_{extension}.csv")
    if not os.path.isfile(rich_path):
        raise FileNotFoundError("rich_ file not found")
    rich_df = pd.read_csv(rich_path)

    path = os.path.join(savefolder, f"fusionResult_{extension}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    df_fusion = pd.read_csv(path)

    path = os.path.join(savefolder, f"changeIDResultAll_{extension}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError("changeIDResultAll_ file not found")
    changeID_df = pd.read_csv(path)

    # Filter rows with score above threshold and valid track_id
    df_filter = rich_df[rich_df['score'] >= score_thres]
    df_filter = df_filter[df_filter["track_id"].fillna(-1).astype(int) >= 0]

    # Remove duplicates: for each (track_id, frame), keep the detection with highest score
    df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
            .drop_duplicates(["track_id", "frame"], keep="first"))

    df_score = df_filter[["track_id", "frame", "score", "class_id"]].copy()

    # Parameters
    last_frame = df_score['frame'].max()
    results = []

    # Process each unique track_id
    for track_id in sorted(df_score['track_id'].unique()):
        # Initialize tracking variables
        note = ''
        detach_frame = None
        attach_start_frame = None
        isDetached = False
        nameBubble = str(track_id)
        missing_frame = 0
        last_attach_frame = None
        
        track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
        
        # Calculate basic statistics
        n_frames_tracked = len(track_data)
        n_unknown = len(track_data[track_data['class_id'] == UNKNOWN])

        # Skip if bubble is never attached
        if ATTACHED not in track_data["class_id"].values:
            note = "no_attached_run"
        else:
            attach_start_frame = track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0]
            last_seen_frame = track_data["frame"].iloc[-1]
            last_attach_frame = attach_start_frame
            
            idx_frame = attach_start_frame
            score = df_score[(df_score["frame"] == idx_frame) & (df_score["track_id"] == track_id)]["score"].iloc[0]
            
            # Track bubble evolution through frames
            while True:
                idx_frame += 1
                
                # Check for fusion events
                mask = (df_fusion["frame"] == idx_frame) & ((df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id))
                if mask.any():
                    track_id = df_fusion.loc[mask, "child"].iat[0]
                    nameBubble += "=>" + str(track_id)
                    last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()

                # Check for ID change events
                mask = (changeID_df["frame"] == idx_frame) & (changeID_df["old_id"] == track_id)
                if mask.any():
                    track_id = changeID_df.loc[mask, "new_id"].iat[0]
                    nameBubble += "<->" + str(track_id)
                    last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()

                # Get detection data for current frame
                subset = df_score[(df_score["frame"] == idx_frame) & (df_score["track_id"] == track_id)]
                
                if subset.empty:
                    missing_frame += 1
                elif len(subset) == 1:
                    score += subset["score"].iloc[0]
                    if subset["class_id"].iloc[0] == DETACHED:
                        detach_frame = idx_frame
                        note = "DETACH"
                        isDetached = True
                        # TODO que se passe t'il si elle est detach apres avoir disparue du track apres un certain temps
                        # TODO renforcer la detection
                        break
                    elif subset["class_id"].iloc[0] == ATTACHED:
                        last_attach_frame = idx_frame
                else:
                    raise ValueError("Multiple values found")

                # Break conditions
                if idx_frame == last_frame:
                    note = "attach_until_end"
                    break
                if idx_frame > last_seen_frame:
                    # Check for merges after disappearance
                    for idx_supp in range(last_seen_frame, last_seen_frame + 1 + n_frames_post_disappear):
                        mask = (df_fusion["frame"] == idx_supp) & ((df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id))
                        if mask.any():
                            track_id = df_fusion.loc[mask, "child"].iat[0]
                            nameBubble += "=>" + str(track_id)
                            last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()
                    
                    if idx_frame > last_seen_frame:
                        if attach_start_frame == last_frame:
                            note = "attach_until_end"
                            break
                        note = f"disappear after frame {last_seen_frame}"
                        break
   
        # Calculate dwell time statistics
        if attach_start_frame is not None and detach_frame is not None:
            dwell_frames = detach_frame - attach_start_frame
        else:
            dwell_frames = np.nan
      
        # Calculate mean score
        mean_score = None
        if attach_start_frame is not None:
            end_frame = detach_frame if detach_frame is not None else last_frame
            n = end_frame - attach_start_frame + 1
            mean_score = score / n if n > 0 else 0
        
        # Store results for this bubble
        results.append({
            "bubble_id": nameBubble,
            "attach_start_frame": attach_start_frame,
            "last_attach_frame": last_attach_frame,
            "detach_frame": detach_frame,
            "dwell_frames": dwell_frames,
            "dwell_seconds": dwell_frames / fps,
            "n_frames_tracked": n_frames_tracked,
            "n_unknown": n_unknown,
            "missing_detection": missing_frame,
            "mean_score_pct": mean_score,
            "detachs?": isDetached,
            "note": note,
        })
        
    # Convert results to DataFrame with proper data types
    results = pd.DataFrame(results).astype({
        "attach_start_frame": "Int64",
        "last_attach_frame": "Int64",
        "detach_frame": "Int64",
        "dwell_frames": "Int64",
    })

    # Remove duplicate evolution chains (where one chain is a subset of another)
    def parse_tokens(series: pd.Series) -> pd.Series:
        """Parse bubble_id strings into lists of integers using regex splitting."""
        return series.astype(str).apply(lambda s: [int(tok) for tok in re.split(r'<->|=>', s)])

    def clean_bubble_ids(df: pd.DataFrame, group_col="detach_frame", id_col="bubble_id") -> pd.DataFrame:
        """
        Remove evolution chains that are subsets of longer chains.
        
        For bubbles with same detachment frame, keep only the longest unique evolution chains.
        """
        df = df.copy()
        df["_tokens"] = parse_tokens(df[id_col])
        df["_len"] = df["_tokens"].apply(len)

        def filter_group(group: pd.DataFrame) -> pd.DataFrame:
            """Filter within each group to keep only non-redundant evolution chains."""
            group = group.sort_values("_len", ascending=False)
            keep = []
            mask = []
            for tok in group["_tokens"]:
                if any(kt[-len(tok):] == tok for kt in keep):
                    mask.append(False)
                else:
                    keep.append(tok)
                    mask.append(True)
            return group[mask]

        try:
            result = df.groupby(group_col, dropna=False, group_keys=False).apply(filter_group)
        except TypeError:
            # Fallback for pandas versions that don't support dropna=False
            sentinel = "__MISSING__"
            df[group_col] = df[group_col].astype(object).where(df[group_col].notna(), sentinel)
            result = df.groupby(group_col, group_keys=False).apply(filter_group)
            result[group_col] = result[group_col].replace(sentinel, pd.NA)

        return result.drop(columns=["_tokens", "_len"])

    results = clean_bubble_ids(results)

    # Save results to CSV
    out_csv = os.path.join(savefolder, f'dwell6_{extension}.csv')
    results.to_csv(out_csv, index=False)

    print(f"Results saved to: {out_csv}")

# #########################################################
# savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
# extension="Test6" 
# # savefolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
# # extension="T113_2_60V_2" 

# analyze_dwell_time(savefolder, extension)