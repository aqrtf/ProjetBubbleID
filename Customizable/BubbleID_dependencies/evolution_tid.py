import os, csv, cv2, re,  numpy as np
import pandas as pd
from csteDef import *

# savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
# extension="Test6" 
# savefolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
# extension="T113_2_60V_2" 

def evolution_tid(savefolder, extension, score_thres=0.7):
    """
    Analyze bubble evolution and tracking data to generate evolution trajectories.
    
    This function processes tracking data to create evolution chains of bubbles,
    handling merges and ID changes, and filtering by score threshold.
    
    Args:
        savefolder (str): Path to the folder containing input CSV files
        extension (str): File extension identifier for input/output files
        score_thres (float): Minimum score threshold for filtering detections
        
    Returns:
        None: Results are saved to {savefolder}/evolutionID_{extension}.csv
    
    Raises:
        FileNotFoundError: If required input files are not found
    """
    
    # Load input data files
    rich_path = os.path.join(savefolder, f"rich_{extension}.csv")
    if not os.path.isfile(rich_path):
        raise FileNotFoundError("rich_ file not found")
    rich_df = pd.read_csv(rich_path)

    path = os.path.join(savefolder, f"fusionResult_{extension}.csv")
    if not os.path.isfile(rich_path):  # Note: This should probably check path instead of rich_path
        raise FileNotFoundError(f"{path} not found")
    df_fusion = pd.read_csv(path)

    path = os.path.join(savefolder, f"changeIDResultAll_{extension}.csv")
    if not os.path.isfile(rich_path):  # Note: This should probably check path instead of rich_path
        raise FileNotFoundError("rich_ file not found")
    changeID_df = pd.read_csv(path)

    # Filter rows with score above threshold and valid track_id
    df_filter = rich_df[rich_df['score'] >= score_thres]
    # Remove rows with tracking problems (negative or NaN track_id)
    df_filter = df_filter[df_filter["track_id"].fillna(-1).astype(int) >= 0]

    # Remove duplicates: for each (track_id, frame), keep the detection with highest score
    df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
            .drop_duplicates(["track_id", "frame"], keep="first"))

    # Extract relevant columns for scoring
    df_score = df_filter[["track_id", "frame", "score", "class_id"]].copy()

    # Parameters
    last_frame = df_score['frame'].max()
    results = []

    # Process each unique track_id
    for track_id in sorted(df_score['track_id'].unique()):

        nameBubble = str(track_id)
        last_seen_frame = None
        
        # Get all data for this track_id sorted by frame
        track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
        
        # Initialize evolution tracking array
        evolution_tid = [None] * last_frame
        first_seen_frame = track_data["frame"].min()
        last_seen_frame = track_data["frame"].max()
        evolution_tid[first_seen_frame-1] = track_id  # frame between 1 and last_frame
        score = 0
        missing_frame = 0
        
        # Track evolution through frames
        for idx_frame in range(first_seen_frame, last_frame+1):
            # Check if bubble merges with another at this frame
            mask = (df_fusion["frame"] == idx_frame) & ((df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id))
            if (mask).any():
                # Bubble merges - update to child track_id
                track_id = df_fusion.loc[mask, "child"].iat[0]
                nameBubble += "=>" + str(track_id)
                last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()

            # Check if bubble changes ID at this frame
            mask = (changeID_df["frame"] == idx_frame) & ((changeID_df["old_id"] == track_id))
            if (mask).any():
                # Bubble changes ID - update to new track_id
                track_id = changeID_df.loc[mask, "new_id"].iat[0]
                nameBubble += "<->" + str(track_id)
                last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()

            # Get score for current frame and track_id
            subset = df_score[(df_score["frame"] == idx_frame) & (df_score["track_id"] == track_id)]
            
            # Validate and process detection
            if subset.empty:
                missing_frame += 1  # Note: missing_frame needs to be initialized
            elif len(subset) == 1:
                score += subset["score"].iloc[0]
                evolution_tid[idx_frame-1] = track_id
            else:
                raise ValueError("Multiple values found")
    
        # Calculate tracking statistics
        not_none_idx = [i for i, x in enumerate(evolution_tid) if x is not None]
        if not not_none_idx:
            n_frames_tracked = missing_frame = -1  # no valid frames found
        else:
            start, end = not_none_idx[0], not_none_idx[-1]
            sublist = evolution_tid[start:end+1]
            # Count frames where bubble was detected
            n_frames_tracked = sum(x is not None for x in sublist)
            # Count frames where bubble was not detected (gaps in tracking)
            missing_frame = sum(x is None for x in sublist)
        
        # Calculate mean score
        mean_score = score/n_frames_tracked
            
        # Store results for this bubble evolution
        results.append({
            "bubble_id": nameBubble,
            "first_seen_frame": first_seen_frame,
            "last_seen_frame": last_seen_frame,
            "n_frames_tracked": n_frames_tracked,
            "missing_detection": missing_frame,
            "mean_score_pct": mean_score,
            "chemin": evolution_tid
        })
            
    # Convert results to DataFrame with proper data types
    results = pd.DataFrame(results).astype({
        "first_seen_frame": "Int16",
        "last_seen_frame": "Int16",
    })

    # Remove duplicate evolution chains (where one chain is a subset of another)
    def parse_tokens(series: pd.Series) -> pd.Series:
        """Parse bubble_id strings into lists of integers using regex splitting."""
        return series.astype(str).apply(lambda s: [int(tok) for tok in re.split(r'<->|=>', s)])

    def clean_bubble_ids(df: pd.DataFrame, group_col="last_seen_frame", id_col="bubble_id") -> pd.DataFrame:
        """
        Remove evolution chains that are subsets of longer chains.
        
        For bubbles ending at the same frame, keep only the longest unique evolution chains
        and remove chains that are suffixes of longer chains.
        """
        df = df.copy()
        df["_tokens"] = parse_tokens(df[id_col])
        df["_len"] = df["_tokens"].apply(len)

        def filter_group(group: pd.DataFrame) -> pd.DataFrame:
            """Filter within each group to keep only non-redundant evolution chains."""
            # Sort by chain length (longest first)
            group = group.sort_values("_len", ascending=False)
            keep = []
            mask = []
            for tok in group["_tokens"]:
                # Check if current token list is a suffix of any kept chain
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

    # Apply cleaning to remove redundant evolution chains
    results = clean_bubble_ids(results)

    # Extract the first track ID from each evolution chain
    results["first_tid"] = results["bubble_id"].str.extract(r'^(\d+)').astype(int)

    # Save results to CSV
    out_csv = os.path.join(savefolder, f'evolutionID_{extension}.csv')
    results.to_csv(out_csv, index=False)

    print(f"Results saved to: {out_csv}")
    
    
# evolution_tid(savefolder, extension, score_thres)