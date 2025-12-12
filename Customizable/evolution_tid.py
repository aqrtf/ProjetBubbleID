import os, csv, cv2, re,  numpy as np
import pandas as pd

from csteDef import *
from functions.readRichFile import readRichFile
from functions.rmmissing import rmmissing
from functions.computeDiameter import bubbleArea
# TODO separer les chemins si la bulle a une taille qui diminue apres un saut de tracking (valable pour les bulles attachees et/ou petites)

# savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
# extension="Test6" 
# savefolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
# extension="T113_2_60V_2" 

def separate_bubble_absorb(evolution_tid, rich_df, areaMax_px = 3000):
    """Quand de petites bulles sont absorbees dans une grosse, on ne detecte pas le merge.
    De plus, qq frame apres une nouvelle bulle apparait au meme endroit et on considere que c'est la meme.
    
    Cette fonction essaie de detecter ces cas et de separer ce track en deux distincts.
    Pour ce faire on regarde si l'aire de la bulle a diminuer apres un saut de detection,
    cela n'est valable que pour les petites bulles. La definition de petites bulles est arbitraire."""
    tids, mask = rmmissing(evolution_tid)
    frames0 = np.where(mask)[0]
    areas = bubbleArea(frames0, tids, rich_df)
    jump_indices = np.where(np.diff(frames0, prepend=frames0[0]) > 1)[0] # on rajoute un element au debut pour garder la meme taille
    diameterDecrease_indices = np.where(np.diff(areas, prepend=areas[0]) < 0)[0]
    possible_separations_idx = np.intersect1d(jump_indices, diameterDecrease_indices)
    separations_idx = []
    for idx in possible_separations_idx:
        if areas[idx]< areaMax_px:
            separations_idx.append(idx)

    # TODO il faut reussir a venir modifier le evolution_tid en plusieurs parties a chaque fois 
    # il est peut etre plus simple de faire directement dans evolutiontid directement (pas besoin de separer)
    # dans ce cas il faut revenir sur la meme bulle a l'iteration suivante
    



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
    rich_df = readRichFile(rich_path, score_thres)

    path = os.path.join(savefolder, f"fusionResult_{extension}.csv")
    if not os.path.isfile(rich_path):  # Note: This should probably check path instead of rich_path
        raise FileNotFoundError(f"{path} not found")
    df_fusion = pd.read_csv(path)

    path = os.path.join(savefolder, f"changeIDResultAll_{extension}.csv")
    if not os.path.isfile(rich_path):  # Note: This should probably check path instead of rich_path
        raise FileNotFoundError("rich_ file not found")
    changeID_df = pd.read_csv(path)


    # Extract relevant columns for scoring
    df_score = rich_df[["track_id", "frame", "score", "class_id"]].copy()

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
        mergeLocation = []
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
                mergeLocation.append(idx_frame)

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
            "chemin": evolution_tid,
            "mergeFrame": mergeLocation,
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
    
    
# evolution_tid(r"Inputs\T87_out", "T87_60V1")