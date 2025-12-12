import os, csv, cv2, re,  numpy as np
import pandas as pd

from csteDef import *
from functions.readRichFile import readRichFile
from functions.rmmissing import rmmissing
from functions.computeDiameter import bubbleArea, bubble_exists
# TODO separer les chemins si la bulle a une taille qui diminue apres un saut de tracking (valable pour les bulles attachees et/ou petites)
maxJumpFrame = 2
areaMax_px = 3000
# savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
# extension="Test6" 
# savefolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
# extension="T113_2_60V_2" 

def flatten_tuple(node):
    """
    Aplatit la structure imbriquée de tuples en une liste de dictionnaires.
    
    Args:
    - node: La structure imbriquée (tuple ou dict).
    
    Returns:
    - list: Liste plate des dictionnaires.
    """
    if isinstance(node, tuple):
        # Si c'est un tuple, prendre le premier élément (dict) et aplatir le reste
        return [node[0]] + flatten_tuple(node[1])
    else:
        # Si c'est un dict (cas de base), retourner une liste avec ce dict
        return [node] 
    
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
    
def analyzeTidEvolution(rich_df, df_score, df_fusion, changeID_df, nombre_frame, track_id, firstFrametoAnalyse=1):
    results = []
    nameBubble = str(track_id)
    last_seen_frame = None
    
    # Get all data for this track_id sorted by frame
    track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
    track_data = track_data[track_data['frame']>= firstFrametoAnalyse]

    # Initialize evolution tracking array
    evolution_tid = [None] * nombre_frame
    mergeLocation = [] # frame start 1 here
    first_seen_frame = track_data["frame"].min()
    last_seen_frame = track_data["frame"].max()
    evolution_tid[first_seen_frame-1] = track_id  # frame between 1 and nombre_frame
    score = 0
    missing_frame = 0
    last_frame_valid = -1
    
    # Track evolution through frames
    for idx_frame in range(first_seen_frame, nombre_frame+1):

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

        # Detection des cas d'absorption de petites bulles non detectees qui font conserver le meme tid pour la nouvelle bulle formee
        if bubble_exists(idx_frame, track_id, rich_df):
            if last_frame_valid != -1:
                if idx_frame - last_frame_valid >= maxJumpFrame:
                    area = bubbleArea(idx_frame-1, track_id, rich_df)
                    if area < areaMax_px: 
                        # on veut un max_age plus court pour les petites bulles => on reset
                        # TODO on perd les premieres frames de la bulle qui va etre absorbee
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
                        results.append({
                            "bubble_id": nameBubble,
                            "first_seen_frame": first_seen_frame,
                            "last_seen_frame": last_frame_valid, 
                            "n_frames_tracked": n_frames_tracked,
                            "missing_detection": missing_frame,
                            "mean_score_pct": mean_score,
                            "chemin": evolution_tid,
                            "mergeFrame": mergeLocation,
                        })
                        nameBubble = str(track_id)
                        last_seen_frame = None
                        
                        # Get all data for this track_id sorted by frame
                        track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
                        track_data = track_data[track_data['frame']>= firstFrametoAnalyse]

                        # Initialize evolution tracking array
                        evolution_tid = [None] * nombre_frame
                        mergeLocation = [] # frame start 1 here
                        first_seen_frame = idx_frame
                        last_seen_frame = track_data["frame"].max()
                        evolution_tid[first_seen_frame-1] = track_id  # frame between 1 and nombre_frame
                        score = 0
                        missing_frame = 0
                        last_frame_valid = -1
            last_frame_valid = idx_frame

        # Get score for current frame and track_id
        subset = df_score[(df_score["frame"] == idx_frame) & (df_score["track_id"] == track_id)]
        
        # Validate and process detection
        if subset.empty:
            missing_frame += 1  
        elif len(subset) == 1:
            score += subset["score"].iloc[0]
            evolution_tid[idx_frame-1] = track_id
        else:
            raise ValueError("Multiple values found")
        
            
        # end of the loop

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

    return results

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
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    df_fusion = pd.read_csv(path)

    path = os.path.join(savefolder, f"changeIDResultAll_{extension}.csv")
    if not os.path.isfile(path): 
        raise FileNotFoundError("rich_ file not found")
    changeID_df = pd.read_csv(path)


    # Extract relevant columns for scoring
    df_score = rich_df[["track_id", "frame", "score", "class_id"]].copy()

    # Parameters
    nombre_frame = df_score['frame'].max()
    results = []

    # Process each unique track_id
    for track_id in sorted(df_score['track_id'].unique()): 

        res = analyzeTidEvolution(rich_df, df_score, df_fusion, changeID_df, nombre_frame, track_id)  
        # Store results for this bubble evolution
        results.extend(res)
            
    # Convert results to DataFrame with proper data types
    results = pd.DataFrame(results).astype({
        "first_seen_frame": "Int16",
        "last_seen_frame": "Int16",
    })


    def clean_bubble_ids(df: pd.DataFrame, group_col="last_seen_frame", id_col="bubble_id") -> pd.DataFrame:
        """
        Remove rows where a bubble_id is a suffix of another bubble_id within the same group.

        Args:
            df (pd.DataFrame): Input DataFrame.
            group_col (str): Column name for grouping (e.g., last_seen_frame).
            id_col (str): Column name for the bubble ID.

        Returns:
            pd.DataFrame: Filtered DataFrame with redundant bubble_ids removed.
        """

        reduced_df = df[[group_col, id_col]].copy()
        reduced_df = reduced_df.drop_duplicates()
        df = df.loc[reduced_df.index].copy()
        def filter_group(group: pd.DataFrame) -> pd.DataFrame:
            """Filter rows within each group to remove redundant bubble_ids."""
            group = group.copy()
            group = group.sort_values(id_col, key=lambda col: col.str.len())  # Sort by length of bubble_id
            to_keep = []

            for idx, row in group.iterrows():
                bubble_id = row[id_col]
                if not any(bubble_id.endswith(kept) for kept in to_keep):
                    to_keep.append(bubble_id)

            return group[group[id_col].isin(to_keep)]

        return df.groupby(group_col, group_keys=False).apply(filter_group)
    
    results = clean_bubble_ids(results)

    # Extract the first track ID from each evolution chain
    results["first_tid"] = results["bubble_id"].str.extract(r'^(\d+)').astype(int)

    # Save results to CSV
    out_csv = os.path.join(savefolder, f'evolutionID_{extension}.csv')
    results.to_csv(out_csv, index=False)

    print(f"Results saved to: {out_csv}")
    

    
    
evolution_tid(r"Inputs\T87_out", "T87_60V1")