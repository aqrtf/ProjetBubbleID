import os
import pandas as pd
import numpy as np


def readRichFile(path, scoreThresh=0):
    # le fichiers existe-il?
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} non trovato.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns] # on retire les espaces
    # Filter rows with score above threshold and valid track_id
    df_filter = df[df['score'] >= scoreThresh]
    df_filter = df_filter[df_filter["track_id"].fillna(-1).astype(int) >= 0]

    # Remove duplicates: for each (track_id, frame), keep the detection with highest score
    df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
            .drop_duplicates(["track_id", "frame"], keep="first"))
    # Conversion des frames: frame1 → frame0 (indexation à partir de 0)
    df_filter["frame0"] = df_filter["frame"].astype(int) - 1
    return df_filter


def bubbleDiameter(frame0, tid, rich_df):
    """
    Calcule le diamètre de la bulle tid a la frame0 frame0 en prenant la moyenne entre le
    diametre evalue par l'aire et par le perimetre (mix method).
    
    Arguments:
        frame0 (int or list): Numéro de la frame (commence à 0)
        tid (int or list): Identifiant de la bulle (track_id)
        rich_df (pandas.DataFrame): DataFrame contenant les propriétés des bulles (doit contenir area_px et perim_px et etre filtre)
        
    Returns:
        diameter_px (float or list): Diamètre de la bulle en pixels
    """
    if isinstance(frame0, int):
        bubble_data = rich_df.loc[(rich_df["frame0"] == frame0) & (rich_df["track_id"] == tid)]
        area_px = bubble_data["area_px"].values[0]
        perim_px = bubble_data["perim_px"].values[0]
        diameter_px = (2 * np.sqrt(area_px / np.pi) + perim_px / np.pi) / 2
        return diameter_px
    else:
        diameter_px = []
        for fr, t in zip(frame0, tid):
            bubble_data = rich_df.loc[(rich_df["frame0"] == fr) & (rich_df["track_id"] == t)]
            area_px = bubble_data["area_px"].values[0]
            perim_px = bubble_data["perim_px"].values[0]
            diameter_px.append((2 * np.sqrt(area_px / np.pi) + perim_px / np.pi) / 2)
        return diameter_px
    
def bubbleArea(frame0, tid, rich_df):
    """
    Retourne l'aire d'une bulle tid a la frame0 frame0.
    
    Arguments:
        frame0 (int or list): Numéro de la frame (commence à 0)
        tid (int or list): Identifiant de la bulle (track_id)
        rich_df (pandas.DataFrame): DataFrame contenant les propriétés des bulles (doit contenir area_px et perim_px et etre filtre)
        
    Returns:
        area_px (float or list): aire de la bulle en pixels
    """
    if isinstance(frame0, int):
        bubble_data = rich_df.loc[(rich_df["frame0"] == frame0) & (rich_df["track_id"] == tid)]
        area_px = bubble_data["area_px"].values[0]
        return area_px
    else:
        area_px = []
        for fr, t in zip(frame0, tid):
            bubble_data = rich_df.loc[(rich_df["frame0"] == fr) & (rich_df["track_id"] == t)]
            area_px.append(bubble_data["area_px"].values[0])
        return area_px
    
def bubble_exists(frame, tid, rich_df):
    """
    Vérifie si une bulle avec le track_id (tid) existe dans la frame donnée.
    
    Args:
    - df (pd.DataFrame): Le dataframe rich filtré.
    - frame (int): Le numéro de la frame commencant a 1.
    - tid (int): Le track_id de la bulle.
    
    Returns:
    - bool: True si la bulle existe, False sinon.
    """
    # Filtrer les lignes où frame et track_id correspondent
    filtered = rich_df[(rich_df['frame'] == frame) & (rich_df['track_id'] == tid)]
    return not filtered.empty

def extractRichData(rich_df, frame0, tid, column_name):
    """
    Extract the value of a specific column from the rich DataFrame based on frame0 and tid.

    Args:
        rich_df (pd.DataFrame): The DataFrame containing the data.
        frame0 (int or list): The frame number(s) (starting from 0).
        tid (int or list): The track ID(s).
        column_name (str): The name of the column to extract.

    Returns:
        The value(s) from the specified column. A single value if frame0 and tid are integers,
        or a list of values if they are lists.
    """
    if isinstance(frame0, int) and isinstance(tid, int):
        # Single frame and tid
        row = rich_df.loc[(rich_df['frame0'] == frame0) & (rich_df['track_id'] == tid)]
        if row.empty:
            return np.nan
        return row[column_name].iloc[0]

    elif isinstance(frame0, list) and isinstance(tid, list):
        # Multiple frames and tids
        if len(frame0) != len(tid):
            raise ValueError("frame0 and tid lists must have the same length.")
        values = []
        for fr, t in zip(frame0, tid):
            row = rich_df.loc[(rich_df['frame0'] == fr) & (rich_df['track_id'] == t)]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(row[column_name].iloc[0])
        return values

    else:
        raise TypeError("frame0 and tid must both be either integers or lists of the same length.")

