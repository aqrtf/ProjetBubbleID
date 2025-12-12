import numpy as np 

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