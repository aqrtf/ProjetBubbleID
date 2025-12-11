import numpy as np 

def bubbleDiameter(frame0, tid, rich_df):
    """
    Calcule le diamètre de la bulle tid a la frame0 frame0 en prenant la moyenne entre le
    diametre evalue par l'aire et par le perimetre (mix method).
    
    Arguments:
        frame0 (int): Numéro de la frame (commence à 0)
        tid (int): Identifiant de la bulle (track_id)
        rich_df (pandas.DataFrame): DataFrame contenant les propriétés des bulles (doit contenir area_px et perim_px et etre filtre)
        
    Returns:
        diameter_px (float): Diamètre de la bulle en pixels
    """
    bubble_data = rich_df.loc[(rich_df["frame0"] == frame0) & (rich_df["track_id"] == tid)]
    area_px = bubble_data["area_px"].values[0]
    perim_px = bubble_data["perim_px"].values[0]
    diameter_px = (2 * np.sqrt(area_px / np.pi) + perim_px / np.pi) / 2
    return diameter_px