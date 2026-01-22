from csteDef import *
import os
import pandas as pd

def correctionLabel(rich_df, y_limit_attached = 820):
    """Corrige les labels des bulles attachées en fonction de leur position verticale.
    Si une bulle attachée est au-dessus d'une certaine limite (y_limit_attached), on change son label à UNKNOWN.

    Args:
        rich_df (pd.Dataframe): rich dataframe
        y_limit_attached (int, optional): Limite verticale pour considerer une bulle attached. Defaults to 820.

    Returns:
        rich_df: Dataframe corrigé des bulles avec les labels mis à jour.
    """
    
    
    for ligne in rich_df.itertuples():
        bottom = ligne.y2
        label = ligne.class_id
        if label==ATTACHED and bottom < y_limit_attached:
            # Si une bulle est attache mais que le bas de celle ci est superieure a y_limit_attached on change son label a unknown
            # En effet on se trouve trop loin du sol
            # !!!! l'origine se trouve en haut a gauche de l'image donc < y_limit_attached signifie plus haut dans l'image
            rich_df.at[ligne.Index, 'class_id'] = UNKNOWN  # Changer le label à UNKNOWN
            
        return rich_df
