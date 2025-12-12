import numpy as np

def rmmissing(data):
    """
    Supprime les valeurs NaN ou None d'une liste ou d'un tableau numpy.
    
    Paramètres :
    ------------
    data : list ou np.ndarray
        Données numériques ou mixtes
    
    Retour :
    --------
    clean_data : np.ndarray
        Données sans NaN / None
    mask : np.ndarray (bool)
        Masque indiquant les valeurs conservées
    """
    data = np.array(data, dtype=float)  # conversion en float pour gérer NaN
    mask = ~np.isnan(data)              # True si valeur non NaN
    return data[mask], mask
