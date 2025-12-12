import numpy as np

def rmoutliers(data, method="quartiles", threshold=1.5):
    """
    Supprime les valeurs aberrantes d'un tableau numpy ou d'une liste.
    
    Paramètres :
    ------------
    data : list ou np.ndarray
        Données numériques
    method : str
        Méthode de détection ('quartiles', 'mean', 'median')
    threshold : float
        Seuil utilisé pour définir un outlier (par défaut 1.5 pour IQR, 3 pour z-score)
    
    Retour :
    --------
    filtered_data : np.ndarray
        Données sans outliers
    mask : np.ndarray (bool)
        Masque indiquant les valeurs conservées
    """
    data = np.array(data)
    
    if method == "quartiles":  # IQR
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method == "mean":  # basé sur moyenne et écart-type
        mean = np.mean(data)
        std = np.std(data)
        mask = np.abs(data - mean) <= threshold * std
    
    elif method == "median":  # basé sur médiane et MAD
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        mask = np.abs(data - median) <= threshold * mad
    
    else:
        raise ValueError("Méthode non reconnue. Choisir 'quartiles', 'mean' ou 'median'.")
    
    return data[mask], mask