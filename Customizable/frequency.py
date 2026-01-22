import os
import ast
import pandas as pd
from csteDef import ATTACHED, DETACHED
from functions.richFileFunctions import readRichFile
from functions.rmmissing import rmmissing

def count_detachment_transitions(savefolder, extension, fps = 4000):
    """
    Compte le nombre de bulles qui passent de l'état 'attached' à l'état 'detached'.
    
    Par sécurité, l'état 'detached' doit apparaître deux fois d'affilée pour 
    considérer qu'il y a détachement.
    
    Args:
        savefolder (str): Dossier contenant les fichiers CSV
        extension (str): Extension pour identifier les fichiers (ex: "Test6")
        
    Returns:
        int: Nombre de bulles ayant effectué une transition attached -> detached
    """
    # Chemins vers les fichiers
    rich_csv = os.path.join(savefolder, f"rich_{extension}.csv")
    evolution_csv = os.path.join(savefolder, f"evolutionID_{extension}.csv")
    
    rich_df = readRichFile(rich_csv)
    
    # Chargement du fichier evolutionID
    df_evol = pd.read_csv(evolution_csv)
    tid_arr = df_evol["chemin"].apply(ast.literal_eval).to_list()
    nb_frame = len(tid_arr[0])
    
    frames_arr = [[j for j, val in enumerate(row) if val is not None] for row in tid_arr]
    bubclass_arr = []
    for irow, row in enumerate(frames_arr):
        x = []
        for fr in row:
            x.append(int(rich_df[(rich_df["frame0"] == fr) &
                                 (rich_df["track_id"] == tid_arr[irow][fr])].iloc[0].at["class_id"]))
        bubclass_arr.append(x)
    
    # Compteur de bulles avec transition attached -> detached
    # count_detachments = 0
    detachBubble = set()
    
    for idx, tid_evol in enumerate(tid_arr):
        tids, _ = rmmissing(tid_evol)
        frames0 = frames_arr[idx]
        labels = bubclass_arr[idx]
        
        # Si labels ne contient qu'un seul type de label, on passe au suivant
        if len(set(labels)) == 1:
            continue
        
        # Vérifier s'il y a une transition attached -> detached (avec detached 2 fois d'affilée)
        for i in range(len(labels) - 2):
            if (labels[i] == ATTACHED and 
                labels[i+1] == DETACHED and 
                labels[i+2] == DETACHED):
                detachBubble.add((frames0[i], int(tids[i])))
                # count_detachments += 1
                # print(f"Frame0 {frames0[i]} : Bulle {df_evol.iloc[idx]['bubble_id']} a effectué une transition attached -> detached")
    freq = len(detachBubble) / (nb_frame / fps)
    print(f"Fréquence de détachement: {freq} Hz")    
    return detachBubble, freq

if __name__ == "__main__":
    # Exemple d'utilisation
    savefolder = r"Inputs\T87_out"
    extension = "T87_60V1"
    
    try:
        detachBubble, frequence = count_detachment_transitions(savefolder, extension)
        detachBubble = sorted(detachBubble, key=lambda x: x[0])  # Trier par frames0 (premier élément du tuple)
        print(f"Bulles avec transition attached -> detached: {detachBubble}")
        print(f"Nombre de bulles avec transition attached -> detached: {len(detachBubble)}")
    except Exception as e:
        print(f"Erreur: {e}")
