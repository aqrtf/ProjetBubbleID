import os, csv, cv2, numpy as np
import pandas as pd
from csteDef import *

min_attached_run=2
savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
extension="Test6" 
score_thres = 0.7

from parentBubble4 import findMerge
fusionDict, changeIDList = findMerge(savefolder, extension, score_thres=0.7, OVERLAP_THRESH=0.1,
                                    MIN_OVERLAP_SAME=0.7, POST_FUSION_FRAMES=2, N_FRAMES_PREVIOUS_DISAPPEAR=3, 
                                    N_FRAMES_POST_DISAPPEAR=2,
                                    IMAGE_SHAPE=(1024, 1024), DILATE_ITERS=1
                                    )



def _replaceChangedID(rich_df, fusionDict, changeIDList):   
       
    """
    Modifie le dataframe rich et le fusionDict pour prendre en compte les changement de track id reperes
    Args:
        rich_df (df): 
        fusionDict (dict): Dictionnaire des fusions sous forme {frame: {child_id: [parent_ids]}}
        changeIDList (list): Liste des corrections à appliquer sous forme [[frame, new_id, old_id], ...]
    
    Returns:
        df: rich dataframe avec les trackid corrige
        dict: Dictionnaire fusion corrigé avec les IDs mis à jour 
    """
    for frame, new_id, old_id in changeIDList:
        rich_df.loc[(rich_df['frame'] >= frame) & (rich_df['track_id'] == new_id), 'track_id'] = old_id
        
    # Dictionnaire qui contiendra les résultats corrigés
    fusion_corrige = {}
    
    # Parcourir chaque frame et ses fusions dans le dictionnaire original
    for frame_fusion, fusions in fusionDict.items():
        # Dictionnaire temporaire pour stocker les fusions corrigées de cette frame
        nouvelles_fusions = {}
        
        # Parcourir chaque paire child_id -> parent_ids dans les fusions de cette frame
        for child_id, parent_ids in fusions.items():
            # Créer un mapping de correction spécifique pour CETTE frame
            # Seules les corrections avec frame_corr <= frame_fusion seront appliquées
            correction_map = {}
            for frame_corr, new_id, old_id in changeIDList:
                # Condition clé : n'appliquer la correction que si la frame actuelle 
                # est postérieure ou égale à la frame de correction
                if frame_fusion >= frame_corr:
                    correction_map[new_id] = old_id  # Mapper new_id vers old_id
            
            # Appliquer les corrections au child_id actuel
            # Si child_id est dans correction_map, on prend la valeur corrigée, sinon on garde l'original
            child_corrige = correction_map.get(child_id, child_id)
            
            # Appliquer les corrections à chaque parent_id dans la liste
            # Pour chaque parent_id, on vérifie s'il doit être corrigé
            parents_corriges = [
                correction_map.get(pid, pid) for pid in parent_ids
            ]
            
            # Gérer les doublons de clés (si deux child_id différents deviennent identiques après correction)
            if child_corrige in nouvelles_fusions:
                # Si le child_id corrigé existe déjà, fusionner les listes de parents
                nouvelles_fusions[child_corrige].extend(parents_corriges)
                # Supprimer les doublons dans la liste des parents fusionnés
                nouvelles_fusions[child_corrige] = list(set(nouvelles_fusions[child_corrige]))
            else:
                # Si le child_id corrigé n'existe pas encore, créer une nouvelle entrée
                nouvelles_fusions[child_corrige] = parents_corriges
        
        # Assigner les fusions corrigées de cette frame au dictionnaire résultat
        fusion_corrige[frame_fusion] = nouvelles_fusions
    
    # Retourner le dictionnaire complet avec toutes les corrections appliquées
    return rich_df, fusion_corrige
                


rich_path = os.path.join(savefolder, f"rich_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError("rich_ file not found")
rich_df = pd.read_csv(rich_path)

# Supprimer les lignes dont le score est inférieur à score_thres
df_filter = rich_df[rich_df['score'] >= score_thres]

# S'il reste des duplicata, tieni, per ogni (track_id, frame0), la detection con score max
df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
        .drop_duplicates(["track_id", "frame"], keep="first"))

df_score = df_filter[["track_id", "frame", "score", "class_id"]].copy()

# df_score, fusionDict = _replaceChangedID(df_score, fusionDict, changeIDList)
fusionList = {} # on remplace la structure de fusionDict en fusionList = {frame: [[child, parent1, parent2], ...], ...}
for frame, tracks in fusionDict.items():
    fusionList[frame] = []
    for new_tid, parents in tracks.items():
        fusionList[frame].append([new_tid]+parents)



tids = set(df_score["track_id"])
frames = set(df_score["frame"])

# Parcourir chaque track_id unique
for track_id in df_score['track_id'].unique():
    df_track = df_score[df_score['track_id'] == track_id].sort_values('frame')
    
    # 1. Détection apparition
    frame_apparition = df_track['frame'].min()
    jump = np.diff(df_track['frame'])-1
    np.insert(jump, 0,0) # on rajoute 0 devant jump pour conserver la meme taille
    maxjump = max(jump)
    lostframe = np.sum(jump)
    
    detachedIdx = np.argwhere(df_track['class_id']==DETACHED)
    if detachedIdx == None: #La bulle ne s'est pas detachee, verifier les merge
        pass
    elif detachedIdx[0] == frame_apparition: #la frame est detachee des le debut
        pass
    else:
        pass
        
    









for tid in tids:
    firstAppearFrame = -1
    for frame in frames:
        existe = ((df_score['track_id'] == tid) & (df_score['frame'] == frame)).any()
        if existe:
            if firstAppearFrame<0:
                firstAppearFrame = frame
            else: #on a deja vu la bulle 
                pass







