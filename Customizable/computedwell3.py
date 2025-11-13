import os, csv, cv2, numpy as np
import pandas as pd
from csteDef import *

min_attached_run=2
savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
extension="Test6" 
score_thres = 0.7
N_FRAMES_POST_DISAPPEAR = 2

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

def _smooth_track(frames, classes, tolerate_unknown_gap=1, class_idx_attached=2, class_idx_unknown=1):
    """
    Lisse la séquence de classes en comblant les petits gaps d'unknown
    """
    if not frames:
        return frames, classes
    
    # Trier par frame
    idx = np.argsort(frames)
    fr_sorted = [frames[i] for i in idx]
    lb_sorted = [classes[i] for i in idx]
    
    out_classes = lb_sorted.copy()
    
    if tolerate_unknown_gap > 0:
        i = 0
        n = len(out_classes)
        while i < n:
            if out_classes[i] == class_idx_attached:
                j = i + 1
                unk_count = 0
                # Compter les unknown consécutifs
                while j < n and out_classes[j] == class_idx_unknown:
                    unk_count += 1
                    j += 1
                # Si le gap d'unknown est acceptable et qu'après c'est attached
                if 0 < unk_count <= tolerate_unknown_gap and j < n and out_classes[j] == class_idx_attached:
                    # Combler le gap
                    for k in range(i + 1, j):
                        out_classes[k] = class_idx_attached
                    i = j
                    continue
            i += 1
    
    return fr_sorted, out_classes

def find_attachment_period(track_data, fusionList, last_frame, 
                          class_idx_attached=2, class_idx_detached=0, 
                          min_attached_run=2, tolerate_unknown_gap=1):
    """
    Trouve la période d'attachement pour une bulle
    """
    frames = track_data['frame'].tolist()
    classes = track_data['class_id'].tolist()
    
    if not frames:
        return {
            'attach_start': None,
            'detach_frame': None, 
            'dwell_frames': 0,
            'note': 'no_frames'
        }
    
    # Appliquer le lissage
    fr_smoothed, cls_smoothed = _smooth_track(frames, classes, tolerate_unknown_gap)
    
    # Trouver la première run attached suffisamment longue
    attach_start = None
    attach_end_idx = None
    current_run = 0
    start_idx = None
    
    for i, cls in enumerate(cls_smoothed):
        if cls == ATTACHED:
            current_run += 1
            if attach_start is None:
                attach_start = fr_smoothed[i]
                start_idx = i
            attach_end_idx = i
        else:
            if current_run >= min_attached_run:
                break
            attach_start = None
            attach_end_idx = None
            start_idx = None
            current_run = 0
    
    if attach_start is None or current_run < min_attached_run:
        return {
            'attach_start': None,
            'detach_frame': None,
            'dwell_frames': 0,
            'note': 'no_attached_run'
        }
    
    # Trouver le frame de détachement
    detach_frame = None
    for j in range(attach_end_idx + 1, len(cls_smoothed)):
        if cls_smoothed[j] == DETACHED:
            detach_frame = fr_smoothed[j]
            break
    
    # Vérifier les fusions si pas de détachement trouvé
    if detach_frame is None:
        # Vérifier si la bulle a fusionné avec une autre
        last_track_frame = fr_smoothed[-1]
        merged = False
        
        for frame_fusion, mergers in fusionList.items():
            if frame_fusion >= last_track_frame:
                for merger in mergers:
                    if track_id in merger[1:]:  # La bulle est un parent dans la fusion
                        detach_frame = frame_fusion
                        merged = True
                        break
                if merged:
                    break
        
        if not merged and last_track_frame == last_frame:
            # La bulle reste attachée jusqu'à la fin de la vidéo
            dwell_frames = last_track_frame - attach_start + 1
            note = "attached_until_end"
        elif not merged:
            # Disparition inexpliquée
            dwell_frames = fr_smoothed[attach_end_idx] - attach_start + 1
            note = "disappeared"
        else:
            dwell_frames = detach_frame - attach_start
            note = "MERGED"
    else:
        dwell_frames = detach_frame - attach_start
        note = "DETACHED"
    
    dwell_frames = max(int(dwell_frames), 0)
    
    return {
        'attach_start': attach_start,
        'detach_frame': detach_frame,
        'dwell_frames': dwell_frames,
        'note': note,
        'start_idx': start_idx,
        'end_frame': detach_frame if detach_frame else fr_smoothed[attach_end_idx]
    }

# Chargement des données
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

df_score, fusionDict = _replaceChangedID(df_score, fusionDict, changeIDList)
fusionList = {} # on remplace la structure de fusionDict en fusionList = {frame: [[child, parent1, parent2], ...], ...}
for frame, tracks in fusionDict.items():
    fusionList[frame] = []
    for new_tid, parents in tracks.items():
        fusionList[frame].append([new_tid]+parents)

# Paramètres
tolerate_unknown_gap = 1
class_idx_attached = 2  # À adapter selon vos constantes
class_idx_detached = 0  # À adapter selon vos constantes
class_idx_unknown = 1   # À adapter selon vos constantes
fps = 4000  # Ou déterminer automatiquement comme dans l'ancien code

last_frame = df_score['frame'].max()
results = []

# Parcourir chaque track_id unique
for track_id in sorted(df_score['track_id'].unique()):
    track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
    
    # Calculer les statistiques de base
    n_frames_tracked = len(track_data)
    n_unknown = len(track_data[track_data['class_id'] == class_idx_unknown])
    
    # Trouver la période d'attachement
    attachment_info = find_attachment_period(
        track_data, fusionList, last_frame,
        class_idx_attached, class_idx_detached,
        min_attached_run, tolerate_unknown_gap
    )
    
    # Calculer le score moyen pendant l'attachement
    mean_score_pct = np.nan
    if attachment_info['attach_start'] is not None and attachment_info['start_idx'] is not None:
        # Extraire les frames de la période d'attachement
        attach_frames = track_data[
            (track_data['frame'] >= attachment_info['attach_start']) & 
            (track_data['frame'] <= attachment_info['end_frame'])
        ]['frame'].tolist()
        
        if attach_frames:
            scores = track_data[
                track_data['frame'].isin(attach_frames)
            ]['score'].astype(float)
            if len(scores) > 0:
                mean_score_pct = float(scores.mean() * 100.0)
    
    # Calculer le temps en secondes
    dwell_seconds = float(attachment_info['dwell_frames']) / float(fps) if fps else 0.0
    
    # Ajouter au résultats
    results.append({
        "bubble_id": track_id,
        "attach_start_frame": attachment_info['attach_start'],
        "detach_frame": attachment_info['detach_frame'],
        "dwell_frames": attachment_info['dwell_frames'],
        "dwell_seconds": dwell_seconds,
        "n_frames_tracked": n_frames_tracked,
        "n_unknown": n_unknown,
        "mean_score_pct": mean_score_pct,
        "note": attachment_info['note']
    })

# Sauvegarder les résultats
out_csv = os.path.join(savefolder, f'dwell2_{extension}.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        "bubble_id", "attach_start_frame", "detach_frame",
        "dwell_frames", "dwell_seconds", "n_frames_tracked",
        "n_unknown", "mean_score_pct", "note"
    ])
    w.writeheader()
    for r in results:
        w.writerow(r)

print(f"Résultats sauvegardés dans: {out_csv}")