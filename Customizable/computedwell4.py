import os, csv, cv2, numpy as np
import pandas as pd
from csteDef import *

min_attached_run=0
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

def _smooth_track(frames, classes, tolerate_unknown_gap=1):
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
            if out_classes[i] == ATTACHED:
                j = i + 1
                unk_count = 0
                # Compter les unknown consécutifs
                while j < n and out_classes[j] == UNKNOWN:
                    unk_count += 1
                    j += 1
                # Si le gap d'unknown est acceptable et qu'après c'est attached
                if 0 < unk_count <= tolerate_unknown_gap and j < n and out_classes[j] == ATTACHED:
                    # Combler le gap
                    for k in range(i + 1, j):
                        out_classes[k] = ATTACHED
                    i = j
                    continue
            i += 1
    
    return fr_sorted, out_classes
 

def followMerge(results):
    # Créer une copie pour éviter les modifications sur l'original
    results_copy = results.copy()
    results_copy = results_copy.set_index("bubble_id")
    
    # Filtrer les bulles qui ont fusionné
    merge_df = results[results["note2"] != None]
    new_results = []
    
    # Parcourir chaque bulle qui a fusionné
    for _, mergeBb in merge_df.iterrows():
        child_id = mergeBb["bubble_id"]
        child = mergeBb["note2"]
        
        if child is None or child not in results_copy.index:
            continue
            
        mergePath = f"{child_id}->{child}"
        first_frame = mergeBb["attach_start_frame"]
        
        # # Calculer le dwell_frames cumulé
        # dwell_frames = mergeBb["dwell_frames"] + results_copy.at[child, "dwell_frames"]
        
        # # Calculer le dwell_seconds cumulé
        # dwell_seconds = mergeBb["dwell_seconds"] + results_copy.at[child, "dwell_seconds"]
        
        # Calculer les autres métriques cumulées
        n_frames_tracked = mergeBb["n_frames_tracked"] + results_copy.at[child, "n_frames_tracked"]
        n_unknown = mergeBb["n_unknown"] + results_copy.at[child, "n_unknown"]
        
        # Calculer le score moyen pondéré
        score1 = mergeBb["mean_score_pct"] if pd.notna(mergeBb["mean_score_pct"]) else 0
        score2 = results_copy.at[child, "mean_score_pct"] if pd.notna(results_copy.at[child, "mean_score_pct"]) else 0
        frames1 = mergeBb["n_frames_tracked"]
        frames2 = results_copy.at[child, "n_frames_tracked"]
        
        if frames1 + frames2 > 0:
            mean_score_pct = (score1 * frames1 + score2 * frames2) / (frames1 + frames2)
        else:
            mean_score_pct = np.nan
        
        # Vérifier si l'enfant a aussi fusionné (fusion en chaîne)
        note2 = results_copy.at[child, "note2"]
        if not note2:
            # Récursion pour suivre la chaîne de fusions
            child_chain = followMerge_single(child, results_copy)
            if child_chain:
                # Mettre à jour avec les données de la chaîne complète
                dwell_frames += child_chain["dwell_frames"] - results_copy.at[child, "dwell_frames"]
                dwell_seconds += child_chain["dwell_seconds"] - results_copy.at[child, "dwell_seconds"]
                final_child = child_chain["bubble_id"].split("->")[-1]
                mergePath = f"{child_id}->{child_chain['bubble_id']}"
                detach_frame = child_chain["detach_frame"]
                note = child_chain["note"]
            else:
                detach_frame = results_copy.at[child, "detach_frame"]
                note = child_chain["note"] # "merged_chain"
        else:
            detach_frame = results_copy.at[child, "detach_frame"]
            # note = "merged"
        
        # Créer le nouvel enregistrement
        new_record = {
            "bubble_id": mergePath,
            "attach_start_frame": first_frame,
            "detach_frame": detach_frame,
            "dwell_frames": dwell_frames,
            "dwell_seconds": dwell_seconds,
            "n_frames_tracked": n_frames_tracked,
            "n_unknown": n_unknown,
            "mean_score_pct": mean_score_pct,
            "note": note,
            "note2":  results_copy.at[child, "note2"] # f"fusion_chain_{child_id}_{child}"
        }
        
        new_results.append(new_record)
    
    # Ajouter les nouveaux résultats à la liste originale (sans les enregistrements fusionnés individuels)
    final_results = []
    
    # Garder seulement les bulles qui n'ont pas fusionné (ou sont le dernier maillon d'une chaîne)
    non_merged_results = results[results["note"] != "merged"].copy()
    final_results.extend(non_merged_results.to_dict('records'))
    final_results.extend(new_results)
    
    return pd.DataFrame(final_results)

def followMerge_single(child_id, results_df):
    """
    Fonction helper pour suivre une chaîne de fusions pour un enfant donné
    """
    if child_id not in results_df.index:
        return None
        
    child_data = results_df.loc[child_id]
    
    if child_data["note2"] == None:
        return None
    
    # Récupérer l'enfant suivant
    next_child = child_data["note2"]
    
    if next_child is None or next_child not in results_df.index:
        return None
    
    # Récursivement suivre la chaîne
    chain_result = followMerge_single(next_child, results_df)
    
    if chain_result:
        # Combiner avec la chaîne existante
        mergePath = f"{child_id}->{chain_result['bubble_id']}"
        dwell_frames = child_data["dwell_frames"] + chain_result["dwell_frames"]
        dwell_seconds = child_data["dwell_seconds"] + chain_result["dwell_seconds"]
        
        return {
            "bubble_id": mergePath,
            "attach_start_frame": child_data["attach_start_frame"],
            "detach_frame": chain_result["detach_frame"],
            "dwell_frames": dwell_frames,
            "dwell_seconds": dwell_seconds,
            "note": chain_result["note"]
        }
    else:
        # Fin de la chaîne
        return {
            "bubble_id": f"{child_id}->{next_child}",
            "attach_start_frame": child_data["attach_start_frame"],
            "detach_frame": results_df.loc[next_child, "detach_frame"],
            "dwell_frames": child_data["dwell_frames"] + results_df.loc[next_child, "dwell_frames"],
            "dwell_seconds": child_data["dwell_seconds"] + results_df.loc[next_child, "dwell_seconds"],
            "note": child_data["note"]
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
# on remplace fusionDict en un dataframe
rows = []
for frame, tracks in fusionDict.items():
    for child, parents in tracks.items():
        parent1 = parents[0] 
        parent2 = parents[1] 
        if len(parents) > 2:
            print(f"WARNING: bubble {child} (frame {frame}) has more than 2 parents")
        rows.append({"frame": frame, "child": child, "parent1": parent1, "parent2": parent2})

df_fusion = pd.DataFrame(rows)
print(df_fusion)
# Paramètres
tolerate_unknown_gap = 1
fps = 4000  # Ou déterminer automatiquement comme dans l'ancien code

last_frame = df_score['frame'].max()
results = []

# Parcourir chaque track_id unique
for track_id in sorted(df_score['track_id'].unique()):
    # Default values
    note = ''
    noteChild = None
    detach_frame = None
    attach_start_frame = None
    
    
    track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
    
    # Calculer les statistiques de base
    n_frames_tracked = len(track_data)
    n_unknown = len(track_data[track_data['class_id'] == UNKNOWN])
    
    # Trouver la période d'attachement
    if ATTACHED not in track_data["class_id"].values: #la bulle n'est jamais attachee
        results.append({
            "bubble_id": track_id,
            "attach_start_frame": None,
            "detach_frame": None,
            "dwell_frames": None,
            "dwell_seconds": None,
            "n_frames_tracked": n_frames_tracked,
            "n_unknown": n_unknown,
            "mean_score_pct": None,
            "note": "no_attached_run",
            "note2":  None
        })
        continue
    
    
    
    if track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0] - track_data['frame'].min() < 3: #NOTE la premiere frame est attache
        attachStartFrame = track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0]
        note = note + 'attached at the begining/'
    if df_fusion["child"].isin([track_id]).any():
        note = "come from a merge/"
        # TODO
    if (not df_fusion["parent1"].isin([track_id]).any()) and (not df_fusion["parent2"].isin([track_id]).any()): # la bulle ne merge pas
        if DETACHED not in track_data["class_id"].values: #la bulle n'est jamais detachee
            attach_start_frame = track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0]  
            if track_data["frame"].iat[-1] == last_frame:
                note = note + "attach until end"
                              
            else:
                note = note + "disappear after frame" + str(int(track_data["frame"].iat[-1]))
                
        else:
            detach_frame = track_data.loc[track_data["class_id"] == DETACHED, "frame"].iloc[0]
            if track_data[track_data["frame"] > detach_frame].class_id.isin([ATTACHED]).any():
                note = note + "WARNING the bubble reattached after"
            note = note + "/DETACHED"
            attach_start_frame = track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0]
    else: # la bulle va merge
        note = note + "/PARENT"
        df_merge = df_fusion[(df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id)]
        if len(df_merge)>1:
            note += "Warning more than 1 merge for this bubble"
        frame_merge = df_merge.at[0, "frame"]
        # detach_frame = track_data.loc[track_data["class_id"] == DETACHED, "frame"].iloc[0]
        if DETACHED not in track_data["class_id"].values: #la bulle n'est jamais detachee
            detach_frame = frame_merge
            note += "/CHILD=" + str(df_merge.at[0, "child"])
            noteChild = df_merge.at[0, "child"]
        
          
    results.append({
        "bubble_id": track_id,
        "attach_start_frame": attach_start_frame,
        "detach_frame": detach_frame,
        "dwell_frames": None,
        "dwell_seconds": None,
        "n_frames_tracked": n_frames_tracked,
        "n_unknown": n_unknown,
        "mean_score_pct": None,
        "note": note,
        "note2":  noteChild
    })
        
    
    # # Calculer le score moyen pendant l'attachement
    # mean_score_pct = np.nan
    # if attachment_info['attach_start'] is not None and attachment_info['start_idx'] is not None:
    #     # Extraire les frames de la période d'attachement
    #     attach_frames = track_data[
    #         (track_data['frame'] >= attachment_info['attach_start']) & 
    #         (track_data['frame'] <= attachment_info['end_frame'])
    #     ]['frame'].tolist()
        
    #     if attach_frames:
    #         scores = track_data[
    #             track_data['frame'].isin(attach_frames)
    #         ]['score'].astype(float)
    #         if len(scores) > 0:
    #             mean_score_pct = float(scores.mean() * 100.0)
    
    # # Calculer le temps en secondes
    # dwell_seconds = float(attachment_info['dwell_frames']) / float(fps) if fps else 0.0
    
    # # Ajouter au résultats
    # results.append({
    #     "bubble_id": track_id,
    #     "attach_start_frame": attachment_info['attach_start'],
    #     "detach_frame": attachment_info['detach_frame'],
    #     "dwell_frames": attachment_info['dwell_frames'],
    #     "dwell_seconds": dwell_seconds,
    #     "n_frames_tracked": n_frames_tracked,
    #     "n_unknown": n_unknown,
    #     "mean_score_pct": mean_score_pct,
    #     "note": attachment_info['note'],
    #     "note2": attachment_info['note2']
    # })

results = pd.DataFrame(results).astype({
    "attach_start_frame": "Int64",
    "detach_frame": "Int64",
    "note2": "Int64",
})
# final_results = followMerge(pd.DataFrame(results))
# final_results = pd.DataFrame(final_results).astype({
#     "attach_start_frame": "Int64",
#     "detach_frame": "Int64",
#     "note2": "Int64",
# })

# Sauvegarder les résultats
out_csv = os.path.join(savefolder, f'dwell4_{extension}.csv')
results.to_csv(out_csv, index=False)
# with open(out_csv, 'w', newline='') as f:
#     w = csv.DictWriter(f, fieldnames=[
#         "bubble_id", "attach_start_frame", "detach_frame",
#         "dwell_frames", "dwell_seconds", "n_frames_tracked",
#         "n_unknown", "mean_score_pct", "note", "note2"
#     ])
#     w.writeheader()
#     for r in results:
#         w.writerow(r)

print(f"Résultats sauvegardés dans: {out_csv}")