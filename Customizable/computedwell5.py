import os, csv, cv2, numpy as np
import pandas as pd
from csteDef import *

min_attached_run=0
# savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
# extension="Test6" 
savefolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
extension="T113_2_60V_2" 
score_thres = 0.7
N_FRAMES_POST_DISAPPEAR = 2


def _replaceChangedID(rich_df, fusion_df, changeID_df):
    """
    Modifie le dataframe rich et le fusion_df pour prendre en compte les changements de track id
    
    Args:
        rich_df (pd.DataFrame): DataFrame rich avec les colonnes ['track_id', 'frame', ...]
        fusion_df (pd.DataFrame): DataFrame des fusions avec colonnes ['frame', 'child', 'parent1', 'parent2', ...]
        changeID_df (pd.DataFrame): DataFrame des corrections avec colonnes ['frame', 'new_id', 'old_id']
    
    Returns:
        pd.DataFrame: rich dataframe avec les track_id corrigés
        pd.DataFrame: DataFrame fusion corrigé avec les IDs mis à jour
    """
    
    # Créer des copies pour éviter les modifications des données originales
    rich_corrige = rich_df.copy()
    fusion_corrige = fusion_df.copy()
    changeID_df = changeID_df.copy()
    
    # 1. Corriger le rich_df
    # Trier les corrections par frame croissant pour appliquer dans l'ordre
    changeID_sorted = changeID_df.sort_values('frame')
    
    for _, correction in changeID_sorted.iterrows():
        frame_corr = correction['frame']
        new_id = correction['new_id']
        old_id = correction['old_id']
        
        # Appliquer la correction : pour toutes les frames >= frame_corr, remplacer new_id par old_id
        mask = (rich_corrige["frame"] >= frame_corr) & (rich_corrige["track_id"] == new_id)
        rich_corrige.loc[mask, "track_id"] = old_id


        cols = ["child", "parent1", "parent2"] # colonne a modifier
        mask = (fusion_corrige["frame"] >= frame_corr)
        fusion_corrige.loc[mask, cols] = fusion_corrige.loc[mask, cols].replace(new_id, old_id)

    return rich_corrige, fusion_corrige

def followMerge(results):
    # Créer une copie pour éviter les modifications sur l'original
    results_copy = results.copy()
    results_copy = results_copy.set_index("bubble_id")
    
    # Filtrer les bulles qui ont fusionné
    merge_df = results[results["child_id"] != None]
    new_results = []
    
    # Parcourir chaque bulle qui a fusionné
    for _, mergeBb in merge_df.iterrows():
        child_id = mergeBb["bubble_id"]
        child = mergeBb["child_id"]
        
        if child is None or child not in results_copy.index:
            continue
            
        mergePath = f"{child_id}->{child}"
        first_frame = mergeBb["attach_start_frame"]
        
        # Calculer le dwell_frames cumulé
        dwell_frames = mergeBb["dwell_frames"] + results_copy.at[child, "dwell_frames"]
        
        # Calculer le dwell_seconds cumulé
        dwell_seconds = mergeBb["dwell_seconds"] + results_copy.at[child, "dwell_seconds"]
        
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
        noteChild = results_copy.at[child, "child_id"]
        if noteChild is not None:
            # Récursion pour suivre la chaîne de fusions
            child_chain = followMerge_single(child, results_copy)
            if child_chain is not None:
                # Mettre à jour avec les données de la chaîne complète
                dwell_frames += child_chain["dwell_frames"] - results_copy.at[child, "dwell_frames"]
                dwell_seconds += child_chain["dwell_seconds"] - results_copy.at[child, "dwell_seconds"]
                final_child = child_chain["bubble_id"].split("->")[-1]
                mergePath = f"{child_id}->{child_chain['bubble_id']}"
                detach_frame = child_chain["detach_frame"]
                note = child_chain["note"]
            else:
                detach_frame = results_copy.at[child, "detach_frame"]
                note = results_copy.at[child, "note"]
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
            "child_id":  results_copy.at[child, "child_id"],
            "note": note,
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
    
    if child_data["child_id"] == None:
        return None
    
    # Récupérer l'enfant suivant
    next_child = child_data["child_id"]
    
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

path = os.path.join(savefolder, f"fusionResult_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError(f"{path} not found")
df_fusion = pd.read_csv(path)

path = os.path.join(savefolder, f"changeIDResult_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError("rich_ file not found")
changeID_df = pd.read_csv(path)

# Supprimer les lignes dont le score est inférieur à score_thres
df_filter = rich_df[rich_df['score'] >= score_thres]

# S'il reste des duplicata, tieni, per ogni (track_id, frame0), la detection con score max
df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
        .drop_duplicates(["track_id", "frame"], keep="first"))

df_score = df_filter[["track_id", "frame", "score", "class_id"]].copy()

df_score, df_fusion = _replaceChangedID(df_score, df_fusion, changeID_df)

# Paramètres
fps = 4000  # TODO Ou déterminer automatiquement comme dans l'ancien code

last_frame = df_score['frame'].max()
results = []

# Parcourir chaque track_id unique
for track_id in sorted(df_score['track_id'].unique()):
    # Default values
    note = ''
    noteChild = None
    detach_frame = None
    attach_start_frame = None
    isDetached = False
    
    
    track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
    
    # Calculer les statistiques de base
    n_frames_tracked = len(track_data)
    n_unknown = len(track_data[track_data['class_id'] == UNKNOWN])
    jump = np.diff(track_data["frame"].values) - 1
    n_frame_undetected = jump.sum()
    
    # Trouver la période d'attachement
    if ATTACHED not in track_data["class_id"].values: #la bulle n'est jamais attachee
        note = "no_attached_run"
  
    else:
        attach_start_frame = track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0]
        if track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0] - track_data['frame'].min() < 3: #TODO la premiere frame est attache
            pass
        else:
            note = note + 'NOT attached at the begining/'
        if df_fusion["child"].isin([track_id]).any():
            note = "come from a merge/"
            # TODO
        if (not df_fusion["parent1"].isin([track_id]).any()) and (not df_fusion["parent2"].isin([track_id]).any()): # la bulle ne merge pas
            if DETACHED not in track_data["class_id"].values: #la bulle n'est jamais detachee
                if track_data["frame"].iat[-1] == last_frame:
                    note = note + "attach until end"
                                
                else:
                    note = note + "disappear after frame" + str(int(track_data["frame"].iat[-1]))
                    
            else:
                detach_frame = track_data.loc[(track_data["class_id"] == DETACHED)
                                              & (track_data["frame"] > attach_start_frame), "frame"].iloc[0]
                if track_data[track_data["frame"] > detach_frame].class_id.isin([ATTACHED]).any():
                    note = note + "WARNING the bubble reattached after"
                if detach_frame is None:
                    note += "..."
                else:
                    note = note + "/DETACHED"
                    isDetached = True
                
        else: # la bulle va merge
            note = note + "/PARENT"
            df_merge = df_fusion[(df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id)]
            if len(df_merge)>1:
                note += "Warning more than 1 merge for this bubble"
            frame_merge = df_merge.iloc[0]["frame"]
            # detach_frame = track_data.loc[track_data["class_id"] == DETACHED, "frame"].iloc[0]
            if DETACHED not in track_data["class_id"].values: #la bulle n'est jamais detachee
                detach_frame = frame_merge
                note += "/CHILD=" + str(df_merge.iloc[0]["child"])
                noteChild = df_merge.iloc[0]["child"]
        
    
    
    
    if attach_start_frame is not None and detach_frame is not None:
        dwell_frames = detach_frame- attach_start_frame
    else:
        dwell_frames = np.nan
        
        
        
    # mean score
    mean_score = None
    if attach_start_frame is not None:
        if detach_frame is None:
            end_frame = last_frame
        else:
            end_frame = detach_frame
        attach_frames = track_data[
            (track_data['frame'] >= attach_start_frame) & 
            (track_data['frame'] <= end_frame)
        ]['frame'].tolist()
        scores = track_data[track_data['frame'].isin(attach_frames)]['score'].astype(float)
        mean_score = float(scores.mean())
        
    results.append({
        "bubble_id": track_id,
        "attach_start_frame": attach_start_frame,
        "detach_frame": detach_frame,
        "dwell_frames": dwell_frames,
        "dwell_seconds": dwell_frames/fps, #TODO
        "n_frames_tracked": n_frames_tracked,
        "n_unknown": n_unknown,
        "missing_detection": n_frame_undetected,
        "mean_score_pct": mean_score,
        "child_id":  noteChild,
        "detachs?": isDetached,
        "note": note,
    })
        

results = pd.DataFrame(results).astype({
    "attach_start_frame": "Int64",
    "detach_frame": "Int64",
    "child_id": "Int64",
    "dwell_frames": "Int64",
})
final_results = followMerge(pd.DataFrame(results))
final_results = pd.DataFrame(final_results).astype({
    "attach_start_frame": "Int64",
    "detach_frame": "Int64",
    "child_id": "Int64",
    "dwell_frames": "Int64",
})

# Sauvegarder les résultats
out_csv = os.path.join(savefolder, f'dwell4_{extension}.csv')
final_results.to_csv(out_csv, index=False)

print(f"Résultats sauvegardés dans: {out_csv}")