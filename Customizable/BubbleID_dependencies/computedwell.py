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

# Chargement des données
rich_path = os.path.join(savefolder, f"rich_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError("rich_ file not found")
rich_df = pd.read_csv(rich_path)

path = os.path.join(savefolder, f"fusionResult_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError(f"{path} not found")
df_fusion = pd.read_csv(path)

path = os.path.join(savefolder, f"changeIDResultAll_{extension}.csv")
if not os.path.isfile(rich_path):
    raise FileNotFoundError("rich_ file not found")
changeID_df = pd.read_csv(path)

# Supprimer les lignes dont le score est inférieur à score_thres
df_filter = rich_df[rich_df['score'] >= score_thres]
# Supprime les lignes avec probleme de trackong
df_filter = df_filter[df_filter["track_id"].fillna(-1).astype(int) >= 0]

# S'il reste des duplicata, tieni, per ogni (track_id, frame0), la detection con score max
df_filter = (df_filter.sort_values(["track_id", "frame", "score"], ascending=[True, True, False])
        .drop_duplicates(["track_id", "frame"], keep="first"))

df_score = df_filter[["track_id", "frame", "score", "class_id"]].copy()


# Paramètres
fps = 4000  # TODO Ou déterminer automatiquement comme dans l'ancien code

last_frame = df_score['frame'].max()
results = []

# Parcourir chaque track_id unique
for track_id in sorted(df_score['track_id'].unique()):
    # Default values
    note = ''
    detach_frame = None
    attach_start_frame = None
    isDetached = False
    nameBubble = str(track_id)
    missing_frame = 0
    last_attach_frame = None
    
    track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
    
    # Calculer les statistiques de base
    n_frames_tracked = len(track_data)
    n_unknown = len(track_data[track_data['class_id'] == UNKNOWN])
    

    
    # Si la bulle n'est jamais attachee on continue
    if ATTACHED not in track_data["class_id"].values: #la bulle n'est jamais attachee
        note = "no_attached_run"
  
    else:
        attach_start_frame = track_data.loc[track_data["class_id"] == ATTACHED, "frame"].iloc[0]
        last_seen_frame = track_data["frame"].iloc[-1]
        last_attach_frame = attach_start_frame
        
        idx_frame = attach_start_frame
        score = df_score[(df_score["frame"] == idx_frame) & (df_score["track_id"] == track_id)]["score"].iloc[0]
        while(True):
            idx_frame+=1
            mask = (df_fusion["frame"] == idx_frame) & ((df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id))
            if (mask).any():
                # on regarde si la bulle est un parent a cette frame (donc est ce qu'elle merge?)
                track_id = df_fusion.loc[mask, "child"].iat[0]
                # La nouvelle bulle a track est child
                nameBubble += "=>" + str(track_id)
                last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()

            mask = (changeID_df["frame"] == idx_frame) & ((changeID_df["old_id"] == track_id))
            if (mask).any():
                # on regarde si la bulle change de trackid
                track_id = changeID_df.loc[mask, "new_id"].iat[0]
                # La nouvelle bulle a track est new_id
                nameBubble += "<->" + str(track_id)
                last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()

            
            # Filtrer
            subset = df_score[(df_score["frame"] == idx_frame) & (df_score["track_id"] == track_id)]
            # Vérifier unicité
            if subset.empty:
                missing_frame += 1
            elif len(subset) == 1:
                score += subset["score"].iloc[0]
                if subset["class_id"].iloc[0]  == DETACHED:
                    detach_frame = idx_frame
                    note = "DETACH"
                    isDetached = True
                    # TODO que se passe t'il si elle est detach apres avoir disparue du track apres un certain temps
                    # TODO renforcer la detection 
                    break
                elif subset["class_id"].iloc[0]  == ATTACHED:
                    last_attach_frame = idx_frame
            else:
                raise ValueError("Plusieurs valeurs trouvées")

            if idx_frame == last_frame:
                note = "attach_until_end"
                break
            if idx_frame > last_seen_frame:
                # Surplus de frame si jamais il y a un merge apres
                for idx_supp in range(last_seen_frame, last_seen_frame+1+N_FRAMES_POST_DISAPPEAR):
                    mask = (df_fusion["frame"] == idx_supp) & ((df_fusion["parent1"] == track_id) | (df_fusion["parent2"] == track_id))
                    if (mask).any():
                        # on regarde si la bulle est un parent a cette frame (donc est ce qu'elle merge?)
                        track_id = df_fusion.loc[mask, "child"].iat[0]
                        # La nouvelle bulle a track est child
                        nameBubble += "=>" + str(track_id)
                        last_seen_frame = df_score.loc[df_score["track_id"] == track_id, "frame"].max()
                        
                if idx_frame > last_seen_frame:
                    if attach_start_frame == last_frame:
                        note = "attach_until_end"
                        break
                    note = "disappear after frame "+ str(last_seen_frame)
                    break
   
    if attach_start_frame is not None and detach_frame is not None:
        dwell_frames = detach_frame- attach_start_frame
    else:
        dwell_frames = np.nan
      
    # # mean score
    mean_score = None
    if attach_start_frame is not None:
        if detach_frame is None:
            end_frame = last_frame
        else:
            end_frame = detach_frame
        n = end_frame-attach_start_frame+1
        mean_score = score/n
        
    results.append({
        "bubble_id": nameBubble,
        "attach_start_frame": attach_start_frame,
        "last_attach_frame": last_attach_frame,
        "detach_frame": detach_frame,
        "dwell_frames": dwell_frames,
        "dwell_seconds": dwell_frames/fps, #TODO
        "n_frames_tracked": n_frames_tracked,
        "n_unknown": n_unknown,
        "missing_detection": missing_frame,
        "mean_score_pct": mean_score,
        "detachs?": isDetached,
        "note": note,
    })
        

results = pd.DataFrame(results).astype({
    "attach_start_frame": "Int64",
    "last_attach_frame": "Int64",
    "detach_frame": "Int64",
    "dwell_frames": "Int64",
})

# On retire les lignes qui sont une partie des autres
import pandas as pd
import re

def parse_tokens(series: pd.Series) -> pd.Series:
    # Découpe chaque bubble_id en liste de nombres
    return series.astype(str).apply(lambda s: [int(tok) for tok in re.split(r'<->|=>', s)])

def clean_bubble_ids(df: pd.DataFrame, group_col="detach_frame", id_col="bubble_id") -> pd.DataFrame:
    df = df.copy()
    df["_tokens"] = parse_tokens(df[id_col])
    df["_len"] = df["_tokens"].apply(len)

    def filter_group(group: pd.DataFrame) -> pd.DataFrame:
        # Trie par longueur décroissante
        group = group.sort_values("_len", ascending=False)
        keep = []
        mask = []
        for tok in group["_tokens"]:
            # Vérifie si tok est suffixe de l’un des déjà gardés
            if any(kt[-len(tok):] == tok for kt in keep):
                mask.append(False)
            else:
                keep.append(tok)
                mask.append(True)
        return group[mask]

    try:
        result = df.groupby(group_col, dropna=False, group_keys=False).apply(filter_group)
    except TypeError:
        # fallback si dropna=False n’est pas supporté
        sentinel = "__MISSING__"
        df[group_col] = df[group_col].astype(object).where(df[group_col].notna(), sentinel)
        result = df.groupby(group_col, group_keys=False).apply(filter_group)
        result[group_col] = result[group_col].replace(sentinel, pd.NA)

    return result.drop(columns=["_tokens", "_len"])

results = clean_bubble_ids(results)

# Sauvegarder les résultats
out_csv = os.path.join(savefolder, f'dwell6_{extension}.csv')
results.to_csv(out_csv, index=False)

print(f"Résultats sauvegardés dans: {out_csv}")