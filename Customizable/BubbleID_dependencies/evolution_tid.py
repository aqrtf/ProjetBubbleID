import os, csv, cv2, numpy as np
import pandas as pd
from csteDef import *

min_attached_run=0
savefolder=r"My_output\Test6"   # Define the folder you want the data to save in
extension="Test6" 
# savefolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
# extension="T113_2_60V_2" 
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


last_frame = df_score['frame'].max()
results = []

# Parcourir chaque track_id unique
for track_id in sorted(df_score['track_id'].unique()):

    nameBubble = str(track_id)
    last_seen_frame = None
    
    track_data = df_score[df_score['track_id'] == track_id].sort_values('frame')
    
    

    evolution_tid = [None] * last_frame
    first_seen_frame = track_data["frame"].min()
    last_seen_frame = track_data["frame"].max()
    evolution_tid[first_seen_frame-1] = track_id # frame entre 1 et last_frame
    score = 0
    for idx_frame in range(first_seen_frame, last_frame+1):
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
            evolution_tid[idx_frame-1] = track_id
        else:
            raise ValueError("Plusieurs valeurs trouvées")
 

    # # indices des entiers dans evolution_tid
    # int_indices = [i for i, x in enumerate(evolution_tid) if isinstance(x, int)]
    # if not int_indices:
    #     n_frames_tracked = missing_frame = -1  # aucun entier
    # else:
    #     start, end = int_indices[0], int_indices[-1]
    #     sublist = evolution_tid[start:end+1]
    #     # nb d'entier = nb de frame ou la bulle est detectee
    #     n_frames_tracked = sum(isinstance(x, int) for x in sublist)
    #     # nb de none entre les entiers = nb de frame ou la bulle n'est pas detecte 
    #     # peut etre qu'elle n'est pas detecte apres ou avant mais pas moyen de le savoir
    #     missing_frame = sum(x is None for x in sublist)
    
    not_none_idx = [i for i, x in enumerate(evolution_tid) if x is not None]
    if not not_none_idx:
        n_frames_tracked = missing_frame = -1  # aucun entier
    
    start, end = not_none_idx[0], not_none_idx[-1]
    sublist = evolution_tid[start:end+1]
    
    n_frames_tracked = sum(x is not None for x in sublist)
    missing_frame = sum(x is None for x in sublist)
    
    # # mean score
    mean_score = score/n_frames_tracked
        
    results.append({
        "bubble_id": nameBubble,
        "first_seen_frame": first_seen_frame,
        "last_seen_frame": last_seen_frame,
        "n_frames_tracked": n_frames_tracked,
        "missing_detection": missing_frame,
        "mean_score_pct": mean_score,
        "chemin": evolution_tid
    })
        

results = pd.DataFrame(results).astype({
    "first_seen_frame": "Int64",
    "last_seen_frame": "Int64",
})

# On retire les lignes qui sont une partie des autres
import pandas as pd
import re

def parse_tokens(series: pd.Series) -> pd.Series:
    # Découpe chaque bubble_id en liste de nombres
    return series.astype(str).apply(lambda s: [int(tok) for tok in re.split(r'<->|=>', s)])

def clean_bubble_ids(df: pd.DataFrame, group_col="last_seen_frame", id_col="bubble_id") -> pd.DataFrame:
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

# on renseigne la valeur du 1er trackID dans les chaines
results["first_tid"] = results["bubble_id"].str.extract(r'^(\d+)').astype(int)

# Sauvegarder les résultats
out_csv = os.path.join(savefolder, f'evolutionID_{extension}.csv')
results.to_csv(out_csv, index=False)

print(f"Résultats sauvegardés dans: {out_csv}")