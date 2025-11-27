import os, csv, ast, json
import numpy as np, pandas as pd
from csteDef import *

savefolder = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\Inputs\T88_out"
extension = "T88_50V1"

def extractPosition(frame0, tid, contours, rich_df, position):
    """
    Extrait la position de la bulle selon le mode choisi.
    position : 'top', 'bottom', 'centroid'
    """
    if position == "centroid":
        coord = rich_df.loc[
            (rich_df["frame"] == frame0 + 1) & (rich_df["track_id"] == tid),
            ["cx_px", "cy_px"]
        ].values[0]
    else:
        frame = frame0 + 1
        detInFrame = rich_df.loc[
            (rich_df["frame"] == frame) & (rich_df["track_id"] == tid),
            "det_in_frame"
        ].values[0]
        clef = str(frame) + '_' + str(detInFrame)
        contourBulle = contours[clef]
        if position == "top":
            coord = min(contourBulle, key=lambda c: c[1])
        elif position == "bottom":
            coord = max(contourBulle, key=lambda c: c[1])
        else:
            raise ValueError(f"Wrong argument for position: {position}")
    return coord


# Chemins vers les fichiers
rich_csv = os.path.join(savefolder, f"rich_{extension}.csv")
evolution_csv = os.path.join(savefolder, f"evolutionID_{extension}.csv")
scale_path = os.path.join(savefolder, f"scale_{extension}.json")
contours_path = os.path.join(savefolder, f"contours_{extension}.json")
imagefolder = os.path.join(savefolder, "trimImages_"+ extension)
output_path = os.path.join(savefolder, f"positions_{extension}.csv")

# Vérifications de sécurité
for path in [rich_csv, evolution_csv, scale_path, contours_path]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} non trovato.")
    
with open(contours_path, 'r') as f:
        contours = json.load(f)

# Chargement du DataFrame principal
rich_df = pd.read_csv(rich_csv)
rich_df.columns = [c.strip().lower() for c in rich_df.columns]
rich_df = rich_df.loc[:, ~pd.Index(rich_df.columns).duplicated(keep='first')]
rich_df["frame0"] = rich_df["frame"].astype(int) - 1
rich_df = rich_df[rich_df["track_id"].fillna(-1).astype(int) >= 0]

# Evolution des tracks
df_evol = pd.read_csv(evolution_csv)
tid_arr = df_evol["chemin"].apply(ast.literal_eval).to_list()

bubclass_arr = []
for irow, row in enumerate(tid_arr):
    x = []
    for fr, tid in enumerate(row):
        if tid is None:
            x.append(None)
        else:
            x.append(int(rich_df[(rich_df["frame0"] == fr) &
                                (rich_df["track_id"] == tid_arr[irow][fr])].iloc[0].at["class_id"]))
    bubclass_arr.append(x)

data = {}
for ilig, tid_vec in enumerate(tid_arr):
    cx_all, cy_all, bx_all, by_all, tx_all, ty_all = [None]*len(tid_vec), [None]*len(tid_vec), [None]*len(tid_vec), [None]*len(tid_vec), [None]*len(tid_vec), [None]*len(tid_vec)
    for frame0, tid in enumerate(tid_vec):
        if tid is not None:
            x, y = extractPosition(frame0, tid, contours, rich_df, 'centroid')
            cx_all[frame0] = int(x)
            cy_all[frame0] = int(y)
            x, y = extractPosition(frame0, tid, contours, rich_df, 'bottom')
            bx_all[frame0] = int(x)
            by_all[frame0] = int(y)
            x, y = extractPosition(frame0, tid, contours, rich_df, 'top')
            tx_all[frame0] = int(x)
            ty_all[frame0] = int(y)
    data[f"{ilig}_tid"]   = tid_arr[ilig]
    data[f"{ilig}_label"] = bubclass_arr[ilig]
    data[f"{ilig}_cx"]    = cx_all
    data[f"{ilig}_cy"]    = cy_all
    data[f"{ilig}_bx"]    = bx_all
    data[f"{ilig}_by"]    = by_all
    data[f"{ilig}_tx"]    = tx_all
    data[f"{ilig}_ty"]    = ty_all

df_out = pd.DataFrame(data)  
# Conversion des colonnes en Int16 avec support des valeurs manquantes
# for col in df_out.columns:
#     df_out[col] = df_out[col].astype(pd.Int16Dtype()) 
df_out.to_csv(output_path)
