import os, math, csv, ast, json
import numpy as np, pandas as pd
from csteDef import *

# Classe pour stocker les vitesses et statistiques
class velocities:
    def __init__(self):
        self.vy = []
        self.dx = []
        self.sizeBlock = []
        self.vMean = -1
        self.vMeanPerBubble = None
        self.vStd = -1
        self.vStdPerBubble = None

    def convert2mm(self, mm_per_px):
        """Convertit toutes les vitesses et déplacements en millimètres."""
        self.vy_mm = [x * mm_per_px for x in self.vy]
        self.dx_mm = [x * mm_per_px for x in self.dx]
        self.vMean_mm = self.vMean * mm_per_px
        self.vMeanPerBubble_mm = np.array(self.vMeanPerBubble) * mm_per_px
        self.vStd_mm = self.vStd * mm_per_px
        self.vStdPerBubble_mm = np.array(self.vStdPerBubble) * mm_per_px


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


def compute_speed_blocks(frames0, track_ids, labels, contours, rich_df,
                         attach_vel, detach_vel, minPointForVelocity, fps):
    """
    Calcule les vitesses par bloc de labels (ATTACHED / DETACHED).
    Remplit les objets attach_vel et detach_vel.
    """
    df = pd.DataFrame({
        "frame": frames0,
        "track_id": track_ids,
        "label": labels
    })
    df["time"] = df["frame"] / fps
    df["block"] = (df["label"] != df["label"].shift()).cumsum()

    for _, group in df.groupby("block"):
        label = group["label"].iloc[0]
        if label == ATTACHED:
            position = 'top'
        elif label == DETACHED:
            position = 'bottom'
        else:
            continue  # on ignore les UNKNOWN

        coords = [extractPosition(fr, tid, contours, rich_df, position)
                  for fr, tid in zip(group["frame"], group["track_id"])]
        if len(coords) < minPointForVelocity:
            continue

        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        t = group["time"].to_numpy()

        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        vy = - dy / dt  # origine en haut à gauche

        if label == ATTACHED:
            attach_vel.vy.append(vy)
            attach_vel.dx.append(dx)
            attach_vel.sizeBlock.append(vy.size)
        else:
            detach_vel.vy.append(vy)
            detach_vel.dx.append(dx)
            detach_vel.sizeBlock.append(vy.size)


def bubble_velocities(savefolder, extension, minPointForVelocity=2, fps=4000):
    """
    Fonction principale qui charge les fichiers, calcule les vitesses
    et retourne deux objets velocities (attach_vel, detach_vel).
    """

    # =============================================================================
    # SECTION 1: INITIALISATION ET CHARGEMENT DES FICHIERS
    # =============================================================================

    # Chemins vers les fichiers
    rich_csv = os.path.join(savefolder, f"rich_{extension}.csv")
    evolution_csv = os.path.join(savefolder, f"evolutionID_{extension}.csv")
    scale_path = os.path.join(savefolder, f"scale_{extension}.json")
    contours_path = os.path.join(savefolder, f"contours_{extension}.json")
    out_csv = os.path.join(savefolder, f"departure_{extension}.csv")

    # Vérifications de sécurité
    for path in [rich_csv, evolution_csv, scale_path, contours_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} non trovato.")

    # Conversion pixels → mm
    with open(scale_path, "r") as f:
        dataScale = json.load(f)
    mm_per_px = float(dataScale["mm_per_px"])

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
    frames_arr = [[j for j, val in enumerate(row) if val is not None] for row in tid_arr]

    bubclass_arr = []
    for irow, row in enumerate(frames_arr):
        x = []
        for fr in row:
            x.append(int(rich_df[(rich_df["frame0"] == fr) &
                                 (rich_df["track_id"] == tid_arr[irow][fr])].iloc[0].at["class_id"]))
        bubclass_arr.append(x)

    # Création des objets velocities
    attach_vel = velocities()
    detach_vel = velocities()

    # Calcul des vitesses par bloc
    for idx, tid_evol in enumerate(tid_arr):
        frames0 = frames_arr[idx]
        labels = bubclass_arr[idx]
        track_ids = [x for x in tid_evol if x is not None]
        # TODO smooth labels
        compute_speed_blocks(frames0, track_ids, labels, contours, rich_df,
                             attach_vel, detach_vel, minPointForVelocity, fps)

    # TODO ajouter une condition si dx trop grand

    # Calcul des statistiques globales
    detach_vel.vMeanPerBubble = [arr.mean() for arr in detach_vel.vy]
    detach_vel.vMean = np.mean(np.concatenate(detach_vel.vy))
    detach_vel.vStdPerBubble = [arr.std() for arr in detach_vel.vy]
    detach_vel.vStd = np.std(np.concatenate(detach_vel.vy))

    attach_vel.vMeanPerBubble = [arr.mean() for arr in attach_vel.vy]
    attach_vel.vMean = np.mean(np.concatenate(attach_vel.vy))
    attach_vel.vStdPerBubble = [arr.std() for arr in attach_vel.vy]
    attach_vel.vStd = np.std(np.concatenate(attach_vel.vy))

    attach_vel.convert2mm(mm_per_px)
    detach_vel.convert2mm(mm_per_px)
    
    return attach_vel, detach_vel

