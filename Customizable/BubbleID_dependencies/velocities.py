import os, math, csv, ast, json
import numpy as np, pandas as pd
from csteDef import *
from computeDiameter import bubbleDiameter

# TODO pour la velocity il faut prendre en compte que entre les merges
# TODO il serait bien de comparer la vitesse avec le diameter

# Classe pour stocker les vitesses et statistiques
class velocities:
    def __init__(self):
        self.vy = []
        self.dx = []
        self.sizeBlock = []
        self.diameter = []
        self.vMean = -1
        self.vMeanPerBlock = None
        self.vStd = -1
        self.vStdPerBlock = None
        
    def computeMean(self):
        """Calcule la vitesse moyenne globale et par bulle."""
        self.vMeanPerBlock = [arr.mean() for arr in self.vy]
        self.vMean = np.mean(np.concatenate(self.vy))
        self.vStdPerBlock = [arr.std() for arr in self.vy]
        self.vStd = np.std(np.concatenate(self.vy))
        self.diameterMeanPerBlock = [arr.mean() for arr in self.diameter]
        self.diameterStdPerBlock = [arr.std() for arr in self.diameter]
        
    def convert2mm(self, mm_per_px):
        """Convertit toutes les vitesses et déplacements en millimètres."""
        self.vy_mm = [x * mm_per_px for x in self.vy]
        self.dx_mm = [x * mm_per_px for x in self.dx]
        self.vMean_mm = self.vMean * mm_per_px
        self.vMeanPerBubble_mm = np.array(self.vMeanPerBlock) * mm_per_px
        self.vStd_mm = self.vStd * mm_per_px
        self.vStdPerBubble_mm = np.array(self.vStdPerBlock) * mm_per_px
        self.diameter_mm = [x * mm_per_px for x in self.diameter]


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


def compute_speed_blocks(frames0, track_ids, labels, mergeFrames, contours, rich_df,
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
    
    df["merge"]=0
    for i in mergeFrames: 
        df["merge"] = df["merge"] + (df["frame"]>i).astype(int) # TODO verifier a partir de quelle frame prendre i, i-1, i-2 ???
    
    
    df["time"] = df["frame"] / fps
    df["block"] = (df["label"] != df["label"].shift()).cumsum() # block par label
    df["block"] = (df["block"]) + (df["merge"] != df["merge"].shift()).cumsum() # block par merge et par label

    for i in mergeFrames: 
        # NOTE on retire la frame ou a lieu le merge de l'analyse, est ce utile ??
        # Supprimer toutes les lignes où col1 est dans la liste
        # df = df[~df["col1"].isin(to_remove)]
        df = df[df["frame"] != i] # TODO verifier a partir de quelle frame prendre i, i-1, i-2 ???



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
        
        diameters = np.array([bubbleDiameter(fr, tid, rich_df)
                     for fr, tid in zip(group["frame"], group["track_id"])])

        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        t = group["time"].to_numpy()

        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        vy = - dy / dt  # origine en haut à gauche
        
        # NOTE normalement on ne devrait plus avoir de vitesse negative puisqu'on ne prend plus ce qui se passe au merge

        # On retire les vitesse negative
        dypos = dy[dy<0]
        if dypos.size <minPointForVelocity:
            continue
        dxpos = dx[dy<0]
        newt = t[1:]
        tpos = newt[dy<0]
        dtpos = np.diff(tpos)
        
        vypos = - dypos[1:]/dtpos

        # TODO ajouter le diametre
        if label == ATTACHED:
            attach_vel.vy.append(vypos)
            attach_vel.dx.append(dxpos)
            attach_vel.sizeBlock.append(vypos.size)
            attach_vel.diameter.append(diameters) # NOTE diameter a un element de plus que vy
        else:
            detach_vel.vy.append(vypos)
            detach_vel.dx.append(dxpos)
            detach_vel.sizeBlock.append(vypos.size)
            detach_vel.diameter.append(diameters)


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

    mergeFrame_arr = df_evol["mergeFrame"].apply(ast.literal_eval).to_list()

    # Création des objets velocities
    attach_vel = velocities()
    detach_vel = velocities()

    # Calcul des vitesses par bloc
    for idx, tid_evol in enumerate(tid_arr):
        frames0 = frames_arr[idx]
        labels = bubclass_arr[idx]
        mergeFrames = mergeFrame_arr[idx]
        track_ids = [x for x in tid_evol if x is not None]
        # TODO smooth labels
        compute_speed_blocks(frames0, track_ids, labels, mergeFrames, contours, rich_df,
                             attach_vel, detach_vel, minPointForVelocity, fps)

    # TODO ajouter une condition si dx trop grand

    # Calcul des statistiques globales
    detach_vel.computeMean()
    attach_vel.computeMean()

    attach_vel.convert2mm(mm_per_px)
    detach_vel.convert2mm(mm_per_px)
    
    return attach_vel, detach_vel

