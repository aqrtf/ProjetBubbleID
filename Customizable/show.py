from csteDef import *
import os, json, ast
import numpy as np
import pandas as pd
import cv2, glob

savefolder = r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet\ProjetBubbleID\My_output\Test0"
extension = "TestGUI"

# Chemins vers les fichiers
rich_csv = os.path.join(savefolder, f"rich_{extension}.csv")
evolution_csv = os.path.join(savefolder, f"evolutionID_{extension}.csv")
scale_path = os.path.join(savefolder, f"scale_{extension}.json")
contours_path = os.path.join(savefolder, f"contours_{extension}.json")
imagefolder = os.path.join(savefolder, "trimImages_"+ extension)
output_path = os.path.join(savefolder, f"track2_{extension}.avi")

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
frames_arr = [[j for j, val in enumerate(row) if val is not None] for row in tid_arr]

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
    
# Génère une palette de n couleurs différentes mais reproductibles
n_bubbles = len(tid_arr)
rng = np.random.default_rng(42)
colorpalette = []
for i in range(n_bubbles):
    # Évite les couleurs trop sombres
    color = tuple(int(c) for c in rng.integers(50, 255, size=3))
    colorpalette.append(color)

# Récupère la liste des images
images = sorted(glob.glob(os.path.join(imagefolder, "*.jpg")))

if not images:
    raise("Aucune image trouvée !")
    

# Lit la première image pour obtenir les dimensions
frame = cv2.imread(images[0])
height, width = frame.shape[:2]

# Crée le writer vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20, (width, height))

for idx_frame, image_path in enumerate(images):
    # Charge l'image
    img = cv2.imread(image_path)
    
    if img is None:
        continue
        
    # --- ICI VOUS AJOUTEZ VOS DESSINS ---
    bubblesID = [(irow, ligne[idx_frame]) for irow, ligne in enumerate(tid_arr) if ligne[idx_frame] is not None]
    for irow, bubble in bubblesID:
        bubbleInfo = rich_df[(rich_df["frame0"]==idx_frame) & (rich_df["track_id"] == bubble)]
        detinFrame = bubbleInfo["det_in_frame"].iloc[0]
        clef = str(idx_frame+1) + "_" + str(detinFrame)
        points = np.array(contours[clef])
        classID = bubclass_arr[irow][idx_frame]
        if classID == ATTACHED:
            className = 'A'
        elif classID == DETACHED:
            className = 'D'
        elif classID == UNKNOWN:
            className = 'U'

        cv2.polylines(img, [points.reshape(-1, 1, 2)], True, colorpalette[irow], 2)
        cx = int(bubbleInfo["cx_px"].iloc[0])
        cy = int(bubbleInfo["cy_px"].iloc[0])
        cv2.putText(img, className+str(irow), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colorpalette[irow], 2, cv2.LINE_AA)

    # Écrit la frame dans la vidéo
    out.write(img)
    
    print(f"Traitement de l'image {idx_frame+1}/{len(images)}")

out.release()
print(f"Vidéo sauvegardée : {output_path}")
