# TODO il ne faut pas que l'intersection des parents soit trop importante


import json
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict

# --------------------------PARAMÈTRES------------------------
IMAGE_SHAPE = (1024, 1024)  # Dimensions des images (hauteur, largeur)
DILATE_ITERS = 1  # Nombre d'itérations de dilatation pour les masques
KERNEL = np.ones((3, 3), np.uint8)  # Noyau pour les opérations morphologiques
OVERLAP_THRESH = 0.1  # Seuil minimum de chevauchement pour considérer une relation parent-enfant

# Pour ameliorer la robustesse on ne prends pas que le mask de la nouvelle bulle 
# a son apparition mais aussi sur les qq frames suivantes. En effet, le tracking 
# n'est pas toujours complet
POST_FUSION_FRAMES = 2  # Frames après fusion pour consolidation du masque

# Les bulles parents ne disparraissent pas toujours juste au moment de la fusion
# Parfois elles ne sont plus detecte plusieurs frames avant
N_FRAMES_PREVIOUS_DISAPPEAR = 3
# Et parfois elle restent detectees sur une ou deux frame avec le child
N_FRAMES_POST_DISAPPEAR = 2

score_thres = 0.7 #Minimum prediction score to have to consider a bubble

# -----------------------------DATA------------------------------------
# Dossier ou sont sauvegarde les donnee apres le modele
dataFolder = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\My_output\SaveData3"
extension = "T113_2_40V_2"
contourFile = dataFolder + "/contours_" + extension +".json"  # Fichier des contours
richFile = dataFolder + "/rich_" + extension +".csv"  # Fichier de tracking

outputFilePath = dataFolder + "/fusionHistory_" + extension + ".txt"


# ------------------------
# UTILITAIRES
# ------------------------
def mask_from_contour(contour, shape):
    """Convertit un contour en masque binaire"""
    mask = np.zeros(shape, dtype=np.uint8)  # Crée un masque vide
    if len(contour) == 0:  # Si le contour est vide, retourne un masque vide
        return mask
    pts = np.array(contour, dtype=np.int32)  # Convertit les points en array numpy
    cv2.fillPoly(mask, [pts], 255)  # Remplit le polygone défini par le contour
    if DILATE_ITERS > 0:  # Applique une dilatation si demandée
        mask = cv2.dilate(mask, KERNEL, iterations=DILATE_ITERS)
    return mask

def mask_area(mask):
    return np.sum(mask>0)

def overlap_ratio(mask1, mask2):
    """Calcule le ratio de chevauchement entre deux masques"""
    inter = np.logical_and(mask1 > 0, mask2 > 0)  # Intersection des deux masques
    interArea = mask_area(inter)
    area1 = mask_area(mask1)  # Aire du premier masque
    area2 = mask_area(mask2)
    childArea = area1 if area1>area2 else area2 
    # Retourne le ratio d'intersection par rapport à l'aire du premier masque 
    return interArea / childArea if childArea > 0 else 0.0

# ------------------------
# CHARGEMENT DES DONNÉES
# ------------------------
def load_json_contours(json_path):
    """Charge les contours depuis le fichier JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)  # Charge tout le fichier JSON
    
    parsed = {}  # Dictionnaire pour stocker les contours parsés
    for key, contour in data.items():
        # Parse la clé "frame_det" en frame et det_in_frame
        frame_str, det_str = key.split('_')
        frame = int(frame_str)
        det_in_frame = int(det_str)
        parsed[(frame, det_in_frame)] = contour  # Stocke avec clé (frame, detection)
    
    return parsed

def build_masks_and_index(json_path, csv_path, image_shape=IMAGE_SHAPE, score_thres = 0.7):
    """
    Construit un index des masques organisé par frame et track_id
    Retourne: dict[frame][track_id] = mask
    """
    contours = load_json_contours(json_path)  # Charge tous les contours
    df = pd.read_csv(csv_path)  # Charge le CSV de tracking

    data_by_frame = defaultdict(dict)  # Structure: frame -> track_id -> mask
    
    for (frame, det_in_frame), contour in contours.items():
        # Trouve la ligne correspondante dans le CSV
        row = df[(df['frame'] == frame) & (df['det_in_frame'] == det_in_frame)]
        if row.empty:  # Si pas de correspondance, on ignore
            continue
        # On ne considere pas les prediction qui ont un score trop faible
        for i in row["score"]:
            if i < float(score_thres):  # <<<<<<<<<<<<<< filtro
                continue

            track_id = int(row.iloc[0]['track_id'])  # Récupère le track_id
            mask = mask_from_contour(contour, image_shape)  # Crée le masque
            if np.sum(mask > 0) == 0:  # Vérifie que le masque n'est pas vide
                continue
            
            # Stocke le masque dans la structure indexée
            data_by_frame[frame][track_id] = mask 
            # NOTE si les deux scores sont valables, on ecrase le premier, 
            # ca ne devrait pas poser pb car si les deux ont le meme tid
            # ils ont des mask similaire mais un etat different

    return data_by_frame

# ------------------------
# FONCTION PRINCIPALE
# ------------------------

def bulle_changement(data_by_frame):
    """Sort 2 dictionnaires qui contiennent les track_id des bulles qui on disparue
    apparue pour chacune des frames"""
    frames = sorted(data_by_frame.keys())  # Liste triée des frames disponibles
    # Parcourt chaque frame
    bulleDisparue = {}
    bulleApparue = {}
    for i, frame in enumerate(frames):
        # WARNING si les frame ne sont pas successive il peux y avoir un probleme
        if frame == 1: #La frame 1 ne nous interresse pas
            continue
        previous_frame = frame - 1
        if previous_frame not in data_by_frame:  # TODO Gérer si frames manquantes
            print("no previous frame find at frame ", frame)
            continue

        current_track_ids = set(data_by_frame[frame].keys())
        previous_track_ids = set(data_by_frame[previous_frame].keys())
        
        # Bulles qui disparaissent entre previous et current
        bulleDisparue[frame] = list(previous_track_ids - current_track_ids)
        
        # Bulles qui apparaissent entre previous et current  
        bulleApparue[frame] = list(current_track_ids - previous_track_ids)
    # print(bulleDisparue)
    # print(bulleApparue)
    return bulleDisparue, bulleApparue


def my_detect_fusion(json_path, csv_path, image_shape=IMAGE_SHAPE, score_thresh=.7):
    """Détecte les fusions de bulles en analysant les chevauchements temporels
    Retourne: dict {new_track_id: {'parents': [parent_ids], 'frame': frame}}
    """
    # Construit l'index des masques par frame
    data_by_frame = build_masks_and_index(json_path, csv_path, image_shape, score_thresh)
    bulleDisparue, bulleApparue = bulle_changement(data_by_frame)
    frames = sorted(data_by_frame.keys())  # Liste triée des frames disponibles

    parentsDict = defaultdict(dict) # frame->new_tid-> (parents_id, frame_parents)
    for frame in frames:
        # Vérifier s'il y a des nouvelles bulles sur cette frame
        if frame not in bulleApparue or not bulleApparue[frame]:
            continue
            
        print(f"Frame {frame}:\n\t{len(bulleApparue[frame])} new bubbles: {bulleApparue[frame]}")
        print(f"\tBubbles disappear btw frame {frame-N_FRAMES_PREVIOUS_DISAPPEAR} and {frame+N_FRAMES_POST_DISAPPEAR}:")
        
        for new_tid in bulleApparue[frame]:
            if new_tid not in data_by_frame[frame]:
                continue
                
            child_mask = data_by_frame[frame][new_tid]
            # Pour ameliorer la robustesse on ne prends pas que le mask de la nouvelle bulle 
            # a son apparition mais aussi sur les qq frames suivantes. En effet, le tracking 
            # n'est pas toujours complet
            for i_frame in range(frame+1, frame+1+POST_FUSION_FRAMES):
                # Vérifier que la bulle existe dans les données
                if (i_frame in data_by_frame and 
                    new_tid in data_by_frame[i_frame]): 

                    child_mask = child_mask + data_by_frame[i_frame][new_tid]


            parentsDict[frame][new_tid] = []
            
            # Chercher les parents dans les frames autour (frame-1, frame, frame+1)
            for search_frame in range(frame-N_FRAMES_PREVIOUS_DISAPPEAR, frame+1+N_FRAMES_POST_DISAPPEAR):
                if search_frame in bulleDisparue and bulleDisparue[search_frame]:
                    print(f"\t\tFrame {search_frame}: {bulleDisparue[search_frame]}")
                    for dis_tid in bulleDisparue[search_frame]:
                        # Vérifier que le parent existe dans les données
                        if dis_tid == new_tid: # un parent ne peut pas etre son propre fils
                            continue
                        if (search_frame-1 in data_by_frame and 
                            dis_tid in data_by_frame[search_frame-1]): 
                            
                            parent_mask = data_by_frame[search_frame-1][dis_tid] 
                            if mask_area(child_mask) <= mask_area(parent_mask): # la nouvelle bulle doit etre plus grandes que ses parents
                                continue

                            ratio = overlap_ratio(parent_mask, child_mask)
                            
                            if ratio > OVERLAP_THRESH:
                                parentsDict[frame][new_tid].append((dis_tid, search_frame-1))
                                print(f"\t\t\tParent trouvé: {dis_tid} (frame {search_frame}) -> {new_tid}, ratio: {ratio:.3f}")
    
    # NETTOYAGE : retirer les entrées vides et celles avec moins de 2 parents
    parentsDict_clean = {}
    for frame, tracks in parentsDict.items():
        # Filtrer pour garder seulement les tracks avec au moins 2 parents
        tracks_with_min_2_parents = {
            track_id: parents 
            for track_id, parents in tracks.items() 
            if len(parents) >= 2
        }
        # Ne garder la frame que si elle contient au moins une track valide
        if tracks_with_min_2_parents:
            parentsDict_clean[frame] = tracks_with_min_2_parents

            if mask_area(np.logical_and(data_by_frame[parentsDict_clean[frame][1]][parentsDict_clean[frame][0]])):
                pass
    
    print("\nRésultats des fusions détectées:")
    for frame, tracks in parentsDict_clean.items():
        for new_tid, parents_tuple in tracks.items():
            print(f"Frame {frame:3d}: {new_tid:3d} <- {parents_tuple}")
            # print(f"Frame {frame:3d}: {new_tid:3d} ← [", end="")
            # for id, _ in parents_tuple:
            #     print(id, end="," )
            # print("]")
    
    return parentsDict_clean

def exportData(parentsDict, outputFile):
    with open(outputFile, 'w') as file:
        file.write(f"{len(parentsDict)} fusions detect:\n")
        for frame, tracks in parentsDict.items():
            for new_tid, parents_tuple in tracks.items():
                file.write(f"Frame {frame:3d}: {new_tid:3d} <- {parents_tuple}\n") # !!!!! on affiche pas la bonne chose

# ------------------------
# EXÉCUTION
# ------------------------


# Lance la détection des fusions
parentsDict = my_detect_fusion(contourFile, richFile, IMAGE_SHAPE, score_thres)
exportData(parentsDict, outputFilePath)
