import json
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict

# ------------------------
# PARAMÈTRES
# ------------------------
IMAGE_SHAPE = (1024, 1024)  # Dimensions des images (hauteur, largeur)
W_PREV = 3  # Fenêtre temporelle : nombre de frames à regarder en arrière
W_NEXT = 3  # Fenêtre temporelle : nombre de frames à regarder en avant
DILATE_ITERS = 1  # Nombre d'itérations de dilatation pour les masques
KERNEL = np.ones((3, 3), np.uint8)  # Noyau pour les opérations morphologiques
OVERLAP_THRESH = 0.1  # Seuil minimum de chevauchement pour considérer une relation parent-enfant


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

def overlap_ratio(mask1, mask2):
    """Calcule le ratio de chevauchement entre deux masques"""
    inter = np.logical_and(mask1 > 0, mask2 > 0)  # Intersection des deux masques
    area1 = np.sum(mask1 > 0)  # Aire du premier masque
    # Retourne le ratio d'intersection par rapport à l'aire du premier masque TODO pk le 1er
    return np.sum(inter) / area1 if area1 > 0 else 0.0

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

def build_masks_and_index(json_path, csv_path, image_shape=IMAGE_SHAPE):
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
        
        track_id = int(row.iloc[0]['track_id'])  # Récupère le track_id
        mask = mask_from_contour(contour, image_shape)  # Crée le masque
        if np.sum(mask > 0) == 0:  # Vérifie que le masque n'est pas vide
            continue
        
        # Stocke le masque dans la structure indexée
        data_by_frame[frame][track_id] = mask
    
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
        current_track_id = list(data_by_frame[frame].keys())
        previous_track_id = list(data_by_frame[frame-1].keys())
        # on cherche les bulles qui disparraissent entre le previous et current
        bulleDisparue[frame] = []
        for x in previous_track_id:
            if x not in current_track_id:
                bulleDisparue[frame].append(x)
        # on cherche les bulles qui apparaissent entre le previous et current
        bulleApparue[frame] = []
        for x in current_track_id:
            if x not in previous_track_id:
                bulleApparue[frame].append(x)
    # print(bulleDisparue)
    # print(bulleApparue)
    return bulleDisparue, bulleApparue

def my_detect_fusion(json_path, csv_path, image_shape=IMAGE_SHAPE):
    """Détecte les fusions de bulles en analysant les chevauchements temporels
    Retourne: dict {new_track_id: {'parents': [parent_ids], 'frame': frame}}
    """
    # Construit l'index des masques par frame
    data_by_frame = build_masks_and_index(json_path, csv_path, image_shape)
    fusion_map = {}  # Stocke les résultats des fusions détectées
    bulleDisparue, bulleApparue = bulle_changement(data_by_frame)
    frames = sorted(data_by_frame.keys())  # Liste triée des frames disponibles

    parentsDict = defaultdict(dict) # frame->new_tid->parents
    for i, frame in enumerate(frames[2:]):
        for new_tid in bulleApparue[frame]:
            child_mask = data_by_frame[frame][new_tid]
            parentsDict[frame][new_tid] = []
            for dis_tid in bulleDisparue[(frame-1):(frame+2)]:
                parent_mask = data_by_frame[frame][dis_tid]  # Masque du parent potentiel
                ratio = overlap_ratio(parent_mask, child_mask)  # Calcule le chevauchement
                if ratio > OVERLAP_THRESH:  # Si le chevauchement dépasse le seuil
                    parentsDict[frame][new_tid].append(dis_tid)  # Ajoute aux parents  
            print(data_by_frame[frame][new_tid])
            break


def detect_fusions(json_path, csv_path, image_shape=IMAGE_SHAPE):
    """
    Détecte les fusions de bulles en analysant les chevauchements temporels
    Retourne: dict {new_track_id: {'parents': [parent_ids], 'frame': frame}}
    """
    # Construit l'index des masques par frame
    data_by_frame = build_masks_and_index(json_path, csv_path, image_shape)
    fusion_map = {}  # Stocke les résultats des fusions détectées

    frames = sorted(data_by_frame.keys())  # Liste triée des frames disponibles


    # Parcourt chaque frame
    for i, frame in enumerate(frames):
        current_bubbles = data_by_frame[frame]  # Bulles de la frame courante
        
        # Récupère les bulles de la frame suivante (si elle existe)
        if i < len(frames) - 1:
            next_bubbles = data_by_frame[frame + 1]
        
        # Sélectionne les frames précédentes dans la fenêtre temporelle
        prev_frames = [f for f in frames if f < frame and f >= frame - W_PREV]
        
        # Agrège toutes les bulles des frames précédentes
        previous_bubbles = {}
        for f in prev_frames:
            previous_bubbles.update(data_by_frame[f])

        # Identifie les nouvelles bulles (celles qui n'existaient pas dans les frames précédentes)
        new_bubbles = [tid for tid in current_bubbles if tid not in previous_bubbles]
        
        # Identifie les bulles disparues (celles qui existaient avant mais plus maintenant OU pas dans la frame suivante)
        disappeared_bubbles = [tid for tid in previous_bubbles if (tid not in current_bubbles or tid not in next_bubbles)]

        # Pour chaque nouvelle bulle détectée
        for new_tid in new_bubbles:
            # Crée un masque agrégé sur plusieurs frames futures
            aggregated_mask = np.zeros(image_shape, dtype=np.uint8)
            for f_next in frames:
                if f_next < frame:  # Ignore les frames passées
                    continue
                if f_next > frame + W_NEXT:  # Stop si on dépasse la fenêtre
                    break
                if new_tid in data_by_frame[f_next]:  # Si la bulle existe dans cette frame future
                    # Ajoute son masque au masque agrégé
                    aggregated_mask = np.logical_or(aggregated_mask, data_by_frame[f_next][new_tid] > 0)
            
            aggregated_mask = (aggregated_mask.astype(np.uint8) * 255)  # Convertit en masque binaire

            # Cherche les parents potentiels parmi les bulles disparues
            parents = []
            for parent_tid in disappeared_bubbles:
                parent_mask = previous_bubbles[parent_tid]  # Masque du parent potentiel
                ratio = overlap_ratio(parent_mask, aggregated_mask)  # Calcule le chevauchement
                if ratio > OVERLAP_THRESH:  # Si le chevauchement dépasse le seuil
                    parents.append(parent_tid)  # Ajoute aux parents

            # Si au moins un parent trouvé, enregistre la fusion
            if parents:
                fusion_map[new_tid] = {'parents': parents, 'frame': frame}

    return fusion_map

# ------------------------
# EXÉCUTION
# ------------------------
# Dossier ou sont sauvegarde les donnee apres le modele
dataFolder = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\My_output\SaveData2"
extension = "Test6"
contourFile = dataFolder + "/contours_" + extension +".json"  # Fichier des contours
richFile = dataFolder + "/rich_" + extension +".csv"  # Fichier de tracking

# Lance la détection des fusions
fusions = detect_fusions(contourFile, richFile, IMAGE_SHAPE)
my_detect_fusion(contourFile, richFile, IMAGE_SHAPE)
# Affiche les résultats (seulement les fusions avec exactement 2 parents)
print("Fusions détectées :")
for new_bubble, info in fusions.items():
    if len(info['parents']) == 2:  # Filtre seulement les fusions à 2 parents
        print(f"Frame {info['frame']} : child {new_bubble} ← parents {info['parents']}")