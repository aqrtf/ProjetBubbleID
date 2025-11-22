import json, os, copy
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass

# ------------------------
# UTILITAIRES
# ------------------------
def mask_from_contour(contour, shape, dilate_iters):
    """Convertit un contour en masque binaire"""
    KERNEL=np.ones((3, 3), np.uint8)
    mask = np.zeros(shape, dtype=np.uint8)  # Crée un masque vide
    if len(contour) == 0:  # Si le contour est vide, retourne un masque vide
        return mask
    pts = np.array(contour, dtype=np.int32)  # Convertit les points en array numpy
    cv2.fillPoly(mask, [pts], 255)  # Remplit le polygone défini par le contour
    if dilate_iters > 0:  # Applique une dilatation si demandée
        mask = cv2.dilate(mask, KERNEL, iterations=dilate_iters)
    return mask

def mask_area(mask):
    return np.sum(mask>0)

def overlap_ratio(mask1, mask2, reference):
    """Calcule le ratio de chevauchement entre deux masques"""
    inter = np.logical_and(mask1 > 0, mask2 > 0)  # Intersection des deux masques
    interArea = mask_area(inter)
    area1 = mask_area(mask1)  # Aire du premier masque
    area2 = mask_area(mask2)
    if reference == "biggest":
        refArea = area1 if area1>area2 else area2
    elif reference == "smallest":
        refArea = area1 if area1<area2 else area2 
    else:
        raise("reference must be either 'biggest' or 'smallest'")

    # Retourne le ratio d'intersection par rapport à l'aire de la reference
    return interArea / refArea if refArea > 0 else 0.0

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

def build_masks_and_index(json_path, csv_path, image_shape, score_thres, dilate_iters):
    """
    Construit un index des masques binaires organisé par frame et track_id.
    
    Charge les contours depuis le fichier JSON et les données de tracking depuis le CSV,
    puis crée des masques binaires pour chaque bulle détectée. Les masques sont organisés
    dans un dictionnaire hiérarchique permettant un accès rapide par frame et track_id.
    
    Args:
        json_path (str): Chemin vers le fichier JSON contenant les contours des bulles
        csv_path (str): Chemin vers le fichier CSV contenant les données de tracking
        image_shape (tuple): Dimensions des images (hauteur, largeur)
        score_thres (float): Seuil de score minimum pour inclure une détection
    
    Returns:
        dict: Structure nested {frame: {track_id: mask}} où chaque mask est un array numpy binaire
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
            mask = mask_from_contour(contour, image_shape, dilate_iters)  # Crée le masque
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
    """
    Identifie les bulles qui apparaissent et disparaissent entre frames successives.
    
    Analyse les transitions entre frames consécutives pour détecter les changements
    dans la présence des track_id. Retourne deux dictionnaires listant les bulles
    qui disparaissent et celles qui apparaissent à chaque frame.
    
    Args:
        data_by_frame (dict): Structure {frame: {track_id: mask}} issue de build_masks_and_index()
        
    Returns:
        tuple: (bulleDisparue, bulleApparue) où:
            - bulleDisparue (dict): {frame: [track_id]} liste des bulles disparues à cette frame
            - bulleApparue (dict): {frame: [track_id]} liste des nouvelles bulles à cette frame
            
    Note:
        La frame 1 est ignorée car c'est la première frame de référence.
        Les frames doivent être consécutives pour une détection correcte.
    """
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


def filtrer_parents_par_intersection(parents_ids, frame_parents, masques_dict, min_overlap_same):
    """
    Filtre les parents potentiels en éliminant ceux qui ont un chevauchement excessif.
    
    Calcule les intersections deux à deux entre les masques des parents candidats et
    retire ceux qui présentent un chevauchement supérieur au seuil, indiquant qu'ils
    pourraient être des détections multiples de la même bulle ou des bulles trop proches.
    
    Args:
        parents_ids (list): Liste des track_id des parents candidats
        frame_parents (int): Numéro de frame où chercher les masques des parents
        masques_dict (dict): Dictionnaire des masques {frame: {track_id: mask}}
        min_overlap_same (float): Seuil d'overlap au-delà duquel deux parents sont considérés comme duplicats
        
    Returns:
        list: Liste filtrée des track_id des parents après suppression des doublons spatiaux
    """
    masques_parents = []
    
    # Récupérer les masques valides
    for parent_id, frame_parent in zip(parents_ids, frame_parents):
        if frame_parent in masques_dict and parent_id in masques_dict[frame_parent]:
            masque = masques_dict[frame_parent][parent_id]
            masques_parents.append(masque)
    
    # Matrice d'overlap entre toutes les paires
    n = len(masques_parents)
    iou_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            iou = overlap_ratio(masques_parents[i], masques_parents[j], reference="smallest")
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    
    # Identifier les parents à retirer (overlap trop élevé)
    a_retirer = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            if iou_matrix[i, j] > min_overlap_same:
                # Retirer celui qui a la plus petite surface
                surface_i = np.sum(masques_parents[i]) // 255
                surface_j = np.sum(masques_parents[j]) // 255
                
                if surface_i < surface_j:
                    a_retirer.add(i)
                else:
                    a_retirer.add(j)

    # Supprimer en partant du plus grand index pour éviter les décalages
    for i in sorted(a_retirer, reverse=True):
        del parents_ids[i]

    return parents_ids


def my_detect_fusion(json_path, csv_path, outputFile, N_FRAMES_PREVIOUS_DISAPPEAR, N_FRAMES_POST_DISAPPEAR, POST_FUSION_FRAMES, OVERLAP_THRESH, score_thresh=.7, min_overlap_same=.7, image_shape=(1024,1024), dilate_iters=0):
    """
    Détecte les fusions de bulles en analysant les chevauchements temporels entre frames.
    
    Identifie les événements de fusion où plusieurs bulles disparues se combinent pour former
    une nouvelle bulle. Utilise une fenêtre temporelle glissante pour rechercher les parents
    potentiels parmi les bulles disparues récentes et valide les fusions par calcul d'overlap
    spatial entre les masques.
    
    Args:
        json_path (str): Chemin vers le fichier JSON des contours
        csv_path (str): Chemin vers le fichier CSV de tracking
        outputFile (file): File object ouvert pour l'écriture des logs de détection
        N_FRAMES_PREVIOUS_DISAPPEAR (int): Fenêtre temporelle pour rechercher les bulles disparues (frames avant)
        N_FRAMES_POST_DISAPPEAR (int): Fenêtre temporelle pour rechercher les bulles disparues (frames après) 
        POST_FUSION_FRAMES (int): Nombre de frames après l'apparition pour consolider le masque enfant
        OVERLAP_THRESH (float): Seuil d'overlap minimum pour valider une relation parent-enfant
        score_thresh (float): Seuil de score pour filtrer les détections
        min_overlap_same (float): Seuil pour éviter les doublons entre parents
        image_shape (tuple): Dimensions des images pour créer les masques
        
    Returns:
        dict: {frame: {new_track_id: [parent_ids]}} Dictionnaire des fusions détectées
    """
    # Construit l'index des masques par frame
    data_by_frame = build_masks_and_index(json_path, csv_path, image_shape, score_thresh, dilate_iters)
    bulleDisparue, bulleApparue = bulle_changement(data_by_frame)
    frames = sorted(data_by_frame.keys())  # Liste triée des frames disponibles

    @dataclass
    class ParentInfo:
        parent_id: int
        frame_parent: int

    parentsDict = defaultdict(lambda: defaultdict(list)) # frame->new_tid-> list de ParentInfo

    for frame in frames:
        # Vérifier s'il y a des nouvelles bulles sur cette frame
        if frame not in bulleApparue or not bulleApparue[frame]:
            continue
            
        outputFile.write(f"Frame {frame}:\n\t{len(bulleApparue[frame])} new bubbles: {bulleApparue[frame]}\n")
        outputFile.write(f"\tBubbles disappear btw frame {frame-N_FRAMES_PREVIOUS_DISAPPEAR} and {frame+N_FRAMES_POST_DISAPPEAR}:\n")
        
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
            
            # Chercher les parents dans les frames autour (frame-1, frame, frame+1)
            for search_frame in range(frame-N_FRAMES_PREVIOUS_DISAPPEAR, frame+1+N_FRAMES_POST_DISAPPEAR):
                if search_frame in bulleDisparue and bulleDisparue[search_frame]:
                    outputFile.write(f"\t\tFrame {search_frame}: {bulleDisparue[search_frame]}\n")
                    for dis_tid in bulleDisparue[search_frame]:
                        # Vérifier que le parent existe dans les données
                        if dis_tid == new_tid: # un parent ne peut pas etre son propre fils
                            continue
                        if (search_frame-1 in data_by_frame and 
                            dis_tid in data_by_frame[search_frame-1]): 
                            
                            parent_mask = data_by_frame[search_frame-1][dis_tid] 
                            if mask_area(child_mask) <= mask_area(parent_mask): # la nouvelle bulle doit etre plus grandes que ses parents
                                continue

                            ratio = overlap_ratio(parent_mask, child_mask, reference='biggest')
                            
                            if ratio > OVERLAP_THRESH:
                                parentsDict[frame][new_tid].append(ParentInfo(parent_id=dis_tid, frame_parent=search_frame-1))
                                outputFile.write(f"\t\t\tFound parent: {dis_tid} (frame {search_frame}) -> {new_tid}, ratio: {ratio:.3f}\n")

    outputFile.write("##########################################################\n")
    outputFile.write(f"Results before cleaning: {len(parentsDict)} fusions detect:\n")
    for frame, tracks in parentsDict.items():
        for new_tid, parents in tracks.items():
            outputFile.write(f"\tFrame {frame:3d}: {new_tid:3d} <- {[info.parent_id for info in parents]}\n")


    # NETTOYAGE : retirer les entrées vides et celles avec moins de 2 parents
    outputFile.write("##########################################################\nCleaning:\n")
    parentsDict_clean = {}
    parentsDict_clean2 = defaultdict(dict)

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

            # Retirer les parents s'ils sont trop proche l'un de l'autre 
            for new_tid, parents_list in parentsDict_clean[frame].items():
                parents_ids = [info.parent_id for info in parents_list]
                frames_parents = [info.frame_parent for info in parents_list]
                
                # Appliquer le filtrage
                parents_filtres = filtrer_parents_par_intersection(
                    list(parents_ids), 
                    list(frames_parents), 
                    data_by_frame, 
                    min_overlap_same=min_overlap_same
                )
                
                # Ne garder que si on a au moins 2 parents après filtrage
                if len(parents_filtres) >= 2:
                    parentsDict_clean2[frame][new_tid] = parents_filtres

    # Nettoyage final : retirer les frames vides dans parentsDict_clean2
    parentsDict_clean2 = {
        frame: tracks 
        for frame, tracks in parentsDict_clean2.items() 
        if tracks  # Garde seulement les frames avec au moins une track
    }

    print("\nRésultats des fusions détectées:")
    outputFile.write("##########################################################\n")
    outputFile.write(f"Results: {len(parentsDict_clean2)} fusions detect:\n")
    for frame, tracks in parentsDict_clean2.items():
        for new_tid, parents in tracks.items():
            print(f"Frame {frame:3d}: {new_tid:3d} <- {parents}")
            outputFile.write(f"\tFrame {frame:3d}: {new_tid:3d} <- {parents}\n")

    
    return parentsDict_clean2


def track_id_changes(json_path, csv_path, outputFile, N_FRAMES_PREVIOUS_DISAPPEAR, score_thresh, MIN_OVERLAP_SAME, existing_fusions=None, image_shape=(1024,1024), dilate_iters=0):
    """
    Détecte les changements simples de track_id où une bulle conserve sa position mais change d'identifiant.
    
    Identifie les cas où une bulle disparaît et réapparaît avec un nouveau track_id sans fusion,
    en excluant les événements déjà détectés comme fusions. Utilise des critères stricts d'overlap
    spatial pour s'assurer qu'il s'agit bien de la même bulle physique.
    
    Args:
        json_path (str): Chemin vers le fichier JSON des contours
        csv_path (str): Chemin vers le fichier CSV de tracking
        outputFile (file): File object ouvert pour l'écriture des logs de détection
        N_FRAMES_PREVIOUS_DISAPPEAR (int): Fenêtre temporelle pour rechercher les bulles disparues (frames avant seulement)
        score_thresh (float): Seuil de score pour filtrer les détections
        MIN_OVERLAP_SAME (float): Seuil d'overlap élevé pour confirmer qu'il s'agit de la même bulle
        existing_fusions (dict, optional): Dictionnaire des fusions déjà détectées à exclure
        image_shape (tuple): Dimensions des images pour créer les masques
        
    Returns:
        list (nx3): Chaque ligne contient [frame, new_tid, parent_tid] 
        
    Note:
        Contrairement aux fusions, seules les frames antérieures sont considérées pour les parents
        et un seul parent est autorisé par changement de track_id.
    """
    # Construit l'index des masques par frame
    data_by_frame = build_masks_and_index(json_path, csv_path, image_shape, score_thresh, dilate_iters)
    bulleDisparue, bulleApparue = bulle_changement(data_by_frame)
    frames = sorted(data_by_frame.keys())

    @dataclass
    class ParentInfo:
        parent_id: int
        frame_parent: int

    parentsDict = defaultdict(lambda: defaultdict(list)) # frame->new_tid-> list de ParentInfo

    for frame in frames:
        # Vérifier s'il y a des nouvelles bulles sur cette frame
        if frame not in bulleApparue or not bulleApparue[frame]:
            continue
            
        for new_tid in bulleApparue[frame]:
            if new_tid not in data_by_frame[frame]:
                continue
                
            child_mask = data_by_frame[frame][new_tid]
            
            # Chercher les parents dans les frames autour (seulement avant pour changement de track)
            for search_frame in range(frame-N_FRAMES_PREVIOUS_DISAPPEAR, frame+1):
                if search_frame in bulleDisparue and bulleDisparue[search_frame]:
                    for dis_tid in bulleDisparue[search_frame]:
                        # Vérifier que le parent existe dans les données
                        if dis_tid == new_tid: # un parent ne peut pas etre son propre fils
                            continue
                        if (search_frame-1 in data_by_frame and 
                            dis_tid in data_by_frame[search_frame-1]): 
                            
                            parent_mask = data_by_frame[search_frame-1][dis_tid]
                            
                            # Vérifier les overlaps
                            if (overlap_ratio(parent_mask, child_mask, reference='biggest') > MIN_OVERLAP_SAME and 
                                overlap_ratio(parent_mask, child_mask, reference='smallest') > MIN_OVERLAP_SAME):
                                
                                parentsDict[frame][new_tid].append(ParentInfo(parent_id=dis_tid, frame_parent=search_frame-1))

    outputFile.write("\n##########################################################\n")
    outputFile.write("TRACK ID CHANGEMENTS\n")
    outputFile.write("##########################################################\n")
    outputFile.write(f"Results before cleaning: {len(parentsDict)} potential changes detected:\n")
    for frame, tracks in parentsDict.items():
        for new_tid, parents in tracks.items():
            outputFile.write(f"\tFrame {frame:3d}: {new_tid:3d} <- {[info.parent_id for info in parents]}\n")

    # NETTOYAGE ÉTAPE 1 : Retirer les doublons de parents et garder le plus tardif
    outputFile.write("----CLEANING----\n")
    outputFile.write("Step 1: Removing duplicate parents (keeping latest frame):\n")
    
    parentsDict_deduplicated = defaultdict(lambda: defaultdict(list))
    
    for frame, tracks in parentsDict.items():
        for new_tid, parents in tracks.items():
            # Grouper les parents par parent_id et garder celui avec la frame_parent la plus élevée
            parent_groups = {}
            for parent_info in parents:
                parent_id = parent_info.parent_id
                # Si on n'a pas encore ce parent, ou si on a une frame plus récente, on met à jour
                if (parent_id not in parent_groups or 
                    parent_info.frame_parent > parent_groups[parent_id].frame_parent):
                    parent_groups[parent_id] = parent_info
            
            # Convertir le dictionnaire en liste
            unique_parents = list(parent_groups.values())
            parentsDict_deduplicated[frame][new_tid] = unique_parents
            
            # Log si des doublons ont été retirés
            if len(unique_parents) < len(parents):
                outputFile.write(f"\tFrame {frame}: {new_tid} - removed duplicates, kept {len(unique_parents)} parents from {len(parents)}\n")

    # NETTOYAGE ÉTAPE 2 : Vérifier qu'il n'y a pas 2 parents différents (ce serait un merge)
    outputFile.write("Step 2: Checking for multiple different parents (should be merges):\n")
    
    parentsDict_single_parent = {}
    removed_due_to_multiple_parents = []
    
    for frame, tracks in parentsDict_deduplicated.items():
        valid_tracks = {}
        for new_tid, parents in tracks.items():
            if len(parents) == 1:
                # Cas idéal : exactement 1 parent
                valid_tracks[new_tid] = parents
            elif len(parents) > 1:
                # Plusieurs parents différents → c'est probablement une fusion
                removed_due_to_multiple_parents.append((frame, new_tid, [p.parent_id for p in parents]))
                outputFile.write(f"\tREMOVED - Frame {frame}: {new_tid} has {len(parents)} different parents: {[p.parent_id for p in parents]} (should be a merge)\n")
            # Si 0 parents, on ignore naturellement
        
        if valid_tracks:
            parentsDict_single_parent[frame] = valid_tracks

    # NETTOYAGE ÉTAPE 3 : Exclure les cas où une fusion a été détectée
    outputFile.write("Step 3: Excluding cases where fusion was detected:\n")

    # EXCLURE les fusions si provided
    parentsDict_final = {}
    if existing_fusions is not None:
        # Convertir existing_fusions en set de tuples (frame, track_id)
        fusion_set = set()
        if isinstance(existing_fusions, dict):
            for frame, tracks in existing_fusions.items():
                for track_id in tracks.keys():
                    fusion_set.add((frame, track_id))
        elif isinstance(existing_fusions, set):
            fusion_set = existing_fusions
        
        # Filtrer
        for frame, tracks in parentsDict_single_parent.items():
            filtered_tracks = {}
            for new_tid, parents in tracks.items():
                if (frame, new_tid) not in fusion_set:
                    filtered_tracks[new_tid] = parents
                else:
                    outputFile.write(f"\tFrame {frame}: {new_tid} comes from a merge\n")
            if filtered_tracks:
                parentsDict_final[frame] = filtered_tracks
    else:
        parentsDict_final = parentsDict_single_parent    # Log des exclusions


    print("\nRésultats des changements de track (après nettoyage):")
    outputFile.write("##########################################################\n")
    outputFile.write(f"Final results: {len(parentsDict_final)} track changes detected:\n")
    
    parentsList_return = []
    for frame, tracks in parentsDict_final.items():
        for new_tid, parents in tracks.items():
            parent_info = parents[0]  # Un seul parent puisque filtré
            print(f"Frame {frame:3d}: {new_tid:3d} <- {parent_info.parent_id} (from frame {parent_info.frame_parent})")
            outputFile.write(f"\tFrame {frame:3d}: {new_tid:3d} <- {parent_info.parent_id} (from frame {parent_info.frame_parent})\n")
            parentsList_return.append([frame, new_tid, parent_info.parent_id])
            
    

    return parentsList_return


def exportData(fusionDict, changeIDList, changeIDList_all, savefolder, extension):
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
    out_csv = os.path.join(savefolder, f'fusionResult_{extension}.csv')
    df_fusion.to_csv(out_csv, index=False)

    df_changeID = pd.DataFrame(changeIDList, columns=["frame", "new_id", "old_id"])
    out_csv = os.path.join(savefolder, f'changeIDResult_{extension}.csv')
    df_changeID.to_csv(out_csv, index=False)

    df_changeID = pd.DataFrame(changeIDList_all, columns=["frame", "new_id", "old_id"])
    out_csv = os.path.join(savefolder, f'changeIDResultAll_{extension}.csv')
    df_changeID.to_csv(out_csv, index=False)



# ------------------------
# EXÉCUTION
# ------------------------
# --------------------------PARAMÈTRES------------------------
# IMAGE_SHAPE = (1024, 1024)  # Dimensions des images (hauteur, largeur)
# DILATE_ITERS = 1  # Nombre d'itérations de dilatation pour les masques
# KERNEL = np.ones((3, 3), np.uint8)  # Noyau pour les opérations morphologiques
# OVERLAP_THRESH = 0.1  # Seuil minimum de chevauchement pour considérer une relation parent-enfant

# # Pour ameliorer la robustesse on ne prends pas que le mask de la nouvelle bulle 
# # a son apparition mais aussi sur les qq frames suivantes. En effet, le tracking 
# # n'est pas toujours complet
# POST_FUSION_FRAMES = 2  # Frames après fusion pour consolidation du masque

# # Les bulles parents ne disparraissent pas toujours juste au moment de la fusion
# # Parfois elles ne sont plus detecte plusieurs frames avant
# N_FRAMES_PREVIOUS_DISAPPEAR = 3
# # Et parfois elle restent detectees sur une ou deux frame avec le child
# N_FRAMES_POST_DISAPPEAR = 2

# score_thres = 0.7 #Minimum prediction score to have to consider a bubble
# MIN_OVERLAP_SAME = 0.7 #minimum overlap btw two bubble to consider equal

# # -----------------------------DATA------------------------------------
# # Dossier ou sont sauvegarde les donnee apres le modele
# dataFolder = "My_output/Test6"
# extension = "Test6"

def findMerge(dataFolder, extension, score_thres=0.7, OVERLAP_THRESH=0.1,
                                    MIN_OVERLAP_SAME=0.7, POST_FUSION_FRAMES=2, N_FRAMES_PREVIOUS_DISAPPEAR=3, 
                                    N_FRAMES_POST_DISAPPEAR=2,
                                    IMAGE_SHAPE=(1024, 1024), DILATE_ITERS=1
                                    ):
    """
    Fonction principale pour détecter les fusions de bulles et les changements de track_id.
    
    Coordonne l'ensemble du processus de détection en chargeant les données, en appliquant
    les algorithmes de détection de fusions et de changements d'identité, et en exportant
    les résultats.
    
    Args:
        dataFolder (str): Chemin vers le dossier ou sont enregistre les rich et contours
        extension (str): Extension des donnees d'interet
        score_thres (float): Seuil de score minimum pour considérer une bulle
        OVERLAP_THRESH (float): Seuil minimum de chevauchement pour les relations parent-enfant
        MIN_OVERLAP_SAME (float): Seuil d'overlap pour considérer deux bulles comme identiques
        POST_FUSION_FRAMES (int): Frames après fusion pour consolidation du masque
        N_FRAMES_PREVIOUS_DISAPPEAR (int): Fenêtre temporelle pour les bulles disparues (frames avant)
        N_FRAMES_POST_DISAPPEAR (int): Fenêtre temporelle pour les bulles disparues (frames après)
        IMAGE_SHAPE (tuple): Dimensions des images (hauteur, largeur)
        DILATE_ITERS (int): Nombre d'itérations de dilatation pour les masques
        
    Returns:
        tuple: (fusionDict, changeIDList) où:
            - fusionDict (dict): Dictionnaire des fusions détectées
            - changeIDList (2D list): Liste des chgmt d'id avec par ligne [frame, new_id, old_id]
    """
    
    contourFile = os.path.join(dataFolder, f"contours_{extension}.json")
    richFile = os.path.join(dataFolder, f"rich_{extension}.csv")
    outputFileHistoryPath = os.path.join(dataFolder, f"fusionHistory_{extension}.csv")
    
    # Lance la détection des fusions
    with open(outputFileHistoryPath, 'w') as f:
        # Écriture des paramètres utilisés
        f.write("PARAMETERS:\n")
        f.write(f"\tIMAGE_SHAPE = {IMAGE_SHAPE}\n")
        f.write(f"\tDILATE_ITERS = {DILATE_ITERS}\n")
        f.write(f"\tOVERLAP_THRESH = {OVERLAP_THRESH}\n")
        f.write(f"\tPOST_FUSION_FRAMES = {POST_FUSION_FRAMES}\n")
        f.write(f"\tN_FRAMES_PREVIOUS_DISAPPEAR = {N_FRAMES_PREVIOUS_DISAPPEAR}\n")
        f.write(f"\tN_FRAMES_POST_DISAPPEAR = {N_FRAMES_POST_DISAPPEAR}\n")
        f.write(f"\tscore_thres = {score_thres}\n")
        f.write(f"\tMIN_OVERLAP_SAME = {MIN_OVERLAP_SAME}\n")
        f.write("="*60 + "\n")
        
        fusionDict = my_detect_fusion(contourFile,
                                      richFile,
                                      f,
                                      N_FRAMES_PREVIOUS_DISAPPEAR,
                                      N_FRAMES_POST_DISAPPEAR,
                                      POST_FUSION_FRAMES,
                                      OVERLAP_THRESH,
                                      score_thres,
                                      MIN_OVERLAP_SAME,
                                      IMAGE_SHAPE,
                                      dilate_iters=DILATE_ITERS)
        
        changeIDList = track_id_changes(contourFile,
                                        richFile,
                                        f,
                                        N_FRAMES_PREVIOUS_DISAPPEAR,
                                        score_thres, MIN_OVERLAP_SAME,
                                        existing_fusions=fusionDict,
                                        image_shape=IMAGE_SHAPE,
                                        dilate_iters=DILATE_ITERS)
    changeIDList_all = copy.deepcopy(changeIDList)
    # Si x->y puis y->x il ne faut pas garder le deuxieme puisque on veut changer tous les y en x dans le reste
    changeIDList_clean = []
    for idx in range(len(changeIDList)):
        if changeIDList[idx][1] == changeIDList[idx][2]: #old and new equal
            continue
        changeIDList_clean.append(changeIDList[idx])
        for j in range(idx+1, len(changeIDList)):
            if changeIDList[j][1] == changeIDList[idx][1]:
                changeIDList[j][1] = changeIDList[idx][2]
            if changeIDList[j][2] == changeIDList[idx][1]:
                changeIDList[j][2] = changeIDList[idx][2]
            
    #TODO il faut faire pareil pour les merges
      
    # Export des résultats finaux
    exportData(fusionDict, changeIDList_clean, changeIDList_all, dataFolder, extension)
    
    return fusionDict, changeIDList_clean

######################################################################################################

# dataFolder=r"My_output\SaveData3"   # Define the folder you want the data to save in
# extension="T113_2_60V_2" 
# # dataFolder = "My_output/Test6/"
# # extension = "Test6"
# findMerge("My_output/Test6", "Test6")