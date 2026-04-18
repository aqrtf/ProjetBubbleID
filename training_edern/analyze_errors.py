import os
import torch
import cv2
import numpy as np
import argparse
import glob

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Boxes, pairwise_iou

# Import from train_kfold to reuse constants and functions
from train_kfold import create_detectron2_dataset_from_labelme, MAPPING, CATEGORY_IDS, FINAL_LABELS

# --- Configuration ---
# ID de la classe "detached"
DETACHED_CLASS_ID = CATEGORY_IDS["detached"]
# Seuil de score pour considérer une prédiction
SCORE_THRESHOLD = 0.5
# Seuil IoU pour considérer une prédiction comme une "bonne" détection (True Positive)
IOU_THRESHOLD = 0.5
# ---------------------

def analyze_errors(model_path, config_path, dataset_path, output_dir):
    """
    Analyse les erreurs d'un modèle sur un jeu de données, en se concentrant sur la classe "detached".
    Sauvegarde des images de visualisation pour les Faux Positifs et les Faux Négatifs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Configuration du prédicteur ---")
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    predictor = DefaultPredictor(cfg)
    
    print("--- Chargement du jeu de données ---")
    # Utilise la même fonction que l'entraînement pour garantir la cohérence
    dataset_dicts, _, _ = create_detectron2_dataset_from_labelme(glob.glob(os.path.join(dataset_path, "*.json")))
    
    # Enregistrement temporaire pour les métadonnées
    dataset_name = "error_analysis_dataset"
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
    metadata = MetadataCatalog.get(dataset_name).set(thing_classes=FINAL_LABELS)

    print(f"Analyse de {len(dataset_dicts)} images...")
    
    # --- Initialisation des compteurs d'erreurs ---
    total_fp = 0
    total_fn = 0
    total_tp = 0

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        
        # --- Préparation des données Ground Truth (GT) et Prédictions (Pred) ---
        
        # Ground Truth pour la classe "detached"
        gt_annos = [ann for ann in d["annotations"] if ann["category_id"] == DETACHED_CLASS_ID]
        if not gt_annos:
            gt_boxes = Boxes(torch.empty((0, 4), device=outputs["instances"].pred_boxes.device))
        else:
            gt_boxes = Boxes([ann["bbox"] for ann in gt_annos])
            gt_boxes.tensor = gt_boxes.tensor.to(outputs["instances"].pred_boxes.device)

        # Prédictions pour la classe "detached"
        pred_instances = outputs["instances"]
        detached_preds_mask = pred_instances.pred_classes == DETACHED_CLASS_ID
        pred_boxes = pred_instances.pred_boxes[detached_preds_mask]
        
        # --- Calcul des correspondances et erreurs ---
        
        # S'il y a des prédictions ou des GT, on calcule l'IoU
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = pairwise_iou(pred_boxes, gt_boxes)
            
            # True Positives: prédictions qui matchent bien une GT
            matched_preds = iou_matrix.max(dim=1).values > IOU_THRESHOLD
            num_tp = matched_preds.sum().item()

            # GT qui sont bien matchées
            matched_gts = iou_matrix.max(dim=0).values > IOU_THRESHOLD
            
            # False Positives: prédictions qui ne matchent aucune GT
            fp_mask = ~matched_preds
            num_fp = fp_mask.sum().item()
            
            # False Negatives: GT qui ne sont matchées par aucune prédiction
            fn_mask = ~matched_gts
            num_fn = fn_mask.sum().item()
            
        else: # Cas simples
            fp_mask = torch.ones(len(pred_boxes), dtype=torch.bool) # Toutes les prédictions sont des FP s'il n'y a pas de GT
            fn_mask = torch.ones(len(gt_boxes), dtype=torch.bool)   # Toutes les GT sont des FN s'il n'y a pas de prédictions
            num_fp = fp_mask.sum().item()
            num_fn = fn_mask.sum().item()
            num_tp = 0

        total_fp += num_fp
        total_fn += num_fn
        total_tp += num_tp

        # --- Visualisation des erreurs ---
        
        if fp_mask.any() or fn_mask.any():
            # Créer une image de base pour la visualisation
            v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
            
            # Dessiner les Faux Négatifs (GT manquées) en ROUGE
            if fn_mask.any():
                fn_boxes_to_draw = gt_boxes[fn_mask]
                for box in fn_boxes_to_draw.tensor.cpu().numpy():
                    v.draw_box(box, edge_color="r", line_style="-")
                    v.draw_text("FN (missed)", tuple(box[:2]), color="r")

            # Dessiner les Faux Positifs (prédictions incorrectes) en JAUNE
            if fp_mask.any():
                fp_boxes_to_draw = pred_boxes[fp_mask]
                for box in fp_boxes_to_draw.tensor.cpu().numpy():
                    v.draw_box(box, edge_color="y", line_style="--")
                    v.draw_text("FP (wrong)", tuple(box[:2]), color="y")

            # Sauvegarder l'image
            vis_img = v.get_image()[:, :, ::-1]
            out_filename = os.path.join(output_dir, f"error_{os.path.basename(d['file_name'])}")
            cv2.imwrite(out_filename, vis_img)

    # --- Affichage du résumé final ---
    print("\n" + "="*40)
    print("--- Résumé de l'analyse quantitative ---")
    print(f"Classe analysée : '{FINAL_LABELS[DETACHED_CLASS_ID]}'")
    print(f"Seuil de score : {SCORE_THRESHOLD}, Seuil d'IoU : {IOU_THRESHOLD}")
    print("-"*40)
    print(f"Vrais Positifs (TP - bulles bien détectées) : {total_tp}")
    print(f"Faux Positifs (FP - détections incorrectes) : {total_fp}")
    print(f"Faux Négatifs (FN - bulles manquées)       : {total_fn}")
    print("="*40)

    if total_fp > total_fn:
        print("\nConclusion : Le modèle a plus de Faux Positifs. Il a tendance à sur-détecter (créer des bulles qui n'existent pas ou mal classées).")
    elif total_fn > total_fp:
        print("\nConclusion : Le modèle a plus de Faux Négatifs. Il a tendance à sous-détecter (manquer des bulles réelles).")
    else:
        print("\nConclusion : Le nombre de Faux Positifs et de Faux Négatifs est équilibré.")
    print(f"\nAnalyse terminée. Les images avec erreurs sont dans : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse les erreurs de détection d'un modèle Detectron2.")
    parser.add_argument("--model", required=True, help="Chemin vers le fichier du modèle (.pth).")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier de configuration (.yaml) du modèle.")
    parser.add_argument("--dataset", required=True, help="Chemin vers le dossier du jeu de données (contenant les .json).")
    parser.add_argument("--output", default="error_analysis", help="Dossier de sortie pour les visualisations d'erreurs.")
    
    args = parser.parse_args()
    
    analyze_errors(args.model, args.config, args.dataset, args.output)