import os
import json
import glob
import numpy as np
import torch
import detectron2
import cv2
import random
import yaml
import copy
import logging
import albumentations as A
from sklearn.model_selection import StratifiedKFold

# Import detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, hooks
from detectron2.engine.hooks import HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# --- HYPERPARAMETERS ---
# Modify these values to tune your training run.

DATASET_PATH = "dataset"
OUTPUT_DIR = "kfold_run"
N_SPLITS = 4
RANDOM_STATE = 42
MAX_ITER = 4000  # Augmenté pour donner sa chance à l'Early Stopping
LR = 0.0005
BATCH_SIZE = 4  # Réduit pour limiter la surchauffe du GPU (était à 8)
EVAL_PERIOD = 400  # Évaluer sur le set de validation tous les 300 itérations
CHECKPOINT_PERIOD = 400 # Sauvegarder un checkpoint tous les 300 itérations

# --- Early Stopping Configuration ---
EARLY_STOPPING_PATIENCE = 3  # Patience de 3 évaluations. Si pas d'amélioration, on arrête.
EARLY_STOPPING_METRIC = "bbox/AP" # Métrique à surveiller pour l'arrêt précoce.

# --- Configuration Section ---

# Label Mapping (from labelme2cocoMy.py)
MAPPING = {
    "detached": "detached",
    "occluseDetached": "detached",
    "occlusedDetached": "detached",
    "occlusedAttached": "occludedAttached",
    "unknown": "occludedAttached",
    "attachedSide": "attached",
    "attached": "attached"
}
CATEGORY_IDS = {
    "detached": 0,
    "occludedAttached": 1,
    "attached": 2
}
FINAL_LABELS = sorted(list(set(MAPPING.values())))

# --- Early Stopping Hook ---
class EarlyStoppingHook(HookBase):
    def __init__(self, patience, metric, goal="max"):
        self._patience = patience
        self._metric = metric
        self._goal = goal
        
        self._patience_counter = 0
        self._best_metric = -float('inf') if self._goal == "max" else float('inf')
        self._logger = logging.getLogger("detectron2")

    def after_step(self):
        # Ce hook dépend des résultats de l'EvalHook stockés
        latest_metrics = self.trainer.storage.latest()
        
        # La métrique n'est présente que lorsque l'évaluation a eu lieu
        if self._metric not in latest_metrics:
            return

        # Récupérer la valeur ET l'itération où elle a été enregistrée
        current_metric, metric_iter = latest_metrics[self._metric]

        # CORRECTION CRUCIALE : N'agir que si la métrique a été enregistrée à l'itération ACTUELLE.
        # Cela empêche le hook de ré-évaluer la même ancienne métrique à chaque pas, ce qui
        # déclenchait l'arrêt prématurément.
        if metric_iter != self.trainer.iter:
            return

        improved = False
        if self._goal == "max":
            if current_metric > self._best_metric:
                self._best_metric = current_metric
                improved = True
        else: # min
            if current_metric < self._best_metric:
                self._best_metric = current_metric
                improved = True
        
        if improved:
            self._patience_counter = 0
            # Le BestCheckpointer de detectron2 s'occupe de sauvegarder
        else:
            self._patience_counter += 1
        
        if self._patience_counter >= self._patience:
            self._logger.info(f"Arrêt précoce déclenché après {self._patience} évaluations sans amélioration.")
            self._logger.info(f"Meilleure métrique obtenue : {self._best_metric:.4f}")
            raise StopIteration # Stoppe la boucle d'entraînement proprement


# --- Helper Functions ---

def get_all_labelme_files(dataset_path):
    """Gathers all labelme json files from the dataset directory."""
    return glob.glob(os.path.join(dataset_path, "*.json"))

def create_detectron2_dataset_from_labelme(labelme_files):
    """
    Creates a Detectron2 formatted list of dicts in memory from a list of labelme files.
    It also returns a list of categories per image for stratification and the category mapping.
    """
    logger = logging.getLogger("detectron2")
    dataset_dicts = []
    image_categories = []

    for i, file_path in enumerate(labelme_files):
        with open(file_path) as f:
            label_data = json.load(f)

        base_path = os.path.splitext(file_path)[0]
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            potential_path = base_path + ext
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        if image_path is None:
            logger.warning(f"Image for {file_path} not found. Skipping.")
            continue

        record = {
            "file_name": image_path,
            "image_id": i,
            "height": label_data["imageHeight"],
            "width": label_data["imageWidth"],
            "annotations": []
        }

        img_cats = set()
        for shape in label_data["shapes"]:
            raw_label = shape["label"]
            label = MAPPING.get(raw_label)
            if label is None:
                continue

            points = np.asarray(shape["points"])
            
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            elif len(points) < 3:
                continue

            img_cats.add(CATEGORY_IDS[label])

            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin

            ann = {
                "bbox": [float(xmin), float(ymin), float(width), float(height)],
                "bbox_mode": detectron2.structures.BoxMode.XYWH_ABS,
                "category_id": CATEGORY_IDS[label],
                "segmentation": [points.flatten().tolist()],
                "iscrowd": 0
            }
            record["annotations"].append(ann)

        dataset_dicts.append(record)

        primary_category = next(iter(img_cats), -1)
        image_categories.append(primary_category)

    categories = [{"id": CATEGORY_IDS[label], "name": label} for label in FINAL_LABELS]

    return dataset_dicts, np.array(image_categories), categories


# --- Global variable to cache the background image ---
# _background_image = None # Désactivé car la soustraction de fond réduisait la précision.

def custom_mapper_with_albumentations(dataset_dict):
    """
    A custom data mapper that uses Albumentations for data augmentation.
    This version handles non-rigid transformations (like GridDistortion) by
    converting segmentations to masks, transforming the masks, and then
    reconstructing the annotations.
    La soustraction de fond a été désactivée car elle semblait réduire la précision.
    """
    # global _background_image # Désactivé

    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # --- Background Subtraction (Désactivé car réduisait la précision) ---
    # if _background_image is None:
    #     background_path = os.path.join(DATASET_PATH, "background.png")
    #     if os.path.exists(background_path):
    #         _background_image = cv2.imread(background_path, cv2.IMREAD_COLOR)
    #         logging.getLogger("detectron2").info(f"Loaded background image from {background_path}")
    # if _background_image is not None:
    #     bg_resized = cv2.resize(_background_image, (image.shape[1], image.shape[0]))
    #     image = cv2.absdiff(image, bg_resized)

    H, W, _ = image.shape

    # 1. Convert polygon segmentations to binary masks
    masks = []
    category_ids = []
    for ann in dataset_dict.get("annotations", []):
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if not isinstance(ann["segmentation"], list) or not ann["segmentation"]:
            continue
            
        poly = np.array(ann["segmentation"][0]).reshape(-1, 2)
        
        # fillPoly requires integer coordinates
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        masks.append(mask)
        category_ids.append(ann["category_id"])

    # 2. Define Albumentations pipeline for images and masks
    # Refined augmentation pipeline (Priority #2)
    transform = A.Compose([
        A.LongestMaxSize(max_size=800),
        A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        # Reduced non-rigid transforms, added MotionBlur
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
    ])

    try:
        # 3. Apply the transformation
        transformed = transform(image=image, masks=masks)
        image_transformed = transformed['image']
        masks_transformed = transformed['masks']
    except (ValueError, IndexError) as e:
        # This can happen if augmentations result in empty masks/images
        logging.getLogger("detectron2").warning(f"Skipping an image due to augmentation error: {e}")
        return None

    # 4. Reconstruct annotations from transformed masks
    annos = []
    for i, mask_t in enumerate(masks_transformed):
        # Find contours from the transformed mask. This can return multiple contours
        # if an augmentation splits a single annotation into multiple parts.
        contours, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Iterate over all found contours for this mask
        for contour in contours:
            # We need at least 3 points to form a valid polygon
            if len(contour) < 3:
                continue

            # Create new bounding box from the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Optional: filter out very small contours that might be noise
            if w <= 1 or h <= 1:
                continue

            annos.append({
                "bbox": [float(x), float(y), float(w), float(h)],
                "bbox_mode": detectron2.structures.BoxMode.XYWH_ABS,
                "segmentation": [contour.flatten().tolist()],
                "category_id": category_ids[i], # All parts get the same category ID
                "iscrowd": 0,
            })

    if not annos:
        return None

    # 5. Finalize the dataset dictionary for Detectron2
    dataset_dict.pop("annotations", None)
    dataset_dict["image"] = torch.as_tensor(image_transformed.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annos, image_transformed.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
    return dataset_dict


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_with_albumentations)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        # Surcharge pour ajouter l'évaluation périodique et l'arrêt précoce
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = self.cfg.DATALOADER.NUM_WORKERS 
        
        ret = super().build_hooks()
        
        # Le BestCheckpointer exécute déjà une évaluation, il n'est donc pas
        # nécessaire d'ajouter un EvalHook séparé. Les résultats de l'évaluation
        # du BestCheckpointer seront disponibles pour les autres hooks.

        # Hook pour sauvegarder le meilleur modèle (qui inclut l'évaluation)
        ret.append(hooks.BestCheckpointer(
            cfg.TEST.EVAL_PERIOD, self.checkpointer, EARLY_STOPPING_METRIC, "max"
        ))

        # Hook pour l'arrêt précoce
        ret.append(EarlyStoppingHook(
            patience=EARLY_STOPPING_PATIENCE,
            metric=EARLY_STOPPING_METRIC
        ))
        
        return ret


# --- Main Training Logic ---

def main():
    setup_logger()
    logger = logging.getLogger("detectron2")

    # 1. Prepare unified dataset
    logger.info("Preparing unified dataset from labelme files...")
    all_labelme_files = get_all_labelme_files(DATASET_PATH)
    all_dataset_dicts, image_categories, categories = create_detectron2_dataset_from_labelme(all_labelme_files)
    
    # 2. K-Fold Cross-validation loop
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_final_metrics = {}
    
    if -1 in image_categories:
        logger.warning("Some images have no categories. Stratification might be suboptimal.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_dataset_dicts)), image_categories)):
        # Condition pour ne lancer que le fold 4, comme demandé.
        # Pour un entraînement complet, commentez ou supprimez cette ligne.
        if (fold + 1) != 4:
            continue

        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        logger.info(f"--- Starting Fold {fold + 1}/{N_SPLITS} ---")

        # Create train/val datasets for this fold
        train_dicts = [all_dataset_dicts[i] for i in train_idx]
        val_dicts = [all_dataset_dicts[i] for i in val_idx]
        
        train_dataset_name = f"bubbleid_train_fold_{fold + 1}"
        val_dataset_name = f"bubbleid_val_fold_{fold + 1}"
        
        # Nettoyer les catalogues au cas où le script est exécuté plusieurs fois
        for d in [train_dataset_name, val_dataset_name]:
            if d in DatasetCatalog.list():
                DatasetCatalog.remove(d)
            if d in MetadataCatalog.list():
                MetadataCatalog.remove(d)
            
        DatasetCatalog.register(train_dataset_name, lambda d=train_dicts: d)
        MetadataCatalog.get(train_dataset_name).set(thing_classes=[c['name'] for c in categories])

        DatasetCatalog.register(val_dataset_name, lambda d=val_dicts: d)
        MetadataCatalog.get(val_dataset_name).set(thing_classes=[c['name'] for c in categories])

        # 3. Configure and train
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.OUTPUT_DIR = fold_output_dir
        
        cfg.DATASETS.TRAIN = (train_dataset_name,)
        cfg.DATASETS.TEST = (val_dataset_name,)
        cfg.DATALOADER.NUM_WORKERS = 4
        
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
        cfg.SOLVER.BASE_LR = LR
        cfg.SOLVER.MAX_ITER = MAX_ITER
        # On réactive la réduction du learning rate. Lorsque les métriques stagnent (plateau),
        # réduire le LR permet d'affiner la recherche d'un meilleur minimum.
        # Cela fonctionne bien avec l'arrêt précoce.
        cfg.SOLVER.STEPS = (3200,) # Réduit le LR à 3200 itérations. Par défaut, il est divisé par 10.
        cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
        cfg.SOLVER.AMP.ENABLED = True
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(FINAL_LABELS)
        
        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
            f.write(cfg.dump())

        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training stopped with an exception: {e}", exc_info=True)

        best_fold_ap = -1.0
        try:
            with open(os.path.join(fold_output_dir, "metrics.json")) as f:
                metrics_lines = f.readlines()
            
            for line in metrics_lines:
                metrics = json.loads(line)
                if EARLY_STOPPING_METRIC in metrics and isinstance(metrics[EARLY_STOPPING_METRIC], (float, int)):
                    current_ap = metrics[EARLY_STOPPING_METRIC]
                    if current_ap > best_fold_ap:
                        best_fold_ap = current_ap
                        
        except FileNotFoundError:
            logger.warning(f"Le fichier metrics.json n'a pas été trouvé pour le fold {fold + 1}. "
                           f"Score AP considéré comme -1.")
        
        all_final_metrics[f"fold_{fold + 1}"] = best_fold_ap
        logger.info(f"Fold {fold+1} Best AP: {best_fold_ap:.4f}")

    # 4. Final Report
    logger.info("--- K-Fold Cross-Validation Finished ---")
    
    best_fold_name = ""
    best_fold_ap = -1
    for fold_name, ap in all_final_metrics.items():
        logger.info(f"Final AP for {fold_name}: {ap:.4f}")
        if ap > best_fold_ap:
            best_fold_ap = ap
            best_fold_name = fold_name
            
    if best_fold_name:
        logger.info(f"\nBest fold was {best_fold_name} with AP: {best_fold_ap:.4f}")
        # Copier le meilleur modèle du meilleur "fold" dans le dossier principal
        BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model_overall")
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        
        best_model_src_path = os.path.join(OUTPUT_DIR, best_fold_name, "model_best.pth")
        
        if os.path.exists(best_model_src_path):
            best_model_dst_path = os.path.join(BEST_MODEL_DIR, "model_final.pth")
            config_dst_path = os.path.join(BEST_MODEL_DIR, "config.yaml")

            with open(best_model_src_path, "rb") as f_src, open(best_model_dst_path, "wb") as f_dst:
                f_dst.write(f_src.read())
                
            config_src_path = os.path.join(OUTPUT_DIR, best_fold_name, "config.yaml")
            if os.path.exists(config_src_path):
                with open(config_src_path, "r") as f_src, open(config_dst_path, "w") as f_dst:
                    f_dst.write(f_src.read())

            logger.info(f"Best overall model copied to {BEST_MODEL_DIR}")
        else:
            logger.warning(f"Could not find best model file at {best_model_src_path}")


if __name__ == "__main__":
    main()
