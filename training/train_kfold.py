
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
from sklearn.model_selection import StratifiedKFold

# Import detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# --- Configuration Section ---

# Paths
BASE_DATASET_FOLDER = "dataset"  # All jsons and images are here
OUTPUT_DIR = "../MODELS/kfold_3classes_tip_png"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

# K-Fold settings
N_SPLITS = 5
RANDOM_STATE = 42

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

# --- Helper Functions ---

def get_all_labelme_files(dataset_path):
    """Gathers all labelme json files from the dataset directory."""
    return glob.glob(os.path.join(dataset_path, "*.json"))

def create_coco_dataset_from_labelme(labelme_files):
    """
    Creates a COCO formatted dictionary in memory from a list of labelme files.
    This is an adapted version of the logic in labelme2cocoMy.py.
    It also returns a list of categories per image for stratification.
    """
    images = []
    annotations = []
    
    image_id_map = {}
    ann_id = 0

    image_categories = []

    for i, file_path in enumerate(labelme_files):
        with open(file_path) as f:
            label_data = json.load(f)

        image_info = {
            "id": i,
            "file_name": file_path.replace(".json", ".png"), # Assuming png format, adjust if necessary
            "height": label_data["imageHeight"],
            "width": label_data["imageWidth"]
        }
        images.append(image_info)
        image_id_map[file_path] = i

        img_cats = set()
        for shape in label_data["shapes"]:
            raw_label = shape["label"]
            label = MAPPING.get(raw_label)
            if label is None:
                continue
            
            img_cats.add(CATEGORY_IDS[label])

            points = np.asarray(shape["points"])
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            width = xmax - xmin
            height = ymax - ymin
            
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))

            ann = {
                "id": ann_id,
                "image_id": i,
                "category_id": CATEGORY_IDS[label],
                "segmentation": [points.flatten().tolist()],
                "bbox": [float(xmin), float(ymin), float(width), float(height)],
                "area": area,
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1
        
        # For stratification, we can use the primary (first) category found
        primary_category = next(iter(img_cats), -1)
        image_categories.append(primary_category)

    categories = [{"id": CATEGORY_IDS[label], "name": label, "supercategory": "bubble"} for label in FINAL_LABELS]
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    return coco_data, np.array(image_categories)


def custom_mapper(dataset_dict):
    """Data augmentation mapper from the notebook."""
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    transform_list = [
        # Geometric augmentations
        T.ResizeShortestEdge(
            [640, 672, 704, 736, 768, 800], sample_style="choice"
        ),
        T.RandomRotation(angle=[-15, 15]),
        # Color augmentations
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

# --- Main Training Logic ---

def main():
    setup_logger()
    logger = logging.getLogger("detectron2")

    # 1. Prepare unified dataset
    logger.info("Preparing unified dataset from labelme files...")
    all_labelme_files = get_all_labelme_files(BASE_DATASET_FOLDER)
    coco_data, image_categories = create_coco_dataset_from_labelme(all_labelme_files)
    
    all_images = coco_data["images"]
    all_annotations = coco_data["annotations"]

    # Create master JSON for inspection if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "master_coco.json"), "w") as f:
        json.dump(coco_data, f)
    
    # 2. K-Fold Cross-validation loop
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_metrics = []
    best_ap = -1
    
    # Ensure stratification is possible
    if -1 in image_categories:
        logger.warning("Some images have no categories. Stratification might be suboptimal.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_images)), image_categories)):
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        logger.info(f"--- Starting Fold {fold + 1}/{N_SPLITS} ---")

        # Create train/val datasets for this fold
        train_images = [all_images[i] for i in train_idx]
        val_images = [all_images[i] for i in val_idx]
        
        train_img_ids = {img['id'] for img in train_images}
        val_img_ids = {img['id'] for img in val_images}

        train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_img_ids]
        val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_img_ids]

        train_coco = {"images": train_images, "annotations": train_annotations, "categories": coco_data['categories']}
        val_coco = {"images": val_images, "annotations": val_annotations, "categories": coco_data['categories']}
        
        # Register datasets for this fold
        train_dataset_name = f"bubbleid_train_fold_{fold + 1}"
        val_dataset_name = f"bubbleid_val_fold_{fold + 1}"
        
        if train_dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(train_dataset_name)
        if val_dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(val_dataset_name)
            
        register_coco_instances(train_dataset_name, {}, train_coco, "")
        register_coco_instances(val_dataset_name, {}, val_coco, "")

        # 3. Configure and train
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.OUTPUT_DIR = fold_output_dir
        
        cfg.DATASETS.TRAIN = (train_dataset_name,)
        cfg.DATASETS.TEST = (val_dataset_name,) # Set test dataset for evaluation
        cfg.DATALOADER.NUM_WORKERS = 2
        
        # Reset weights from model zoo for each fold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 5000 # Increased iterations for better convergence
        cfg.SOLVER.STEPS = (3000, 4000) # Learning rate decay
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(FINAL_LABELS)

        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # 4. Evaluate
        logger.info(f"--- Evaluating Fold {fold + 1} ---")
        cfg.MODEL.WEIGHTS = os.path.join(fold_output_dir, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)
        
        evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir=os.path.join(fold_output_dir, "evaluation"))
        val_loader = build_detection_test_loader(cfg, val_dataset_name)
        metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
        
        ap_metric = metrics.get('bbox', {}).get('AP', 0)
        all_metrics.append(ap_metric)
        
        # 5. Save best model
        if ap_metric > best_ap:
            best_ap = ap_metric
            os.makedirs(BEST_MODEL_DIR, exist_ok=True)
            best_model_path = os.path.join(BEST_MODEL_DIR, "best_model.pth")
            config_path = os.path.join(BEST_MODEL_DIR, "config.yaml")
            
            with open(os.path.join(fold_output_dir, "model_final.pth"), "rb") as f_src:
                with open(best_model_path, "wb") as f_dst:
                    f_dst.write(f_src.read())
            
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f)

            logger.info(f"New best model saved from Fold {fold + 1} with AP: {ap_metric:.4f}")

    # 6. Final Report
    logger.info("--- K-Fold Cross-Validation Finished ---")
    mean_ap = np.mean(all_metrics)
    std_ap = np.std(all_metrics)
    
    logger.info(f"AP scores for each fold: {all_metrics}")
    logger.info(f"Mean AP: {mean_ap:.4f}")
    logger.info(f"Standard Deviation of AP: {std_ap:.4f}")
    logger.info(f"Best model saved in: {BEST_MODEL_DIR}")

if __name__ == "__main__":
    main()
