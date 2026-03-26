
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
import argparse
import wandb
import albumentations as A
from sklearn.model_selection import StratifiedKFold

# Import detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

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

# --- Helper Functions ---

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Detectron2 k-fold training script with W&B and Albumentations")
    parser.add_argument("--dataset", default="dataset", help="Path to the dataset folder with labelme JSONs and images")
    parser.add_argument("--output-dir", default="../MODELS/kfold_run", help="Directory to save models and logs")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of k-fold splits")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for k-fold split")
    parser.add_argument("--max-iter", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--lr", type=float, default=0.00025, help="Base learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Images per batch (solver.ims_per_batch)")
    parser.add_argument("--wandb-project", default="bubbleid-kfold", help="W&B project name")
    return parser.parse_args()


def get_all_labelme_files(dataset_path):
    """Gathers all labelme json files from the dataset directory."""
    return glob.glob(os.path.join(dataset_path, "*.json"))

def create_coco_dataset_from_labelme(labelme_files):
    """
    Creates a COCO formatted dictionary in memory from a list of labelme files.
    This is an adapted version of the logic in labelme2cocoMy.py.
    It also returns a list of categories per image for stratification.
    """
    logger = logging.getLogger("detectron2")
    images = []
    annotations = []
    
    image_id_map = {}
    ann_id = 0

    image_categories = []

    for i, file_path in enumerate(labelme_files):
        with open(file_path) as f:
            label_data = json.load(f)

        base_path = os.path.splitext(file_path)[0]
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = base_path + ext
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        if image_path is None:
            logger.warning(f"Image for {file_path} not found. Skipping.")
            continue

        image_info = {
            "id": i,
            "file_name": image_path,
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


def custom_mapper_with_albumentations(dataset_dict):
    """
    A custom data mapper that uses Albumentations for data augmentation.
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # Define the augmentation pipeline
    transform = A.Compose([
        A.Resize(800, 800),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.GaussNoise(p=0.2),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.3))

    # Convert annotations to the format Albumentations expects
    bboxes = []
    category_ids = []
    masks = []
    for obj in dataset_dict.get("annotations", []):
        if obj.get("iscrowd", 0) == 1:
            continue
        bboxes.append(obj["bbox"])
        category_ids.append(obj["category_id"])
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for seg in obj["segmentation"]:
            poly = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        masks.append(mask)

    # Apply transformations
    try:
        transformed = transform(image=image, masks=masks, bboxes=bboxes, category_ids=category_ids)
        image_transformed = transformed['image']
        transformed_masks = transformed['masks']
        transformed_bboxes = transformed['bboxes']
        transformed_category_ids = transformed['category_ids']
    except (ValueError, IndexError): # Can happen if all bboxes are removed
        return None # Skip this image

    # Convert transformed data back to Detectron2 format
    annos = []
    for i in range(len(transformed_bboxes)):
        mask = transformed_masks[i]
        if mask.sum() == 0:
            continue
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = [c.flatten().tolist() for c in contours if c.shape[0] >= 3]
        if not segmentation:
            continue

        anno = {
            "bbox": transformed_bboxes[i],
            "bbox_mode": detectron2.structures.BoxMode.XYWH_ABS,
            "segmentation": segmentation,
            "category_id": transformed_category_ids[i],
            "iscrowd": 0,
        }
        annos.append(anno)

    if not annos:
        return None

    dataset_dict.pop("annotations", None)
    dataset_dict["image"] = torch.as_tensor(image_transformed.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annos, image_transformed.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class WandbHook(HookBase):
    def after_step(self):
        metrics = self.trainer.storage.latest()
        log_metrics = {k: v[0] for k, v in metrics.items() if isinstance(v, tuple) and isinstance(v[0], (int, float))}
        if log_metrics:
            wandb.log(log_metrics)

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_with_albumentations)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, WandbHook())
        return hooks

# --- Main Training Logic ---

def main(args):
    setup_logger()
    logger = logging.getLogger("detectron2")

    # Init wandb
    wandb.init(project=args.wandb_project, config=args)

    OUTPUT_DIR = args.output_dir
    BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")

    # 1. Prepare unified dataset
    logger.info("Preparing unified dataset from labelme files...")
    all_labelme_files = get_all_labelme_files(args.dataset)
    coco_data, image_categories = create_coco_dataset_from_labelme(all_labelme_files)
    
    all_images = coco_data["images"]
    all_annotations = coco_data["annotations"]

    # Create master JSON for inspection if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "master_coco.json"), "w") as f:
        json.dump(coco_data, f)
    
    # 2. K-Fold Cross-validation loop
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    all_metrics = []
    best_ap = -1
    
    # Ensure stratification is possible
    if -1 in image_categories:
        logger.warning("Some images have no categories. Stratification might be suboptimal.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_images)), image_categories)):
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        logger.info(f"--- Starting Fold {fold + 1}/{args.n_splits} ---")

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
        
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
        cfg.SOLVER.BASE_LR = args.lr
        cfg.SOLVER.MAX_ITER = args.max_iter
        cfg.SOLVER.STEPS = (int(args.max_iter * 0.6), int(args.max_iter * 0.8)) # Dynamic steps
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(FINAL_LABELS)

        # Log full config to wandb for the first fold
        if fold == 0:
             wandb.config.update(cfg)

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

        # Log evaluation metrics to wandb
        eval_log = {f"fold_{fold+1}/{k}": v for k, v in metrics.get("bbox", {}).items()}
        wandb.log(eval_log)
        
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
    
    # Log summary to wandb
    wandb.summary["mean_ap"] = mean_ap
    wandb.summary["std_ap"] = std_ap
    wandb.summary["best_ap"] = best_ap

    logger.info(f"AP scores for each fold: {all_metrics}")
    logger.info(f"Mean AP: {mean_ap:.4f}")
    logger.info(f"Standard Deviation of AP: {std_ap:.4f}")
    logger.info(f"Best model saved in: {BEST_MODEL_DIR}")

    wandb.finish()

if __name__ == "__main__":
    args = get_args()
    main(args)
