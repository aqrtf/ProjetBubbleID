"""Script pour creer des modele a partir d'image de qualite differentes, 
enregistre les poids puis teste les performances et les ranges dans un csv"""
# Import Libraries:
import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.data import transforms as T
# import some common libraries
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
import yaml, copy

# import some common detectron2 utilities
from detectron2 import model_zoo, structures
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import ColorMode


import labelme2cocoMy
from PIL import Image
import os

def convert_all_png_to_jpg(images_folder, output_folder, quality):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(images_folder, filename)
            img = Image.open(png_path).convert("RGB")
            
            # Conserver le nom de fichier, changer l'extension
            base_name = os.path.splitext(filename)[0]
            jpg_path = os.path.join(output_folder, f"{base_name}.jpg")
            
            img.save(jpg_path, "JPEG", quality=quality)


qualities = [100, 90, 80, 70, 60]

for qual in qualities:
    pass
# Pour le training set
datasetFolderPath = "DATASETS\\dataset_tip_png" # path of the dataset folder
labelme_folder_path_train = datasetFolderPath + "\\train" # chemin du dossier ou sont enregistre les images annotees avec labelme
coco_path_train = datasetFolderPath + "\\train.json" # output path

# Et pour le validation set
labelme_folder_path_val = datasetFolderPath + "\\val"
coco_path_val = datasetFolderPath + "\\val.json"


# output directody where save the model
outDirModel = "..\\MODELS\\" + "3classes_tip_png"




# Pour le training set
labelme2cocoMy.labelme2coco(labelme_folder_path_train, coco_path_train)
# Et pour le validation set
labelme2cocoMy.labelme2coco(labelme_folder_path_val, coco_path_val)

# on retire les datasets deja existant
for d in ["my_dataset_train", "my_dataset_test"]:
    if d in DatasetCatalog.list():
        DatasetCatalog.remove(d)
    if d in MetadataCatalog.list():
        MetadataCatalog.remove(d)

register_coco_instances("my_dataset_train", {}, coco_path_train, labelme_folder_path_train)
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

register_coco_instances("my_dataset_test", {}, coco_path_val, labelme_folder_path_val)
val_metadata = MetadataCatalog.get("my_dataset_test")
val_dataset_dicts = DatasetCatalog.get("my_dataset_test")


#####################################################################
# Parametres pour le training
cfg = get_cfg()
cfg.OUTPUT_DIR = outDirModel
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_MATLAB1.pth")  # path to the model we just trained
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # Default is 512, using 256 for this dataset.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # NOTE !!!!!! il faut modifier ce parametre en fonction du nombre de classe
# NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg) #Create an instance of of DefaultTrainer with the given congiguration
#trainer.resume_or_load(resume=False) #Load a pretrained model if available (resume training) or start training from scratch if no pretrained model is available

####################################################################
# Alteration des images / flip pour generalisation
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    mean = 0
    std_dev = 25
    gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    #noisy_image = cv2.add(image, gaussian_noise)
    
    transform_list = [
        #T.Resize((800,600)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        #T.RandomRotation(angle=[90, 90]),
        #T.RandomNoise(mean=0.0, std=0.1),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
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

trainer=CustomTrainer(cfg)
trainer.resume_or_load(resume=False)


####################################################################
# Training
trainer.train()
config_yaml_path = os.path.join(outDirModel, "config.yaml")
with open(config_yaml_path, 'w') as file:
    yaml.dump(cfg, file)

#####################################################################
#####################################################################
# import the librairies
import cv2, os
from torch.cuda import is_available

# Define the parameters for the model:
datasetFolder = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\training\DATASETS\dataset_tip_png"
imagesfolder = os.path.join(datasetFolder, "val/")   # Define the path to the folder of images
videopath = os.path.join(datasetFolder, "valVideo.avi")   # Define the path to the avi video
savefolder="../training/Output/"   # Define the folder you want the data to save in
savefolder2 = "../training/Performances_tip/" 
extension="all_jpg"    # Define the extension you want all the saved data to have. This should be unique for each experiment
thres=0.5    # Define the threshold for what the model identifies as a bubble
modelweights = "..\\MODELS\\" + "Models_3classes_all" + "\\model_final.pth"
# modelweights=r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\training\Models_3classes_all\model_final.pth"     # Define the path to the saved model weights.
# device='cpu'   # Specify if running on "cpu" or "gpu"
device = "cuda" if is_available() else "cpu"
print(f"Used device : {device}")


validationCOCO_path = os.path.join(datasetFolder, "val.json")



#####################################################################
# On creer le coco avec les fichier deja existant rich et contour
import json, csv


contours_json_path = savefolder + 'contours_' + extension + '.json'
rich_csv_path = savefolder + 'rich_' + extension + '.csv'
outputCoco_path = os.path.join(savefolder2,"predictions_" + extension + ".json")

# Lire le fichier JSON d'entrée
with open(contours_json_path, 'r', encoding='utf-8') as f_json:
    contours = json.load(f_json)

# Lire le fichier rich.csv et ne selectionner que les colonne qui nous interressent
donnees_filtrees = []

with open(rich_csv_path, "r", encoding="utf-8") as f:
    lecteur = csv.DictReader(f)
    for ligne in lecteur:
        entree = {
            "frame": int(ligne["frame"]),
            "det_in_frame": int(ligne["det_in_frame"]),
            "x1": int(ligne["x1"]),
            "y1": int(ligne["y1"]),
            "x2": int(ligne["x2"]),
            "y2": int(ligne["y2"]),
            "score": float(ligne["score"]),
            "class_id": int(ligne["class_id"])
        }
        donnees_filtrees.append(entree)


if len(donnees_filtrees) != len(contours):
    raise("Le fichier contours et rich n'ont pas le meme nb de lignes")

outputFile = []
for idx, imgCode in enumerate(contours):
    # on verifie qu'on parle de la meme image
    i, d = map(int, imgCode.split("_"))
    if i != (donnees_filtrees[idx]["frame"]) or d != (donnees_filtrees[idx]["det_in_frame"]):
        print("WARNING: les 2 fichiers ne sont pas compatibles")
        print(i, donnees_filtrees[idx]["frame"], d, donnees_filtrees[idx]["det_in_frame"])
        continue
    
    width = (donnees_filtrees[idx]["x2"]) - (donnees_filtrees[idx]["x1"])
    heigth = (donnees_filtrees[idx]["y2"]) - (donnees_filtrees[idx]["y1"])
    flat_coords = [x for pt in contours[imgCode] for x in pt]
    prediction = {
        "image_id": donnees_filtrees[idx]["frame"] - 1 , #Dans ce fichier les idx commencent a 1 alors que dans l'autre ils commencent a 0
        "category_id": donnees_filtrees[idx]["class_id"],
        "segmentation": [flat_coords],
        "score": donnees_filtrees[idx]["score"],
        "bbox": [donnees_filtrees[idx]["x1"], donnees_filtrees[idx]["y1"], width, heigth ]
    }
    outputFile.append(prediction)
os.makedirs(savefolder2, exist_ok=True)   
# Écrire dans un fichier JSON
with open(outputCoco_path, "w", encoding="utf-8") as f:
    json.dump(outputFile, f, ensure_ascii=False, indent=2)


############################################################################
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import sys
import io

# --- Charger les fichiers ---
annType = 'segm'  # 'bbox' pour détection, 'segm' pour segmentation

coco_gt = COCO(validationCOCO_path)       # fichier COCO ground truth
coco_dt = coco_gt.loadRes(outputCoco_path)  # fichier prédictions

# --- Évaluation ---
coco_eval = COCOeval(coco_gt, coco_dt, annType)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
# --- Capture de la sortie de summarize ---
buffer = io.StringIO()
sys.stdout = buffer  # Redirige stdout vers le buffer
coco_eval.summarize()
sys.stdout = sys.__stdout__  # Restaure stdout

# --- Sauvegarde dans un fichier texte ---
outputResult_path = os.path.join(savefolder2,"performance_" + extension + ".json")

with open(outputResult_path, "w", encoding="utf-8") as f:
    f.write(buffer.getvalue())




