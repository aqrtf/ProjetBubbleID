# BubbleID
print("Load BubbleID_My")
# Import libraries:
import sys

# sys.path.insert(0, r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet2\BubbleID\BubbleID")
import torch, detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo, structures
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import torch

torch.cuda.is_available()

from detectron2.engine import DefaultTrainer
import os
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import time
from tqdm import tqdm
import numpy as np
import cv2
import filterpy
import torch
import super_gradients as sg
import matplotlib.pyplot as plt
from ocsort import ocsort
import colorsys
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import PIL.Image
from scipy.spatial import cKDTree

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cv2
import numpy as np

import torch, detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.data import transforms as T
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo, structures
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
import torch

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import math, json
from math import sqrt, pi



# Define Helper Functions
def get_image_paths(directory):
    image_extensions = ['*.jpg']  # Add more extensions as needed

    image_paths = []
    for extension in image_extensions:
        pattern = os.path.join(directory, '**', extension)
        image_paths.extend(glob.glob(pattern, recursive=True))

    return sorted(image_paths)  # Sort the list of image paths alphabetically


def get_color(number):
    """ Converts an integer number to a color """
    # change these however you want to
    hue = number * 30 % 180
    saturation = number * 103 % 256
    value = number * 50 % 256

    # expects normalized values
    color = colorsys.hsv_to_rgb(hue / 179, saturation / 255, value / 255)

    return [int(c * 255) for c in color]


# Define Function for computing iou
def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)


class DataAnalysis:
    def __init__(self, imagesfolder, videopath, savefolder, extension, modelweightsloc, device):
        self.imagesfolder = imagesfolder
        self.videopath = videopath
        self.savefolder = savefolder
        self.extension = extension
        self.modeldirectory, self.modelweights = os.path.split(modelweightsloc)
        self.device = device
        
    def trimVideo(self, N_frames_extr=50):
        """Take a subvideo of the whole one and save each frame separately for the following analysis. 
            Determine the time of each frame and save in time

        Args:
            N_frames_extr (int, optional): The number of frames taken for the analysis. 
            Takes the first one. Defaults to 50.
        Outs:
            Time_xxx.csv : the time at each frame
            
        """
        # 2) Creazione cartella
        os.makedirs(self.imagesfolder, exist_ok=True)
        # temps des frames
        time = np.zeros(N_frames_extr)
        # 3) Lettura del video e salvataggio dei primi 50 frame
        cap = cv2.VideoCapture(self.videopath)
        for idx in range(N_frames_extr):
            ret, frame = cap.read()
            if not ret:
                print(f"Si sono estratti solo {idx} frame, stop.")
                break
            cv2.imwrite(os.path.join(self.imagesfolder, f"frame_{idx:03d}.jpg"), frame)
            time[idx] = cap.get(cv2.CAP_PROP_POS_MSEC) # read the time of each frame
        cap.release()
        print(f"Estratti {min(idx+1,N_frames_extr)} frame in '{self.imagesfolder}'")

        # create a df for the time
        time = time[:N_frames_extr] # reshape the array if there is not enough frame
        df_time = pd.DataFrame({"Frame": range(N_frames_extr), "Time_ms": time})
        timePath = os.path.join(self.savefolder, f"time_{self.extension}.csv")
        df_time.to_csv(timePath, index=False)

        # save the video with only the selected frames
        dst = os.path.join(self.imagesfolder, "..", "trimmed_50.avi")
        # Prendi codec e fps dal video originale
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID") # format de compression de la video
        out    = cv2.VideoWriter(dst, fourcc, fps, (width, height)) #preparation d'une variable pour save la video

        for i in range(N_frames_extr):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        print("Trimmed video salvato in", dst)



    def set_scale_by_two_points(
            self,
            frame_idx: int = 0,
            physical_mm: float = 20.0,  # 2 cm
            save: bool = True,
        ) -> float:
        """
        Seleziona 2 punti sul frame (0-based) e calcola la scala mm/px.
        UI:
          - click SINISTRO: aggiungi punto (max 2)
          - linea tra i punti: mostrata in tempo reale
          - puntatore: croce guida sul mouse
          - R o click DESTRO: resetta e rifai
          - ENTER: conferma e chiudi
          - ESC: annulla

        Ritorna mm_per_px e lo salva in scale_<EXT>.json se save=True.
        """

        # carica il frame
        img_path = os.path.join(self.imagesfolder, f"frame_{frame_idx:03d}.jpg")
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Frame non trovato: {img_path}")
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # tenta backend GUI per matplotlib
        pts = []
        accepted = {"ok": False}
        try:
            import matplotlib
            for _bk in ("Qt5Agg", "QtAgg", "TkAgg", "WXAgg", "MacOSX"):
                try:
                    matplotlib.use(_bk, force=True)
                    break
                except Exception:
                    continue
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Cursor

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_rgb)
            ax.set_axis_off()
            ax.set_title(f"Clicca 2 punti (ENTER=OK, R o click destro=rifai, ESC=annulla)  —  {physical_mm:.1f} mm")
            # croce puntatore
            cursor = Cursor(ax, useblit=True, color='lime', linewidth=1)

            # artisti dinamici
            scat = None  # punti cliccati
            temp_line = None  # linea provvisoria (1 punto + mouse)
            perm_line = None  # linea definitiva (2 punti)
            dist_txt = ax.text(10, 20, "dist: -", color='yellow',
                               fontsize=10, ha='left', va='center', bbox=dict(facecolor='black', alpha=0.4))

            def _reset():
                nonlocal scat, temp_line, perm_line
                pts.clear()
                if scat is not None:
                    scat.remove();
                    scat = None
                if temp_line is not None:
                    temp_line.remove();
                    temp_line = None
                if perm_line is not None:
                    perm_line.remove();
                    perm_line = None
                dist_txt.set_text("dist: -")
                fig.canvas.draw_idle()

            def _update_scatter():
                nonlocal scat
                if scat is not None:
                    scat.remove()
                if pts:
                    xs = [p[0] for p in pts];
                    ys = [p[1] for p in pts]
                    scat = ax.scatter(xs, ys, s=40, c='yellow', edgecolors='black', zorder=3)

            def on_click(ev):
                # sinistro: aggiungi punto; destro: reset
                if ev.inaxes != ax:
                    return
                if ev.button == 1:
                    if len(pts) < 2 and ev.xdata is not None and ev.ydata is not None:
                        pts.append((float(ev.xdata), float(ev.ydata)))
                        _update_scatter()
                        # chiudi provvisoria se siamo al secondo punto e disegna definitiva
                        if len(pts) == 2:
                            nonlocal temp_line, perm_line
                            if temp_line is not None:
                                temp_line.remove();
                                temp_line = None
                            x1, y1 = pts[0];
                            x2, y2 = pts[1]
                            perm_line, = ax.plot([x1, x2], [y1, y2], '-', lw=2, color='yellow')
                            dpx = math.hypot(x2 - x1, y2 - y1)
                            dist_txt.set_text(f"dist: {dpx:.2f} px  →  mm/px = {physical_mm / dpx:.6f}")
                            fig.canvas.draw_idle()
                elif ev.button == 3:
                    _reset()

            def on_move(ev):
                # aggiorna la linea provvisoria quando c'è un solo punto
                if ev.inaxes != ax or len(pts) != 1 or ev.xdata is None or ev.ydata is None:
                    return
                nonlocal temp_line
                x1, y1 = pts[0]
                if temp_line is None:
                    temp_line, = ax.plot([x1, ev.xdata], [y1, ev.ydata], '-', lw=2, color='yellow')
                else:
                    temp_line.set_data([x1, ev.xdata], [y1, ev.ydata])
                fig.canvas.draw_idle()

            def on_key(ev):
                if ev.key in ("escape",):  # ESC
                    plt.close(fig)
                elif ev.key in ("r", "R"):
                    _reset()
                elif ev.key in ("enter", "return"):
                    if len(pts) == 2:
                        accepted["ok"] = True
                        plt.close(fig)

            cid1 = fig.canvas.mpl_connect('button_press_event', on_click)
            cid2 = fig.canvas.mpl_connect('motion_notify_event', on_move)
            cid3 = fig.canvas.mpl_connect('key_press_event', on_key)

            plt.show()

        except Exception as e:
            # fallback minimale: inserisci da tastiera
            print("[warn] GUI non disponibile, uso input da tastiera. (Motivo:", e, ")")
            x1, y1 = map(float, input("pt1 (x y): ").split())
            x2, y2 = map(float, input("pt2 (x y): ").split())
            pts = [(x1, y1), (x2, y2)]
            accepted["ok"] = True

        if not accepted["ok"] or len(pts) != 2:
            raise RuntimeError("Calibrazione annullata o incompleta.")

        (x1, y1), (x2, y2) = pts[0], pts[1]
        dpx = math.hypot(x2 - x1, y2 - y1)
        if dpx <= 0:
            raise ValueError("Distanza pixel nulla.")
        mm_per_px = float(physical_mm) / float(dpx)
        px_per_mm = float(dpx) / float(physical_mm)

        # salva in RAM e file
        self.mm_per_px = mm_per_px
        if save:
            os.makedirs(self.savefolder, exist_ok=True)
            out_json = os.path.join(self.savefolder, f"scale_{self.extension}.json")
            payload = {
                "mm_per_px": mm_per_px,
                "px_per_mm": px_per_mm,
                "physical_mm_between_points": float(physical_mm),
                "frame_idx": int(frame_idx),
                "pt1": [int(round(x1)), int(round(y1))],
                "pt2": [int(round(x2)), int(round(y2))],
                "image_size_hw": [int(H), int(W)],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ui": "matplotlib_cursor"
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[scale] salvato: {out_json}  (mm/px = {mm_per_px:.6f})")
        else:
            print(f"[scale] mm/px = {mm_per_px:.6f} (non salvato su file)")

        return mm_per_px

    #def GenerateData(self, thres=0.5):
    def GenerateData(
            self,
            thres=0.5,
            *,
            save_rich: bool = True,  # scrive rich_<EXT>.csv
            save_masks: bool = False,  # salva masks_<EXT>.npz (può pesare molto)
            save_contours: bool = True,  # salva contours_<EXT>.json
            iou_thresh_tid: float = 0.50,  # soglia IoU per associare track_id ↔ detection
            rich_basename: str | None = None  # nome base del CSV (default: rich_<EXT>)
    ):
        import os, json, math, copy, glob
        import csv
        import cv2
        import numpy as np
        import torch
        from tqdm import tqdm

        directory_path = self.imagesfolder
        video_file = self.videopath

        # define save locations (invariati)
        file_path = os.path.join(self.savefolder, f'bb-Boiling-{self.extension}.txt')
        output_file_path = os.path.join(self.savefolder, f'bb-Boiling-output-{self.extension}.txt')
        vapor_file = os.path.join(self.savefolder, f'vapor_{self.extension}.npy')
        vapor_base_file = os.path.join(self.savefolder, f'vaporBase_bt-{self.extension}.npy')
        bubble_size_file = os.path.join(self.savefolder, f'bubble_size_bt-{self.extension}.npy')
        bubind_file = os.path.join(self.savefolder, f'bubind_{self.extension}.npy')
        frameind_file = os.path.join(self.savefolder, f'frames_{self.extension}.npy')
        classind_file = os.path.join(self.savefolder, f'class_{self.extension}.npy')
        bubclassind_file = os.path.join(self.savefolder, f'bubclass_{self.extension}.npy')

        # ---- Opzioni per output "ricco" (nuovi file) ----
        SAVE_RICH = bool(save_rich)
        SAVE_MASKS = bool(save_masks)
        SAVE_CONTOURS = bool(save_contours)
        IOU_THRESH_TID = float(iou_thresh_tid)
        RICH_BASENAME = rich_basename or f"rich_{self.extension}"

        # Collezionatori nuovi (non intaccano i file storici)
        _rows_rich = []  # una riga per detection filtrata
        _mask_store = {}  # "frame_det" -> mask (bool)
        _contour_store = {}  # "frame_det" -> lista punti [[x,y],...]

        # scala: usiamo SOLO quella già impostata dall'utente
        # (nessun fallback: l'utente ha già chiamato set_scale_by_two_points(..., physical_mm=2.0))
        mm_per_px = float(self.mm_per_px)  # deve essere presente
        # px_per_mm = 1.0 / mm_per_px  # non necessario qui, ma lo lascio commentato

        # Make save folder if it does not exist
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)

        # ---- model/predictor (invariato nella sostanza) ----
        print("Load model")
        cfg = get_cfg()
        cfg.OUTPUT_DIR = self.modeldirectory
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000 # 1000 iterations seems good enough for this dataset
        cfg.SOLVER.STEPS = [] # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # Default is 512, using 256 for this dataset.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.modelweights)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set a custom testing threshold
        # if self.device == 'cpu':
        #     cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)

        # ---- helpers per IoU & features su maschere ----
        # IoU = interserction over union 
        # IoU high == good prediction7
        # bbox == bounding box
        def _bbox_iou(b1, b2):
            """calculate the Intersection over Union (IoU) between 
            the 2 bounding boxes b1 and b2

            Args:
                b1 (float tuple/list): (x1,y1,x2,y2) bottom lhs and up rhs coord of the rectangle
                b2 (float tuple/list): (x1,y1,x2,y2)

            Returns:
                float: IoU
            """
            # b = (x1,y1,x2,y2)
            xA = max(b1[0], b2[0])
            yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2])
            yB = min(b1[3], b2[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter <= 0:
                return 0.0
            a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
            a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
            union = a1 + a2 - inter
            return float(inter / union) if union > 0 else 0.0

        def _mask_features(mask_bool):
            """
            Ritorna: area_px, perim_px, feret_max_px, feret_min_px,
                     ell_major_px, ell_minor_px, ell_ecc, cx_px, cy_px,
                     equiv_diam_px
            """
            mb = mask_bool.astype("uint8")
            # search of the contours of the image. Return an array of points around the contours
            # RETR_EXTERNAL : ignore the internal hole
            # CHAIN_APPOX_NON : no simplification of the contour found
            cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                # if no contour found, return this default values
                return (0, 0.0, 0.0, 0.0, math.nan, math.nan, math.nan, math.nan, math.nan, 0.0)

            # on conserve le contour avec la plus grande aire
            cnt = max(cnts, key=cv2.contourArea)
            area_px = int(cv2.contourArea(cnt))
            perim_px = float(cv2.arcLength(cnt, True))

            rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
            w, h = rect[1]
            # distance de feret == distance max entre 2 points du contour 
            # -> on prend la largeur et la hauteur du rectangle circonscrit (incline) au contour
            feret_max_px = float(max(w, h))
            feret_min_px = float(min(w, h))

            # fit the contour with an ellipse, 5 points min is needed
            ell_major_px = math.nan
            ell_minor_px = math.nan
            ell_ecc = math.nan
            if len(cnt) >= 5:
                (_, (MA, ma), _) = cv2.fitEllipse(cnt)  # MA, ma = Major Axis, minor axis length
                major = max(MA, ma)
                minor = min(MA, ma)
                ell_major_px = float(major)
                ell_minor_px = float(minor)
                # exentricity of the ellipse
                ell_ecc = float(math.sqrt(1.0 - (minor / major) ** 2)) if major > 0 else math.nan

            M = cv2.moments(cnt)
            # calcul des moments du contours m00 = somme des pixel; m10 = somme x * pixel
            if M["m00"] != 0:
                # on calcule le centroid du contour
                cx_px = float(M["m10"] / M["m00"])
                cy_px = float(M["m01"] / M["m00"])
            else:
                cx_px = cy_px = math.nan

            equiv_diam_px = float(2.0 * math.sqrt(area_px / math.pi)) if area_px > 0 else 0.0
            return (area_px, perim_px, feret_max_px, feret_min_px,
                    ell_major_px, ell_minor_px, ell_ecc, cx_px, cy_px, equiv_diam_px)

        # ---- immagini ----
        print("Load image paths")
        image_paths = get_image_paths(directory_path)
        image_paths_sub = image_paths[:]  # no-op ma esplicito

        print("Run instance segmentation model and save data")
        Bounding_Box = np.empty((0, 7), dtype=float)
        bubble_size = []
        vapor = []
        vapor_base = []

        for i in tqdm(range(len(image_paths_sub))):
            img = cv2.imread(image_paths_sub[i])
            if img is None:
                continue

            outputs = predictor(img)
            inst_cpu = outputs["instances"].to("cpu")
            boxes_all = inst_cpu.pred_boxes.tensor.numpy().tolist()
            masks_all = inst_cpu.pred_masks.numpy()
            scores_all = inst_cpu.scores.numpy().tolist()
            classes_all = inst_cpu.pred_classes.numpy().tolist()

            # ---- RICH: calcolo feature su detection filtrate ----
            if SAVE_RICH:
                boxes = inst_cpu.pred_boxes.tensor.numpy()
                scores = inst_cpu.scores.numpy()
                clss = inst_cpu.pred_classes.numpy()
                masks = inst_cpu.pred_masks.numpy()

                # filtro principale: score > thres
                keep = scores > thres

                # (opzionale ma coerente con la tua logica dei file .txt)
                # escludo box con y2 <= 0; includo la "finestra" storica se presente
                if boxes.shape[0] > 0:
                    y2 = boxes[:, 3]
                    cond = (y2 > 0) | ((y2 > 502) & (y2 < 533))  # stessa tua condizione base
                    keep = keep & cond

                boxes_k = boxes[keep]
                scores_k = scores[keep]
                clss_k = clss[keep]
                masks_k = masks[keep]

                det_count = 0
                for j in range(len(scores_k)):
                    x1, y1, x2, y2 = [int(round(v)) for v in boxes_k[j].tolist()]
                    mbool = masks_k[j]

                    (area_px, perim_px, feret_max_px, feret_min_px,
                     ell_major_px, ell_minor_px, ell_ecc, cx_px, cy_px,
                     equiv_diam_px) = _mask_features(mbool)

                    # conversioni in mm/mm^2 (SOLO scala dell'utente)
                    perim_mm = perim_px * mm_per_px
                    feret_max_mm = feret_max_px * mm_per_px
                    feret_min_mm = feret_min_px * mm_per_px
                    ell_major_mm = (ell_major_px * mm_per_px) if not math.isnan(ell_major_px) else math.nan
                    ell_minor_mm = (ell_minor_px * mm_per_px) if not math.isnan(ell_minor_px) else math.nan
                    equiv_diam_mm = equiv_diam_px * mm_per_px
                    area_mm2 = area_px * (mm_per_px ** 2)
                    cx_mm = cx_px * mm_per_px
                    cy_mm = cy_px * mm_per_px

                    row = {
                        "frame": i + 1,
                        "det_in_frame": det_count,
                        "track_id": -1,  # lo inseriamo dopo, a tracking generato
                        "x1": x1, "y1": y1, "x2": y2 if False else x2, "y2": y2,  # (keep names; ensure x2 is x2)
                        "score": float(scores_k[j]),
                        "class_id": int(clss_k[j]),
                        "area_px": area_px,
                        "perim_px": perim_px,
                        "feret_max_px": feret_max_px,
                        "feret_min_px": feret_min_px,
                        "ell_major_px": ell_major_px,
                        "ell_minor_px": ell_minor_px,
                        "ell_ecc": ell_ecc,
                        "cx_px": cx_px,
                        "cy_px": cy_px,
                        "equiv_diam_px": equiv_diam_px,
                        "mm_per_px": mm_per_px,
                        "area_mm2": area_mm2,
                        "perim_mm": perim_mm,
                        "feret_max_mm": feret_max_mm,
                        "feret_min_mm": feret_min_mm,
                        "ell_major_mm": ell_major_mm,
                        "ell_minor_mm": ell_minor_mm,
                        "cx_mm": cx_mm,
                        "cy_mm": cy_mm,
                        "equiv_diam_mm": equiv_diam_mm,
                    }
                    _rows_rich.append(row)

                    key = f"{i + 1}_{det_count}"
                    if SAVE_MASKS:
                        _mask_store[key] = mbool
                    if SAVE_CONTOURS:
                        mb = mbool.astype("uint8")
                        cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if cnts:
                            cnt = max(cnts, key=cv2.contourArea)
                            _contour_store[key] = cnt[:, 0, :].astype(int).tolist()
                        else:
                            _contour_store[key] = []
                    det_count += 1

            # ---- costruzione Bounding_Box (file storico) ----
            keep_idx = []
            conv_boxes = []
            for j, (x1, y1, x2, y2) in enumerate(boxes_all):
                if y2 > 0 or (502 < y2 < 533 and 320 < x2 < 515):
                    keep_idx.append(j)
                    conv_boxes.append([x1, y1, x2, y2])

            if len(conv_boxes) > 0:
                sel_scores = [scores_all[j] for j in keep_idx]
                sel_classes = [classes_all[j] for j in keep_idx]
                box_data = [[i + 1] + conv_boxes[j] + [sel_scores[j]] + [sel_classes[j]] for j in
                            range(len(conv_boxes))]
                Bounding_Box = np.vstack([Bounding_Box, np.array(box_data, dtype=float)])

            # ---- metriche vapor / bubble_size con soglia thres (stessa logica) ----
            scores_t = torch.as_tensor(scores_all)
            if scores_t.numel() > 0:
                idx_keep = torch.nonzero(scores_t > thres).squeeze(1)
                if idx_keep.numel() > 0:
                    masks_keep = torch.as_tensor(masks_all)[idx_keep]
                    classes_np = np.array(classes_all, dtype=int)
                    cls_keep = classes_np[idx_keep.cpu().numpy()]

                    combined = torch.any(masks_keep, dim=0)
                    vapor.append(int(torch.sum(combined).item()))

                    base_idx = np.where(cls_keep == 0)[0]
                    if base_idx.size > 0:
                        masks_base = masks_keep[torch.from_numpy(base_idx).to(masks_keep.device)]
                        combined_b = torch.any(masks_base, dim=0)
                        vapor_base.append(int(torch.sum(combined_b).item()))
                    else:
                        vapor_base.append(0)

                    pix = torch.sum(masks_keep, dim=(1, 2)).cpu().numpy()
                    bubble_size.append(pix)
                else:
                    vapor.append(0)
                    vapor_base.append(0)
                    bubble_size.append(np.array([], dtype=float))
            else:
                vapor.append(0)
                vapor_base.append(0)
                bubble_size.append(np.array([], dtype=float))

        # ---- Save vapor e bubble_size (invariato) ----
        np.save(vapor_file, np.array(vapor, dtype=object))
        np.save(vapor_base_file, np.array(vapor_base, dtype=object))
        np.save(bubble_size_file, np.array(bubble_size, dtype=object), allow_pickle=True)

        # ---- Save bounding box data storico (frame,x1,y1,x2,y2,score,class) ----
        with open(file_path, 'w') as f:
            for row in Bounding_Box.tolist():
                row[0] = int(round(row[0]))
                row[-1] = int(round(row[-1]))
                f.write(','.join(
                    [str(row[0])] +
                    [f'{float(v):.4f}' for v in row[1:-1]] +
                    [str(row[-1])]
                ) + '\n')

        print("Perform OCSort tracking on saved data")
        tracker = ocsort.OCSort(det_thresh=thres, max_age=10, min_hits=20)
        img_info = (1024, 1024)
        img_size = (1024, 1024)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error opening video file")

        # ricarica le detections per frame (x1,y1,x2,y2,score) per il tracker
        frame_data = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                data = list(map(float, parts[1:-1]))  # x1,y1,x2,y2,score
                frame_data.setdefault(frame_id, []).append(data)

        with open(output_file_path, 'w') as f:
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                xyxyc = frame_data.get(i + 1, [])
                i += 1
                arr = np.array(xyxyc, dtype=float)
                if arr.size == 0:
                    arr = np.empty((0, 5), dtype=float)
                _ = tracker.update(arr, img_info, img_size)
                for tr in tracker.trackers:
                    track_id = tr.id
                    hits = tr.hits
                    x1, y1, x2, y2 = np.round(tr.get_state()).astype(int).squeeze()
                    f.write(f'{i},{track_id},{hits},{x1},{y1},{x2},{y2}\n')
            cap.release()
            del cap

        # === ASSEGNAZIONE TRACK_ID NEL RICH CSV (*** DOPO *** aver scritto l'output) ===
        if SAVE_RICH:
            tracks_by_frame = {}
            if os.path.isfile(output_file_path):
                with open(output_file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 7:
                            continue
                        fr = int(parts[0])
                        tid = int(parts[1])
                        x1 = int(parts[3]);
                        y1 = int(parts[4]);
                        x2 = int(parts[5]);
                        y2 = int(parts[6])
                        tracks_by_frame.setdefault(fr, []).append((tid, (x1, y1, x2, y2)))

            for row in _rows_rich:
                fr = row["frame"]
                det_box = (row["x1"], row["y1"], row["x2"], row["y2"])
                best_tid, best_iou = -1, 0.0
                for (tid, tbox) in tracks_by_frame.get(fr, []):
                    iou = _bbox_iou(det_box, tbox)
                    if iou > best_iou:
                        best_iou, best_tid = iou, tid
                row["track_id"] = int(best_tid) if best_iou >= IOU_THRESH_TID else -1

            # salva CSV
            rich_csv = os.path.join(self.savefolder, f"{RICH_BASENAME}.csv")
            if len(_rows_rich) > 0:
                with open(rich_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(_rows_rich[0].keys()))
                    w.writeheader()
                    w.writerows(_rows_rich)
                print(f"[RICH] CSV salvato: {rich_csv}  (rows={len(_rows_rich)})")
            else:
                print("[RICH] Nessuna detection sopra soglia: CSV non scritto.")

            # opzionali
            if SAVE_MASKS and len(_mask_store) > 0:
                masks_npz = os.path.join(self.savefolder, f"masks_{self.extension}.npz")
                np.savez_compressed(masks_npz, **_mask_store)
                print(f"[RICH] NPZ maschere salvato: {masks_npz}  (keys={len(_mask_store)})")

            if SAVE_CONTOURS and len(_contour_store) > 0:
                contours_json = os.path.join(self.savefolder, f"contours_{self.extension}.json")
                with open(contours_json, "w", encoding="utf-8") as f:
                    json.dump(_contour_store, f)
                print(f"[RICH] JSON contorni salvato: {contours_json}  (keys={len(_contour_store)})")

        print("Match tracking results to bubble indices")

        # ---- ricostruzione matching/indici (invariata nella logica) ----
        real_data = np.loadtxt(file_path, delimiter=",")
        if real_data.ndim == 1:
            real_data = real_data[None, :]
        keep = real_data[:, -2] >= thres
        real = real_data[keep][:, 0:-2]
        real = np.round(real).astype(int)

        pred_data = np.loadtxt(output_file_path, delimiter=",").astype(int)
        if pred_data.ndim == 1:
            pred_data = pred_data[None, :]

        # realg: bboxes per frame
        if real.size > 0:
            max_frame_real = int(real[:, 0].max())
        else:
            max_frame_real = 0
        real_dict = {i: [] for i in range(1, max_frame_real + 1)}
        for row in real:
            fr = int(row[0])
            real_dict.setdefault(fr, []).append(list(row[1:5]))
        realg = [real_dict.get(i, []) for i in range(1, max_frame_real + 1)]

        # predg: per frame -> [track_id, hits, x1,y1,x2,y2]
        pred_arr = np.array(pred_data, dtype=object)
        frame_ids = pred_arr[:, 0].astype(int)
        max_frame_pred = int(frame_ids.max()) if frame_ids.size > 0 else 0
        pred_dict = {i: [] for i in range(0, max_frame_pred + 1)}
        for row in pred_arr:
            key = int(row[0])
            pred_dict.setdefault(key, []).append(list(row[1:]))
        predg = [pred_dict.get(i, []) for i in range(0, max_frame_pred + 1)]

        # ricostruzione tracks/values (stessa logica, robusta)
        n_frames = len(predg)
        tracks = [[] for _ in range(n_frames)]
        values = [[] for _ in range(n_frames)]

        if n_frames > 0 and len(predg[0]) > 0:
            arr0 = np.asarray(predg[0], dtype=object)
            tracks[0] = np.asarray(arr0[:, 0], dtype=int).tolist()
            values[0] = [list(map(int, v)) for v in np.asarray(arr0[:, 2:], dtype=int)]
        elif n_frames > 0:
            tracks[0] = []
            values[0] = []

        for k in range(n_frames - 1):
            if len(predg[k]) == 0 or len(predg[k + 1]) == 0:
                tracks[k + 1] = []
                values[k + 1] = []
                continue

            frame1 = np.asarray(predg[k], dtype=object)
            frame2 = np.asarray(predg[k + 1], dtype=object)

            vector2 = frame1[:, 0].astype(int).tolist()  # id nel frame k
            vector1 = frame2[:, 0].astype(int).tolist()  # id nel frame k+1
            idx_map = {v: idx for idx, v in enumerate(vector2)}

            new_tracks, new_values = [], []
            for ii, v in enumerate(vector1):
                j = idx_map.get(v, -1)
                if j != -1:
                    # tieni solo se "colonna 1" (hits) cambia
                    if frame2[ii][1] != frame1[j][1]:
                        new_tracks.append(int(frame2[ii][0]))
                        new_values.append(list(map(int, frame2[ii][2:])))
                else:
                    new_tracks.append(int(frame2[ii][0]))
                    new_values.append(list(map(int, frame2[ii][2:])))

            tracks[k + 1] = new_tracks
            values[k + 1] = new_values

        # riordino con IoU rispetto a realg (robusto agli edge-cases)
        safe_n = min(len(values), len(realg))
        for i in range(safe_n):
            if not values[i] or not realg[i]:
                continue
            a = np.asarray(realg[i], dtype=float)
            b = np.asarray(values[i], dtype=float)
            if a.ndim == 1: a = a[None, :]
            if b.ndim == 1: b = b[None, :]
            if a.shape[1] > 4: a = a[:, :4]
            if b.shape[1] > 4: b = b[:, :4]
            if a.size == 0 or b.size == 0:
                continue
            iou = iou_batch(a, b)
            if iou.size == 0:
                continue
            sort = np.argmax(iou, axis=1).tolist()
            t_i = np.asarray(tracks[i], dtype=object)
            v_i = np.asarray(values[i], dtype=object)
            m = min(len(sort), len(t_i), len(v_i))
            if m > 0:
                sort = sort[:m]
                tracks[i] = t_i[sort].tolist()
                values[i] = v_i[sort].tolist()

        # frames: per track_id -> lista dei frame dove appare
        data = tracks
        if any(len(x) for x in data):
            max_number = int(max(max(sub, default=-1) for sub in data if sub))
        else:
            max_number = -1
        frames = [[] for _ in range(max_number + 1)] if max_number >= 0 else []
        for fr_idx, row in enumerate(data):
            for tid in row:
                frames[tid].append(fr_idx)

        # bubInd: per track_id -> indici nel frame
        bubInd = [[] for _ in range(max_number + 1)] if max_number >= 0 else []
        for row in data:
            for idx, tid in enumerate(row):
                bubInd[tid].append(idx)

        # save (invariato)
        np.save(bubind_file, np.array(bubInd, dtype=object))
        np.save(frameind_file, np.array(frames, dtype=object))

        # classi per frame (dalla tabella real_data)
        classes_all = real_data[:, -1] if real_data.size > 0 else np.array([])
        if real_data.size > 0:
            max_frame_real2 = int(real_data[:, 0].max())
        else:
            max_frame_real2 = 0
        cls_dict = {i: [] for i in range(1, max_frame_real2 + 1)}
        for row in real_data:
            fr = int(row[0])
            cls = int(row[-1])
            cls_dict.setdefault(fr, []).append(cls)
        realgG = [cls_dict.get(i, []) for i in range(1, max_frame_real2 + 1)]
        np.save(classind_file, np.array(realgG, dtype=object))

        # bub_class: classi lungo il tempo per ogni track
        bub_class = copy.deepcopy(frames)
        for j in range(len(bubInd)):
            for k in range(len(frames[j])):
                fr = frames[j][k]
                ix = bubInd[j][k]
                if fr < len(realgG) and ix < len(realgG[fr]):
                    bub_class[j][k] = realgG[fr][ix]
                else:
                    bub_class[j][k] = -1
        np.save(bubclassind_file, np.array(bub_class, dtype=object))
        print("Finish")

    def Plotvf(self):
        vf_path = self.savefolder + f'vapor_{self.extension}.npy'
        vidstart = 0
        vf = np.load(vf_path) / (832 * 600)
        time = [(i / 150) + vidstart for i in range(len(vf))]
        df = pd.DataFrame(data=vf)
        df['value'] = df.iloc[:, 0].rolling(window=300).mean()
        plt.figure(figsize=(5, 10))
        fig, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(time, df, color='lightgray')
        ax1.plot(time, df['value'], color='darkblue', label='Rolling Average')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Vapor Fraction')
        ax1.legend()
        saveloc = self.savefolder + f'vaporfig_{self.extension}.png'
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()

    def Plotbc(self):
        bs_path = self.savefolder + f'/bubble_size_bt-{self.extension}.npy'
        bs = np.load(bs_path, allow_pickle=True)
        count = []
        for i in range(len(bs)):
            count.append(len(bs[i]))
        df = pd.DataFrame(data=count)

        vidstart = 0
        time = [(i / 150) + vidstart for i in range(len(count))]
        df['value'] = df.iloc[:, 0].rolling(window=300).mean()
        plt.figure(figsize=(5, 10))
        fig, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(time, df, color='lightgray')
        ax1.plot(time, df['value'], color='darkblue', label='Rolling Average')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Bubble Count')
        ax1.legend()
        saveloc = self.savefolder + f'bcfig_{self.extension}.png'
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()

    def PlotInterfaceVelocity(self, bubble):
        directory_path = self.imagesfolder

        bubind_file = self.savefolder + f'/bubind_{self.extension}.npy'
        frameind_file = self.savefolder + f'/frames_{self.extension}.npy'

        # load model
        cfg = get_cfg()
        cfg.OUTPUT_DIR = self.modeldirectory
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.modelweights)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        # if self.device == 'cpu':
        #     cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)

        print("Model Loaded")
        bubInd = np.load(bubind_file, allow_pickle=True)
        frames = np.load(frameind_file, allow_pickle=True)

        def get_image_paths(directory):
            """
            Get a list of file paths for all image files in the specified directory and its subdirectories.

            Args:
            directory (str): The directory to search for image files.

            Returns:
            List[str]: A list of file paths for all image files found, sorted alphabetically.
            """
            image_extensions = ['*.jpg']  # Add more extensions as needed

            image_paths = []
            for extension in image_extensions:
                pattern = os.path.join(directory, '**', extension)
                image_paths.extend(glob.glob(pattern, recursive=True))

            return sorted(image_paths)  # Sort the list of image paths alphabetically

        # Get a list of image file paths sorted alphabetically
        image_paths = get_image_paths(directory_path)

        # Set the output video file name
        output_file = f'./vel_bubble{bubble}1.avi'
        frame_width = 832
        frame_height = 600
        '''
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 10, (frame_width, frame_height), isColor=True)
        '''
        skip = 5
        angles = []
        # Save Contours of a bubble in each frame

        image_paths_sub = image_paths[0:]

        # output_video = 'bubble0.avi'
        # frame_size = (832, 600)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(output_video, fourcc, 20.0, frame_size, isColor=False)
        # bubble=12
        contours = []
        centroids = []
        values = []
        # for i in tqdm(range(len(frames[bubble]))):
        indexs = []
        print(len(frames[bubble]))
        for i in tqdm(range(len(frames[bubble]))):
            new_im = cv2.imread(image_paths_sub[frames[bubble][i]])
            outputs = predictor(new_im)
            box = outputs["instances"].pred_boxes
            box = box.tensor
            box = box.cpu().tolist()
            masks = outputs["instances"].pred_masks.cpu()
            scores = outputs["instances"].scores
            scores = scores.cpu().tolist()
            k = 0
            for j in range(len(box)):
                x1, y1, x2, y2 = box[j]
                if y2 > -1000:
                    if k == bubInd[bubble][i]:
                        mask = np.uint8(masks[j]) * 255
                        contours1, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                        if len(contours1) > 1:
                            largest_contour = max(contours1, key=cv2.contourArea)
                            contours1 = np.array(largest_contour).reshape((-1, 2))
                        else:
                            contours1 = np.array(contours1).reshape((-1, 2))
                            # out.write(mask)
                        contours.append(contours1)
                        indexs.append(j)
                    k += 1

        avg_mag = []
        mag = []
        contour_connections = []
        for j in range(len(frames[bubble]) - skip - 1):

            newimg = cv2.imread(image_paths_sub[frames[bubble][j]])

            frame1 = frames[bubble][j]
            frame2 = frames[bubble][j + skip]
            tree_set1 = cKDTree(contours[j + skip])
            distances, indices = tree_set1.query(contours[j], k=1)
            if j == 1:
                fig, ax = plt.subplots()
                ax.imshow(newimg)
                ax.scatter(contours[j][:, 0], contours[j][:, 1], label='Set 1', marker='o', s=1)
                ax.scatter(contours[j + skip][:, 0], contours[j + skip][:, 1], label='Set 2', marker='x', s=1)
                plt.show()

            velocitys = (np.array(distances) / 184) / ((frame2 - frame1) / 3000)
            distances = velocitys
            avg_mag.append(np.mean(np.array(distances)))
            mag.append(list(distances))
            # Connect the paired points
            ang = []
            # print(len(contours[j]), len(indices), max(indices))
            '''
            if j ==0:
                values.append(indices[0])
            else:
                values.append(indices[values[-1]])
            '''

            for i in range(0, len(contours[j]), 1):
                point1 = contours[j][i]
                point2 = contours[j + skip][indices[i]]
                vector = point1 - point2
                angle_rad = np.arctan2(vector[1], vector[0])
                angle_rad = np.degrees(angle_rad)
                if angle_rad < 0:
                    angle_rad += 360
                ang.append(angle_rad)
                # ax.arrow(point1[0], point1[1], point2[0] - point1[0], point2[1] - point1[1],
                #         head_width=4, head_length=6, fc='k', ec='k', linewidth=0.5)
            angles.append(ang)
            contour_connections.append(contours[j + skip][indices])

            direction = []
        for i in range(len(contours) - skip - 1):
            # for i in range(1):
            new_im = cv2.imread(image_paths_sub[frames[bubble][i]])
            outputs = predictor(new_im)
            masks = outputs["instances"].pred_masks.cpu()
            mask = np.uint8(masks[indexs[i]]) * 255
            class_val = []
            for j in range(len(contours[i])):
                x_coord = contour_connections[i][j][0]
                y_coord = contour_connections[i][j][1]
                if x_coord >= 832:
                    x_coord = 831
                elif x_coord <= 0:
                    x_coord = 0

                if y_coord <= 0:
                    y_coord = 0
                elif y_coord >= 600:
                    y_coord = 599

                if mask[y_coord][x_coord] == 255:
                    class_val.append(1)
                else:
                    class_val.append(0)
            direction.append(class_val)

        for i in range(len(direction)):
            for j in range(len(direction[i])):
                if direction[i][j] == 1:
                    mag[i][j] = mag[i][j] * -1

        length = len(mag[0])
        for i in range(len(mag)):
            if len(mag[i]) <= length:
                length = len(mag[i])

        num_entries = 200
        data = np.empty((len(mag), num_entries))
        # Calculate indices for evenly spaced entries
        for i in range(len(mag)):
            indices = np.linspace(0, len(mag[i]) - 1, num_entries, dtype=int)
            data[i:] = np.array(mag[i])[indices]
        # data=data[:-20]

        data1 = np.empty((len(mag), num_entries))
        # data1=data1[:-20]
        split_val = 100
        data1[:, 0:-split_val] = data[:, split_val:]
        data1[:, -split_val:] = data[:, 0:split_val]

        data = data1

        data_smoothed = gaussian_filter(data, sigma=2)

        # Generate some data
        total_time = len(data) / 3000
        # Plot the data with a color bar

        image = plt.imshow(data_smoothed.T, extent=[0, total_time, 0, 200], aspect='auto', cmap='turbo', alpha=1)
        cbar = plt.colorbar()
        cbar.set_label('Velocity Magnitude (cm/s)')  # Set label for the color bar
        image.set_clim(vmin=-30, vmax=30)  # Set the range of values to display

        plt.xticks(np.arange(0, total_time, 0.01), [str(i) for i in np.arange(0, total_time, 0.01)])

        plt.xlabel('Time (s)')
        plt.ylabel('Location Along Bubble \n Perimeter')
        # Control the range of values displayed on the color bar
        saveloc = self.savefolder + f'velocity_{self.extension}_{bubble}.png'
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()

    def show_bubble_contours(self, frame_indices, thres=0.5, out_dir=None):
        """
        Disegna i contorni delle bolle (già salvati da GenerateData) sui frame indicati.
        Usa rich_<EXT>.csv per filtrare per punteggio e contours_<EXT>.json per i contorni.
        NON usa il predictor.

        Parametri
        ---------
        frame_indices : list[int]
            Indici 0-based dei file immagine (frame_000.jpg -> idx=0, ecc.).
        thres : float
            Soglia di confidenza (score) per selezionare le bolle da disegnare.
        out_dir : str | None
            Cartella di salvataggio PNG; default: <savefolder>/visual
        """
        import os, csv, json
        import numpy as np
        import cv2

        # --- percorsi
        if out_dir is None:
            out_dir = os.path.join(self.savefolder, "visual")
        os.makedirs(out_dir, exist_ok=True)

        rich_csv = os.path.join(self.savefolder, f"rich_{self.extension}.csv")
        contours_json = os.path.join(self.savefolder, f"contours_{self.extension}.json")

        if not os.path.isfile(rich_csv):
            raise FileNotFoundError(f"rich CSV non trovato: {rich_csv}. Esegui GenerateData con SAVE_RICH=True.")
        if not os.path.isfile(contours_json):
            print(f"[WARN] contours JSON non trovato: {contours_json}. Disegnerò le bbox al posto dei contorni.")
            contours = {}
        else:
            with open(contours_json, "r", encoding="utf-8") as f:
                contours = json.load(f)  # chiavi: "frame_detInFrame" -> [[x,y],...]

        # --- carica e indicizza le righe del CSV per frame
        frame_rows = {}  # frame_id(1-based) -> [row, ...]
        with open(rich_csv, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    fr = int(float(r.get("frame", -1)))
                    scr = float(r.get("score", 0.0))
                    deti = int(float(r.get("det_in_frame", -1)))
                except:
                    continue
                if scr < thres or fr < 0 or deti < 0:
                    continue
                # normalizza tipi per sicurezza
                row = {}
                for k, v in r.items():
                    if k in ("frame", "det_in_frame", "track_id", "class_id"):
                        try:
                            row[k] = int(float(v))
                        except:
                            row[k] = -1
                    else:
                        try:
                            row[k] = float(v)
                        except:
                            row[k] = v
                frame_rows.setdefault(fr, []).append(row)

        # colore deterministico da track_id (BGR per OpenCV)
        def _color_for_id(tid: int):
            if tid is None or tid < 0:
                return (160, 160, 160)  # grigio per "senza ID"
            r = (37 * (tid + 1)) % 255
            g = (17 * (tid + 1)) % 255
            b = (29 * (tid + 1)) % 255
            return (int(b), int(g), int(r))

        # --- loop sui frame richiesti
        for idx in frame_indices:
            frame_id = int(idx) + 1  # CSV usa frame 1-based
            img_path = os.path.join(self.imagesfolder, f"frame_{idx:03d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                print(f"[skip] frame {idx:03d} non trovato: {img_path}")
                continue

            rows = frame_rows.get(frame_id, [])
            if not rows:
                # niente detections sopra soglia
                out_png = os.path.join(out_dir, f"frame_vis_{idx:03d}.png")
                cv2.imwrite(out_png, img)
                print(f"[saved] {out_png} (nessuna bolla >= {thres})")
                continue

            vis = img.copy()
            for r in rows:
                tid = r.get("track_id", -1)
                color = _color_for_id(tid)

                # prova a disegnare il contorno se disponibile
                key = f"{frame_id}_{r.get('det_in_frame', -1)}"
                pts = contours.get(key, None)

                if pts and len(pts) > 0:
                    cnt = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(vis, [cnt], -1, color, 2)
                else:
                    # fallback: disegna la bbox
                    x1 = int(round(r.get("x1", 0)));
                    y1 = int(round(r.get("y1", 0)))
                    x2 = int(round(r.get("x2", 0)));
                    y2 = int(round(r.get("y2", 0)))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # etichetta piccola (ID + score)
                txt = f"id:{tid}  s:{r.get('score', 0):.2f}"
                org = (int(round(r.get("x1", 0))) + 5, int(round(r.get("y1", 0))) - 6)
                cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            out_png = os.path.join(out_dir, f"frame_vis_{idx:03d}.png")
            cv2.imwrite(out_png, vis)
            print(f"[saved] {out_png}")

    def make_tracked_video(
            self,
            n_frames=50,
            out_name=None,
            fps=10,
            contour_thickness=2,
            iou_match_thresh=0.5,
            score_thres=0.7,
            class_idx_attached=2,
            class_idx_detached=0,
            class_idx_unknown=1,
    ):
        import os, glob, json, re
        import cv2, numpy as np

        if out_name is None:
            out_name = f"tracked_{self.extension}"

        # --- files base ---
        track_glob = glob.glob(os.path.join(self.savefolder, f"*-output-{self.extension}.txt"))
        if not track_glob:
            raise FileNotFoundError(f"File tracking '*-output-{self.extension}.txt' non trovato in {self.savefolder}")
        track_file = track_glob[0]
        contours_path = os.path.join(self.savefolder, f"contours_{self.extension}.json")
        if not os.path.exists(contours_path):
            raise FileNotFoundError(f"'{contours_path}' non trovato. Esegui GenerateData(..., save_contours=True).")
        rich_csv = os.path.join(self.savefolder, f"rich_{self.extension}.csv")
        use_rich = os.path.exists(rich_csv)

        # --- lista frame ---
        def _natkey(p):
            m = re.findall(r'\d+', os.path.basename(p))
            return int(m[-1]) if m else p

        img_list = sorted(glob.glob(os.path.join(self.imagesfolder, "frame_*.jpg")), key=_natkey)
        if not img_list:
            img_list = sorted(glob.glob(os.path.join(self.imagesfolder, "frame_*.png")), key=_natkey)
        if not img_list:
            raise FileNotFoundError(f"Nessun frame trovato in {self.imagesfolder}")
        n_frames = min(n_frames, len(img_list))

        # --- tracking (1-based) ---
        tracks = {}
        with open(track_file, "r") as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) < 7: continue
                fr, tid = int(p[0]), int(p[1])
                x1, y1, x2, y2 = map(int, p[3:7])
                tracks.setdefault(fr, []).append((tid, x1, y1, x2, y2))

        rng = np.random.default_rng(seed=42)
        palette = {tid: tuple(int(c) for c in rng.integers(40, 255, size=3))
                   for tid in {t for lst in tracks.values() for t, *_ in lst}}

        # --- contours ---
        with open(contours_path, "r") as f:
            contours_data = json.load(f)
        idx_map, frame_contours = {}, {}

        def _ensure_numpy(poly):
            arr = np.asarray(poly, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("contour con shape non valida; atteso Nx2")
            return arr

        if isinstance(contours_data, dict):
            key_samples = list(contours_data.keys())[:3]
            looks_like_pairs = any("_" in k for k in key_samples)
            if looks_like_pairs:
                for k, poly in contours_data.items():
                    try:
                        fs, ds = k.split("_");
                        fr1, di = int(fs), int(ds)
                        arr = _ensure_numpy(poly)
                        idx_map[(fr1, di)] = arr
                        frame_contours.setdefault(fr1, []).append(arr)
                    except Exception:
                        pass
            else:
                for fs, items in contours_data.items():
                    try:
                        fr1 = int(fs)
                    except Exception:
                        continue
                    frame_contours.setdefault(fr1, [])
                    if isinstance(items, list):
                        for j, it in enumerate(items):
                            if isinstance(it, dict) and "contour" in it:
                                arr = _ensure_numpy(it["contour"])
                                di = int(it.get("det_index", j))
                                idx_map[(fr1, di)] = arr
                                frame_contours[fr1].append(arr)
                            else:
                                arr = _ensure_numpy(it)
                                frame_contours[fr1].append(arr)
        else:
            raise ValueError("Formato contours_<EXT>.json non riconosciuto")

        # --- opz: rich ---
        df_rich = None
        if use_rich:
            import pandas as pd
            try:
                df_rich = pd.read_csv(rich_csv)
                df_rich.columns = [c.strip().lower() for c in df_rich.columns]
                if "track_id" not in df_rich.columns and "tid" in df_rich.columns:
                    df_rich = df_rich.rename(columns={"tid": "track_id"})
                # normalizza eventuale nome indice detection
                if "det_in_frame" in df_rich.columns and "det_index" not in df_rich.columns:
                    df_rich = df_rich.rename(columns={"det_in_frame": "det_index"})
                if "idx_in_frame" in df_rich.columns and "det_index" not in df_rich.columns:
                    df_rich = df_rich.rename(columns={"idx_in_frame": "det_index"})
                df_rich = df_rich[df_rich["track_id"].fillna(-1).astype(int) >= 0]
            except Exception as e:
                print(f"[warn] impossibile leggere {rich_csv}: {e}")
                df_rich = None
                use_rich = False

        det_file = os.path.join(self.savefolder, f"bb-Boiling-{self.extension}.txt")
        have_det = os.path.exists(det_file)

        # --- utils ---
        def _bbox_iou(b1, b2):
            xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
            xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            a1 = max(0, b1[2] - b1[0] + 1) * max(0, b1[3] - b1[1] + 1)
            a2 = max(0, b2[2] - b2[0] + 1) * max(0, b2[3] - b2[1] + 1)
            den = a1 + a2 - inter
            return inter / den if den > 0 else 0.0

        def _contour_bbox(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            return (x, y, x + w - 1, y + h - 1)

        def _contour_center(cnt):
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x1, y1, x2, y2 = _contour_bbox(cnt)
            return ((x1 + x2) // 2, (y1 + y2) // 2)

        # --- writer ---
        img0 = cv2.imread(img_list[0], cv2.IMREAD_UNCHANGED)
        if img0 is None: raise FileNotFoundError(img_list[0])
        if img0.ndim == 3 and img0.shape[2] == 4:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        H, W = img0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_path = os.path.join(self.savefolder, f"{out_name}.avi")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join(self.savefolder, f"{out_name}.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError("Impossibile aprire VideoWriter.")

        # --- loop ---
        for idx in range(n_frames):
            img = cv2.imread(img_list[idx], cv2.IMREAD_UNCHANGED)
            if img is None: continue
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            base = os.path.basename(img_list[idx])
            m = re.findall(r'\d+', base)
            frame0 = int(m[-1]) if m else idx
            frame1 = frame0 + 1

            to_draw = []  # (tid, cnt)
            if use_rich and df_rich is not None:
                rows = df_rich[df_rich["frame"] == frame1]
                if "score" in rows.columns:
                    rows = rows[rows["score"] >= float(score_thres)]  # <<<<<<<<<<<<<< filtro
                for _, r in rows.iterrows():
                    tid = int(r["track_id"])
                    cnt = None
                    det_idx = None
                    if "det_index" in r and not (r["det_index"] != r["det_index"]):
                        det_idx = int(r["det_index"])
                    if det_idx is not None and (frame1, det_idx) in idx_map:
                        cnt = idx_map[(frame1, det_idx)]
                    else:
                        det_bb = None
                        if {"x1", "y1", "x2", "y2"}.issubset(rows.columns):
                            det_bb = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
                        if det_bb is None:
                            for ttid, x1, y1, x2, y2 in tracks.get(frame1, []):
                                if ttid == tid: det_bb = (x1, y1, x2, y2); break
                        if det_bb is not None:
                            best_iou, best_cnt = -1, None
                            for cnt_cand in frame_contours.get(frame1, []):
                                iou = _bbox_iou(det_bb, _contour_bbox(cnt_cand))
                                if iou > best_iou:
                                    best_iou, best_cnt = iou, cnt_cand
                            if best_iou >= iou_match_thresh and best_cnt is not None:
                                cnt = best_cnt
                    if cnt is not None:
                        to_draw.append((tid, cnt))
            else:
                # fallback: uso detections con score >= thres per aiutare il match
                dets = []
                if have_det:
                    with open(det_file, "r") as fdet:
                        for line in fdet:
                            pp = line.strip().split(",")
                            if len(pp) < 7: continue
                            fr = int(pp[0])
                            if fr != frame1: continue
                            x1, y1, x2, y2 = map(int, pp[1:5])
                            sc = float(pp[5])
                            if sc >= float(score_thres):  # <<<<<<<<<<<<<< filtro
                                dets.append((x1, y1, x2, y2))
                for tid, tx1, ty1, tx2, ty2 in tracks.get(frame1, []):
                    tbb = (tx1, ty1, tx2, ty2)
                    match_bb = tbb
                    if dets:
                        best_iou, best_bb = -1, None
                        for dbb in dets:
                            iou = _bbox_iou(tbb, dbb)
                            if iou > best_iou:
                                best_iou, best_bb = iou, dbb
                        if best_iou >= iou_match_thresh and best_bb is not None:
                            match_bb = best_bb
                    best_iou, best_cnt = -1, None
                    for cnt_cand in frame_contours.get(frame1, []):
                        iou = _bbox_iou(match_bb, _contour_bbox(cnt_cand))
                        if iou > best_iou:
                            best_iou, best_cnt = iou, cnt_cand
                    if best_iou >= iou_match_thresh and best_cnt is not None:
                        to_draw.append((tid, best_cnt))

            for tid, cnt in to_draw:
                color = palette.get(tid, (255, 255, 255))
                cv2.polylines(img, [cnt.reshape(-1, 1, 2)], True, color, contour_thickness)
                cx = int(np.mean(cnt[:, 0]));
                cy = int(np.mean(cnt[:, 1]))
                cv2.putText(img, str(tid), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            writer.write(img)

        writer.release()
        print(f"[OK] Video tracciato (contorni) salvato in: {out_path}")

    def ComputeAttachDwellWithSurface(self,
                                      fps=4000,
                                      class_idx_attached=2,
                                      class_idx_detached=0,
                                      class_idx_unknown=1,
                                      tolerate_unknown_gap=1,
                                      min_attached_run=2,
                                      out_csv=None):
        import os, csv, cv2, numpy as np
        import pandas as pd

        frames_path = os.path.join(self.savefolder, f'frames_{self.extension}.npy')
        bubclass_path = os.path.join(self.savefolder, f'bubclass_{self.extension}.npy')
        rich_csv = os.path.join(self.savefolder, f"rich_{self.extension}.csv")
        if not os.path.isfile(frames_path) or not os.path.isfile(bubclass_path):
            raise FileNotFoundError("frames_*.npy o bubclass_*.npy non trovati.")
        frames_arr = np.load(frames_path, allow_pickle=True)
        bubclass_arr = np.load(bubclass_path, allow_pickle=True)

        # opzionale: score per frame da rich
        df_score = None
        if os.path.isfile(rich_csv):
            df = pd.read_csv(rich_csv)
            df.columns = [c.strip().lower() for c in df.columns]
            if "track_id" not in df.columns and "tid" in df.columns:
                df = df.rename(columns={"tid": "track_id"})
            if "score" in df.columns and "frame" in df.columns:
                df["frame0"] = df["frame"].astype(int) - 1
                # tieni, per ogni (track_id, frame0), la detection con score max
                df = (df.sort_values(["track_id", "frame0", "score"], ascending=[True, True, False])
                      .drop_duplicates(["track_id", "frame0"], keep="first"))
                df_score = df[["track_id", "frame0", "score"]].copy()

        # --- FPS ---
        if fps is None:
            fps = 1.0
            if self.videopath and os.path.isfile(self.videopath):
                cap = cv2.VideoCapture(self.videopath)
                if cap.isOpened():
                    _fps = cap.get(cv2.CAP_PROP_FPS)
                    if _fps and _fps > 0:
                        fps = float(_fps)
                cap.release()

        def _smooth(frames_list, labels_list):
            if not frames_list: return frames_list, labels_list
            idx = np.argsort(frames_list)
            fr = [int(frames_list[i]) for i in idx]
            lb = [int(labels_list[i]) for i in idx]
            out = lb[:]
            if tolerate_unknown_gap > 0:
                i = 0;
                n = len(out)
                while i < n:
                    if out[i] == class_idx_attached:
                        j = i + 1;
                        unk = 0
                        while j < n and out[j] == class_idx_unknown:
                            unk += 1;
                            j += 1
                        if 0 < unk <= tolerate_unknown_gap and j < n and out[j] == class_idx_attached:
                            for k2 in range(i + 1, j): out[k2] = class_idx_attached
                            i = j;
                            continue
                    i += 1
            return fr, out

        results = []
        for tid, (fr_list, cls_list) in enumerate(zip(frames_arr, bubclass_arr)):
            fr_list = list(fr_list) if isinstance(fr_list, (list, np.ndarray)) else []
            cls_list = list(cls_list) if isinstance(cls_list, (list, np.ndarray)) else []
            if not fr_list:
                results.append({"bubble_id": tid, "attach_start_frame": None, "detach_frame": None,
                                "dwell_frames": 0, "dwell_seconds": 0.0,
                                "n_frames_tracked": 0, "n_unknown": 0,
                                "mean_score_pct": np.nan, "note": "no_frames"})
                continue

            n_unknown = sum(1 for c in cls_list if int(c) == class_idx_unknown)
            fr_s, lb_s = _smooth(fr_list, cls_list)

            # trova la prima run attached >= min_attached_run
            attach_start = None;
            attach_end_i = None;
            run = 0;
            start_i = None
            for i, lab in enumerate(lb_s):
                if lab == class_idx_attached:
                    run += 1
                    if attach_start is None:
                        attach_start = fr_s[i];
                        start_i = i
                    attach_end_i = i
                else:
                    if run >= min_attached_run: break
                    attach_start = None;
                    attach_end_i = None;
                    start_i = None;
                    run = 0

            if attach_start is None or run < min_attached_run:
                results.append({"bubble_id": tid, "attach_start_frame": None, "detach_frame": None,
                                "dwell_frames": 0, "dwell_seconds": 0.0,
                                "n_frames_tracked": len(fr_list), "n_unknown": n_unknown,
                                "mean_score_pct": np.nan, "note": "no_attached_run"})
                continue

            detach_frame = None
            for j in range(attach_end_i + 1, len(lb_s)):
                if lb_s[j] == class_idx_detached:
                    detach_frame = fr_s[j];
                    break

            if detach_frame is None:
                dwell_frames = fr_s[attach_end_i] - attach_start + 1
                end_frame = fr_s[attach_end_i]
                note = "no_detach_found"
            else:
                dwell_frames = detach_frame - attach_start
                end_frame = detach_frame
                note = "ok"

            dwell_frames = max(int(dwell_frames), 0)
            dwell_seconds = float(dwell_frames) / float(fps if fps else 1.0)

            # --- media score sui frame dell'intervallo ---
            mean_score_pct = np.nan
            if df_score is not None and start_i is not None:
                run_frames = [f for f in fr_s[start_i:] if f <= end_frame]
                if run_frames:
                    s = (df_score[(df_score["track_id"].astype(int) == tid) &
                                  (df_score["frame0"].isin(run_frames))]["score"].astype(float))
                    if len(s) > 0:
                        mean_score_pct = float(s.mean() * 100.0)

            results.append({"bubble_id": tid,
                            "attach_start_frame": attach_start,
                            "detach_frame": detach_frame,
                            "dwell_frames": dwell_frames,
                            "dwell_seconds": dwell_seconds,
                            "n_frames_tracked": len(fr_list),
                            "n_unknown": n_unknown,
                            "mean_score_pct": mean_score_pct,
                            "note": note})

        if out_csv is None:
            out_csv = os.path.join(self.savefolder, f'dwell_{self.extension}.csv')

        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                "bubble_id", "attach_start_frame", "detach_frame",
                "dwell_frames", "dwell_seconds", "n_frames_tracked",
                "n_unknown", "mean_score_pct", "note"
            ])
            w.writeheader()
            for r in results: w.writerow(r)

        print(f"[ComputeAttachDwell] salvato: {out_csv}  (fps={fps})")
        return results

    def ComputeDepartureDiameter(self,
                                 *,
                                 k: int = 3,
                                 fit_kind: str = "linear",
                                 class_idx_attached: int = 2,
                                 class_idx_detached: int = 0,
                                 class_idx_unknown: int = 1,
                                 tolerate_unknown_gap: int = 1,
                                 min_attached_run: int = 1,
                                 out_csv: str | None = None):
        import os, math, csv
        import numpy as np, pandas as pd

        rich_csv = os.path.join(self.savefolder, f"rich_{self.extension}.csv")
        frames_path = os.path.join(self.savefolder, f'frames_{self.extension}.npy')
        bubclass_path = os.path.join(self.savefolder, f'bubclass_{self.extension}.npy')
        if out_csv is None:
            out_csv = os.path.join(self.savefolder, f"departure_{self.extension}.csv")
        if not os.path.isfile(rich_csv):
            raise FileNotFoundError(f"{rich_csv} non trovato.")
        if not os.path.isfile(frames_path) or not os.path.isfile(bubclass_path):
            raise FileNotFoundError("frames_*.npy o bubclass_*.npy non trovati.")
        if not getattr(self, "mm_per_px", None):
            raise RuntimeError("mm_per_px non impostato.")

        mm_per_px = float(self.mm_per_px)
        df = pd.read_csv(rich_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep='first')]
        if "track_id" not in df.columns and "tid" in df.columns:
            df = df.rename(columns={"tid": "track_id"})
        if "frame" not in df.columns: raise ValueError("Manca 'frame' nel rich CSV.")
        df["frame0"] = df["frame"].astype(int) - 1
        df = df[df["track_id"].fillna(-1).astype(int) >= 0]

        need = {"area_px", "perim_px", "feret_max_px", "feret_min_px", "ell_major_px", "ell_minor_px", "score"}
        miss = [c for c in need if c not in df.columns]
        if miss: raise ValueError(f"Mancano colonne nel rich CSV: {miss}")

        frames_arr = np.load(frames_path, allow_pickle=True)
        bubclass_arr = np.load(bubclass_path, allow_pickle=True)

        def _smooth(frames_list, labels_list):
            if not frames_list: return frames_list, labels_list
            idx = np.argsort(frames_list)
            fr = [int(frames_list[i]) for i in idx]
            lb = [int(labels_list[i]) for i in idx]
            out = lb[:]
            if tolerate_unknown_gap > 0:
                i = 0;
                n = len(out)
                while i < n:
                    if out[i] == class_idx_attached:
                        j = i + 1;
                        unk = 0
                        while j < n and out[j] == class_idx_unknown:
                            unk += 1;
                            j += 1
                        if 0 < unk <= tolerate_unknown_gap and j < n and out[j] == class_idx_attached:
                            for k2 in range(i + 1, j): out[k2] = class_idx_attached
                            i = j;
                            continue
                    i += 1
            return fr, out

        def _find_departure(frames0, labels):
            fr_s, lb_s = _smooth(frames0, labels)
            attach_start = None;
            attach_end_i = None;
            run = 0
            for i, lab in enumerate(lb_s):
                if lab == class_idx_attached:
                    run += 1
                    if attach_start is None: attach_start = fr_s[i]
                    attach_end_i = i
                else:
                    if run >= min_attached_run: break
                    attach_start = None;
                    attach_end_i = None;
                    run = 0
            if attach_start is None or run < min_attached_run:
                return None, None, None, None, None
            last_attached = fr_s[attach_end_i]
            detach_frame = None
            for j in range(attach_end_i + 1, len(lb_s)):
                if lb_s[j] == class_idx_detached:
                    detach_frame = fr_s[j];
                    break
            return attach_start, attach_end_i, last_attached, detach_frame, fr_s

        def _series_for_track(tid, col_name):
            # elimina eventuali duplicati di nome colonna (già fatto sopra, ma qui è idempotente)
            dfl = df.loc[:, ~pd.Index(df.columns).duplicated(keep='first')]

            # colonne da prendere: frame0 + col_name (+ score solo se serve come tie-break)
            cols = ["frame0", col_name]
            use_tiebreak = (col_name != "score") and ("score" in dfl.columns)
            if use_tiebreak:
                cols.append("score")

            sdf = dfl[dfl["track_id"].astype(int) == tid][cols].copy()
            if sdf.empty:
                return []

            # se ho 'score' (e NON sto chiedendo proprio la serie score), tie-break per score max
            if use_tiebreak:
                sdf = (sdf.sort_values(["frame0", "score"], ascending=[True, False])
                       .drop_duplicates("frame0", keep="first"))
            else:
                sdf = sdf.sort_values(["frame0"]).drop_duplicates("frame0", keep="first")

            pairs = []
            for r in sdf.itertuples(index=False):
                f0 = int(getattr(r, "frame0"))
                v = getattr(r, col_name)
                try:
                    v = float(v)
                except Exception:
                    continue
                if np.isfinite(v):
                    pairs.append((f0, v))
            return pairs

        def _discrete_at(series, f0):
            if not series: return math.nan
            series = sorted(series, key=lambda x: x[0])
            for f, v in reversed(series):
                if f == f0: return v
                if f < f0:  return v
            return math.nan

        def _interp_at(series, f_eval, deg):
            if f_eval is None or not series: return math.nan
            pre = [(f, v) for (f, v) in sorted(series) if f < f_eval]
            if len(pre) < (deg + 1): return math.nan
            xs = np.array([f for f, _ in pre[-k:]], float)
            ys = np.array([v for _, v in pre[-k:]], float)
            m = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[m], ys[m]
            if xs.size < (deg + 1): return math.nan
            try:
                c = np.polyfit(xs, ys, deg=1 if fit_kind.lower().startswith("lin") else 2)
                return float(np.polyval(c, float(f_eval)))
            except:
                return math.nan

        methods = [
            ("area", lambda a, p, fM, fm, eM, eN: 2.0 * np.sqrt(a / np.pi)),
            ("feret_max", lambda a, p, fM, fm, eM, eN: fM),
            ("feret_min", lambda a, p, fM, fm, eM, eN: fm),
            ("ell_maj", lambda a, p, fM, fm, eM, eN: eM),
            ("ell_min", lambda a, p, fM, fm, eM, eN: eN),
            ("perim", lambda a, p, fM, fm, eM, eN: p / np.pi),
            ("mix", lambda a, p, fM, fm, eM, eN: 0.5 * (2.0 * np.sqrt(a / np.pi) + p / np.pi)),
        ]
        fit_deg = 1 if fit_kind.lower().startswith("lin") else 2

        rows_out = []
        n_tracks = len(frames_arr)
        for tid in range(n_tracks):
            frames0 = list(frames_arr[tid]) if isinstance(frames_arr[tid], (list, np.ndarray)) else []
            labels = list(bubclass_arr[tid]) if isinstance(bubclass_arr[tid], (list, np.ndarray)) else []
            if not frames0:
                base = {"bubble_id": tid, "attach_start_frame": None, "last_attached_frame": None,
                        "detach_frame": None, "note": "no_frames", "k": k, "fit_kind": fit_kind,
                        "conf_dep_last_attached_pct": np.nan, "conf_dep_first_detached_pct": np.nan,
                        "conf_dep_mean_pct": np.nan}
                for m, _ in methods:
                    base[f"D_{m}_px_discr"] = np.nan;
                    base[f"D_{m}_px_interp"] = np.nan
                    base[f"D_{m}_mm_discr"] = np.nan;
                    base[f"D_{m}_mm_interp"] = np.nan
                rows_out.append(base);
                continue

            attach_start, attach_end_i, last_attached, detach_frame, fr_s = _find_departure(frames0, labels)
            if attach_start is None:
                base = {"bubble_id": tid, "attach_start_frame": None, "last_attached_frame": None,
                        "detach_frame": None, "note": "no_attached_run", "k": k, "fit_kind": fit_kind,
                        "conf_dep_last_attached_pct": np.nan, "conf_dep_first_detached_pct": np.nan,
                        "conf_dep_mean_pct": np.nan}
                for m, _ in methods:
                    base[f"D_{m}_px_discr"] = np.nan;
                    base[f"D_{m}_px_interp"] = np.nan
                    base[f"D_{m}_mm_discr"] = np.nan;
                    base[f"D_{m}_mm_interp"] = np.nan
                rows_out.append(base);
                continue

            ser_area = _series_for_track(tid, "area_px")
            ser_perim = _series_for_track(tid, "perim_px")
            ser_fmax = _series_for_track(tid, "feret_max_px")
            ser_fmin = _series_for_track(tid, "feret_min_px")
            ser_eMaj = _series_for_track(tid, "ell_major_px")
            ser_eMin = _series_for_track(tid, "ell_minor_px")
            ser_score = _series_for_track(tid, "score")  # (frame0, score)

            def _series_method(name):
                if name == "area":
                    return ser_area
                elif name == "perim":
                    return ser_perim
                elif name == "feret_max":
                    return ser_fmax
                elif name == "feret_min":
                    return ser_fmin
                elif name == "ell_maj":
                    return ser_eMaj
                elif name == "ell_min":
                    return ser_eMin
                elif name == "mix":
                    a = dict(ser_area);
                    p = dict(ser_perim)
                    fs = sorted(set(a) & set(p))
                    return [(f, 0.5 * (2.0 * np.sqrt(a[f] / np.pi) + p[f] / np.pi)) for f in fs]
                else:
                    return []

            def _score_at(f0):
                if not ser_score: return np.nan
                d = dict(ser_score)
                if f0 in d: return float(d[f0])
                # ultimo score precedente
                prev = [d[f] for f in d.keys() if f <= f0]
                return float(prev[-1]) if prev else np.nan

            base = {
                "bubble_id": tid,
                "attach_start_frame": int(attach_start) if attach_start is not None else None,
                "last_attached_frame": int(last_attached) if last_attached is not None else None,
                "detach_frame": int(detach_frame) if detach_frame is not None else None,
                "note": "ok" if detach_frame is not None else "no_detach_found",
                "k": k, "fit_kind": fit_kind
            }

            # --- confidenza (media tra ultimo A e primo D) ---
            sA = _score_at(last_attached)
            sD = _score_at(detach_frame) if detach_frame is not None else np.nan
            if np.isfinite(sA) and np.isfinite(sD):
                sM = 0.5 * (sA + sD)
            elif np.isfinite(sA):
                sM = sA
            else:
                sM = np.nan
            base["conf_dep_last_attached_pct"] = (sA * 100.0) if np.isfinite(sA) else np.nan
            base["conf_dep_first_detached_pct"] = (sD * 100.0) if np.isfinite(sD) else np.nan
            base["conf_dep_mean_pct"] = (sM * 100.0) if np.isfinite(sM) else np.nan

            for (mname, _) in methods:
                if mname in {"area", "perim", "feret_max", "feret_min", "ell_maj", "ell_min"}:
                    comp = _series_method(mname)
                    d_series = []
                    for f, val in comp:
                        if mname == "area":
                            d_series.append((f, 2.0 * math.sqrt(val / math.pi)))
                        elif mname == "perim":
                            d_series.append((f, val / math.pi))
                        else:
                            d_series.append((f, float(val)))
                else:
                    d_series = _series_method("mix")

                d_px_discr = _discrete_at(d_series, last_attached)
                target_eval = detach_frame if detach_frame is not None else last_attached
                d_px_interp = _interp_at(d_series, target_eval, fit_deg)
                if not np.isfinite(d_px_interp): d_px_interp = d_px_discr

                base[f"D_{mname}_px_discr"] = d_px_discr
                base[f"D_{mname}_px_interp"] = d_px_interp
                base[f"D_{mname}_mm_discr"] = d_px_discr * mm_per_px if np.isfinite(d_px_discr) else np.nan
                base[f"D_{mname}_mm_interp"] = d_px_interp * mm_per_px if np.isfinite(d_px_interp) else np.nan

            rows_out.append(base)

        cols_head = ["bubble_id", "attach_start_frame", "last_attached_frame", "detach_frame", "note", "k", "fit_kind",
                     "conf_dep_last_attached_pct", "conf_dep_first_detached_pct", "conf_dep_mean_pct"]
        mcols = []
        for m, _ in methods:
            mcols += [f"D_{m}_px_discr", f"D_{m}_px_interp", f"D_{m}_mm_discr", f"D_{m}_mm_interp"]
        cols = cols_head + mcols

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols);
            w.writeheader()
            for r in rows_out:
                for c in cols:
                    if c not in r: r[c] = ""
                w.writerow(r)

        print(f"[ComputeDepartureDiameter] salvato: {out_csv}")
        return rows_out

def TrainSegmentationModel(datapath, ):
    register_coco_instances("my_dataset_train", {}, datapath, "")
    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = "./Models"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_MATLAB1.pth")  # path to the model we just trained
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # We have 1 classes.
    # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    def custom_mapper(dataset_dict, savename):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        mean = 0
        std_dev = 25
        gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        # noisy_image = cv2.add(image, gaussian_noise)

        transform_list = [
            # T.Resize((800,600)),
            T.RandomBrightness(0.8, 1.8),
            T.RandomContrast(0.6, 1.3),
            T.RandomSaturation(0.8, 1.4),
            # T.RandomRotation(angle=[90, 90]),
            # T.RandomNoise(mean=0.0, std=0.1),
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

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()  # Start the training process

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, savename)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)


def TrainCNNClassification(savename):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((480, 640)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Set data directory
    data_dir = './output'

    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Create dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}

    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=6, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=6, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Initialize the model
    model = CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize variables to keep track of best accuracy and corresponding model weights
    best_accuracy = 0.0
    best_model_weights = None

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Check if current phase is validation and if current accuracy is better than the best accuracy
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                # Save the model weights
                best_model_weights = model.state_dict()

    # Save the best model weights
    torch.save(best_model_weights, savename)
    print("Training complete!")