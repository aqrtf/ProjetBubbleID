#!/usr/bin/env python3
"""
side_by_side.py
Crée une vidéo 2x2 ou 1x2 selon le nombre de vidéos (2 à 4), avec les noms affichés.
Usage:
    python side_by_side.py video1.mp4 video2.mp4 [video3.mp4] [video4.mp4] output.mp4
"""

import sys
import cv2
from pathlib import Path
import numpy as np

def open_cap(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {path}")
    return cap

def get_props(cap):
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frames, fps, width, height

def add_label(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (5,5), (5+text_w, 5+text_h+baseline), bg_color, -1)
    cv2.putText(frame, text, (5, 5+text_h), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def create_black_frame(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def main(video_paths, out_path):
    if not (2 <= len(video_paths) <= 4):
        raise ValueError("Il faut fournir entre 2 et 4 vidéos.")

    # Ajouter des vidéos "noires" si moins de 4 vidéos
    while len(video_paths) < 4:
        video_paths.append(None)  # placeholder pour fond noir

    caps = [open_cap(p) if p is not None else None for p in video_paths]
    props = []
    for i, cap in enumerate(caps):
        if cap is not None:
            props.append(get_props(cap))
        else:
            props.append((0, 0, 640, 360))  # résolution par défaut pour le fond noir

    frames_list, fps_list, widths, heights = zip(*props)
    max_frames = max([f for f in frames_list if f > 0])
    fps = next((f for f in fps_list if f > 0), 25)

    # Redimensionner toutes les vidéos à la même hauteur (min des hauteurs non nulles)
    target_h = min([h for h in heights if h > 0])
    new_ws = [int(w * target_h / h) if h > 0 else int(w * target_h / 360) for w, h in zip(widths, heights)]

    # Déterminer la disposition
    if len([p for p in video_paths if p is not None]) == 2:
        out_width = sum(new_ws[:2])
        out_height = target_h
    else:  # 3 ou 4 vidéos => grille 2x2
        out_width = max(new_ws[0]+new_ws[1], new_ws[2]+new_ws[3])
        out_height = target_h*2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))

    filenames = [Path(p).name if p is not None else "No Video" for p in video_paths]

    for frame_idx in range(max_frames):
        frames = []
        for i, cap in enumerate(caps):
            if cap is not None:
                ret, f = cap.read()
                if not ret:
                    f = create_black_frame(new_ws[i], target_h)
            else:
                f = create_black_frame(new_ws[i], target_h)
            f = cv2.resize(f, (new_ws[i], target_h))
            f = add_label(f, filenames[i])
            frames.append(f)

        if len([p for p in video_paths if p is not None]) == 2:
            combined = cv2.hconcat(frames[:2])
        else:
            top = cv2.hconcat(frames[:2])
            bottom = cv2.hconcat(frames[2:4])
            combined = cv2.vconcat([top, bottom])

        writer.write(combined)
        if frame_idx % 100 == 0:
            print(f"Traitement: frame {frame_idx}/{max_frames}")

    for cap in caps:
        if cap is not None:
            cap.release()
    writer.release()
    print("Terminé — fichier de sortie créé :", out_path)

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: python side_by_side.py video1.mp4 video2.mp4 [video3.mp4] [video4.mp4] output.mp4")
        sys.exit(1)
    main(sys.argv[1:-1], sys.argv[-1])
