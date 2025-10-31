import cv2
import matplotlib.pyplot as plt

def afficher_frames_autour(video_path, frame_number):
    """
    Affiche 8 frames: 4 avant, la frame centrale, et 3 après en 2 lignes de 4
    """
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return
    
    # Calculer le nombre total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Vérifier que la frame demandée est valide
    if frame_number < 1 or frame_number > total_frames:
        print(f"Erreur: Frame {frame_number} hors limites (1-{total_frames})")
        cap.release()
        return
    
    # Calculer les frames à afficher (4 avant + centrale + 3 après = 8 frames)
    start_frame = max(1, frame_number - 4)
    end_frame = min(total_frames, frame_number + 3)
    
    frames_a_afficher = list(range(start_frame, end_frame + 1))
    
    # Configuration du subplot : 2 lignes × 4 colonnes
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    
    # Aplatir les axes pour faciliter l'itération
    axes_flat = axes.flatten()
    
    # Lire et afficher chaque frame
    for i, frame_idx in enumerate(frames_a_afficher):
        # Positionner la tête de lecture sur la frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)  # -1 car OpenCV commence à 0
        
        # Lire la frame
        ret, frame = cap.read()
        if not ret:
            print(f"Erreur: Impossible de lire la frame {frame_idx}")
            continue
        
        # Convertir BGR (OpenCV) en RGB (Matplotlib)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Afficher la frame
        axes_flat[i].imshow(frame_rgb)
        
        # Encadrer la frame - méthode directe avec set_linewidth
        if frame_idx == frame_number:
            # Cadre rouge épais pour la frame centrale
            for spine in axes_flat[i].spines.values():
                spine.set_color('red')
                spine.set_linewidth(1)
                spine.set_visible(True)
        else:
            # Cadre noir fin pour les autres frames
            for spine in axes_flat[i].spines.values():
                spine.set_color('black')
                spine.set_linewidth(.5)
                spine.set_visible(True)
        
        axes_flat[i].set_xticks([])
        axes_flat[i].set_yticks([])
        
        # Alternative : utiliser une boîte rectangulaire
        # axes_flat[i].add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=4, transform=axes_flat[i].transAxes))
    
    # Désactiver les axes inutilisés (au cas où il y aurait moins de 8 frames)
    for i in range(len(frames_a_afficher), len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Ajuster l'espacement entre les images
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    
    # Libérer la vidéo
    cap.release()


# Utilisation
for i in [6, 9, 10, 21, 24, 49, 85, 88]:
    afficher_frames_autour(r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\My_output\SaveData3\tracked_T113_2_60V_2.avi", i)