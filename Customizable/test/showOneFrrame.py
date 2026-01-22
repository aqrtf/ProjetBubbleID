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
    
    
    frames_a_afficher = frame_number
    

    

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # -1 car OpenCV commence à 0
    
    # Lire la frame
    ret, frame = cap.read()
    if not ret:
        print(f"Erreur: Impossible de lire la frame {frame_number}")
        
    
    # Convertir BGR (OpenCV) en RGB (Matplotlib)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Définir la zone à zoomer : (x1, y1) = (400, 930), (x2, y2) = (500, 830)
    x1, y1 = 400, 930
    x2, y2 = 500, 830
    # Attention : en numpy, l'ordre est [y1:y2, x1:x2]
    frame_zoom = frame_rgb[y2:y1, x1:x2] if y1 > y2 else frame_rgb[y1:y2, x1:x2]

    # Afficher la zone zoomée
    plt.imshow(frame_zoom)
    plt.axis('off')
    # plt.axis('off')
    # plt.savefig(r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet\rapport\Figure\beforeIdealMerge.png", bbox_inches='tight', pad_inches=0)
    # plt.close()
    plt.show()

    # Libérer la vidéo
    cap.release()



afficher_frames_autour(r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet\ProjetBubbleID\Inputs\T87_out\tracked_T87_50V2.avi", 253)
# Utilisation
# for i in [6, 9, 10, 21, 24, 49, 85, 88]:
#     afficher_frames_autour(r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\My_output\SaveData3\tracked_T113_2_60V_2.avi", i)