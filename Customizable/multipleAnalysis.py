import os, json
import BubbleID_My as BubbleID
from torch.cuda import is_available
folderPath = os.path.dirname(os.path.abspath(__file__))

chipName = [#"T87",
            "T88",
            "T89"]
prefixName_all = [#"T87_2_PB_250806_BC015_7_Chip_7_",
              "T88_PB_250828_BC015_7_Chip_6_",
              "T89_2_PB_250807_BC015_7_Chip_1_"]

videoFolderRoot = r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\Inputs"
tensions = [50, 60, 75, 85, 100]
model_weight = "model_cav_jpeg"
nFrameExtract = 550
score_thres=0.7
mm_per_px = 0.0227
for idx_chip, chip in enumerate(chipName):
    videoFolder = os.path.join(videoFolderRoot, chip)
    savefolder = os.path.join(videoFolderRoot, chip+"_out")
    prefixName = prefixName_all[idx_chip]
    for t in tensions:
        for tnum in (1, 2):
            videoName = prefixName + str(t) + "V_" + str(tnum) + "_4actv.avi"
            extension = chip + "_" + str(t) + "V" + str(tnum)

            # Création des chemins
            imagesfolder = os.path.join(savefolder, f"trimImages_{extension}")
            model_path = os.path.join(os.path.dirname(folderPath), "MODELS", model_weight + ".pth")
            device = "cuda" if is_available() else "cpu"
            
            # Créer le dossier de sauvegarde
            os.makedirs(savefolder, exist_ok=True)
            
            # Instancier la classe
            test120 = BubbleID.DataAnalysis(
                imagesfolder,
                os.path.join(videoFolder, videoName),
                savefolder,
                extension,
                model_path,
                device
            )

            test120.trimVideo(N_frames_extr=nFrameExtract)

            # mm = test120.set_scale_by_two_points(frame_idx=0, physical_mm=20.0, save=True)
            test120.mm_per_px = mm_per_px
            os.makedirs(savefolder, exist_ok=True)
            out_json = os.path.join(savefolder, f"scale_{extension}.json")
            payload = {
                    "mm_per_px": mm_per_px
                }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            test120.GenerateData(
                thres=0.5, 
                save_rich=True, 
                save_masks=False, 
                save_contours=True, 
                iou_thresh_tid=0.5
            )

            test120.make_tracked_video(n_frames=nFrameExtract, fps=5, score_thres = score_thres)

            import BubbleID_dependencies.parentBubble as parentBubble
            parentBubble.findMerge(savefolder, extension)

            import BubbleID_dependencies.evolution_tid as evolution_tid
            evolution_tid.evolution_tid(savefolder, extension, score_thres)

            import departureDiameter 
            departureDiameter.ComputeDepartureDiameter(savefolder, extension)

            from BubbleID_dependencies.bubbleProperties import mainProperties
            mainProperties(savefolder, extension, chipName=chip, tension=t)
                
                
            
                
                

            
