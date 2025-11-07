# import the librairies
import cv2, os
from torch.cuda import is_available
# Define the parameters for the model:
videopath = r"C:\Users\faraboli\Desktop\BubbleID\video\avi\T113_2_PB_250904_BC023_8_TIP_Chip_3_40V_2_1actv.avi"    # Define the path to the avi video
savefolder="My_output/Test6/"   # Define the folder you want the data to save in
extension="Test6"    # Define the extension you want all the saved data to have. This should be unique for each experiment
imagesfolder = savefolder + "/trimImages_" + extension   # Define the path to the folder of images
thres=0.5    # Define the threshold for what the model identifies as a bubble
modelweights = "..\\MODELS\\" + "3classes_tip_png" + "\\model_final.pth"
# modelweights=r"C:\Users\faraboli\Desktop\BubbleID\BubbleIDGit\ProjetBubbleID\training\Models_3classes_all\model_final.pth"     # Define the path to the saved model weights.
# device='cpu'   # Specify if running on "cpu" or "gpu"
device = "cuda" if is_available() else "cpu"
print(f"Used device : {device}")
import importlib
import BubbleID_My

# recharge le module pour prendre en compte les modifications
BubbleID_My = importlib.reload(BubbleID_My)

import BubbleID_My as BubbleID
print(BubbleID.__file__)

# Instantiating the class
test120=BubbleID.DataAnalysis(imagesfolder,videopath,savefolder,extension,modelweights,device)
test120.findMerge()
# 2) calcola dwell con fusione etichetta + contatto (bbox veloce)
ATTACHED, DETACHED, UNKNOWN = 2, 0, 1

dwell = test120.ComputeAttachDwellWithSurface(
    fps=4000,                  # se None prova a leggerlo dal video
    class_idx_attached=ATTACHED,
    class_idx_detached=DETACHED,
    class_idx_unknown=UNKNOWN,
    tolerate_unknown_gap=1,
    min_attached_run=2,
    out_csv=os.path.join(savefolder, f"dwell_{extension}.csv")  # opzionale; se None usa dwell_<EXT>.csv
)
