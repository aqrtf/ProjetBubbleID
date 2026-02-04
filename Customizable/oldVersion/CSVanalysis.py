import pandas as pd
import matplotlib.pyplot as plt
from csteDef import *
import numpy as np
from scipy.ndimage import median_filter

class BubbleKinematics:
    def __init__(self, track_id, frame, time, cx, cy, state):
        self.track_id = track_id
        self.state = state
        self.startTime = time[0]
        self.endTime = time[-1]
        self.startFrame = frame[0]
        self.endFrame = frame[-1]
        
        self.time = time
        self.frame = frame
        # centroid position
        self.cx = cx
        self.cy = cy
        
    def calcVelocities(self):
        self.vx = np.diff(self.cx) / np.diff(self.time)
        self.vy = np.diff(self.cy) / np.diff(self.time)
        self.vAbs = np.sqrt(self.vx**2 + self.vy**2)
        self.timeVel = self.time[1:]
        
        self.stateVel = self.state[1:]
        self.vxDetached = self.vx[self.stateVel == DETACHED]
        self.vyDetached = self.vy[self.stateVel == DETACHED]
        self.vAbsDetached = self.vAbs[self.stateVel == DETACHED]
        self.timeVelDetached = self.timeVel[self.stateVel == DETACHED]
        
    def plotVelocity(self):
         # colore deterministico da track_id (BGR per OpenCV)
        def _color_for_id(tid: int):
            if tid is None or tid < 0:
                return (160, 160, 160)  # grigio per "senza ID"
            r = (37 * (tid + 1)) % 255
            g = (17 * (tid + 1)) % 255
            b = (29 * (tid + 1)) % 255
            return (r/255, g/255, b/255)
        
        if not hasattr(self, 'vx_smooth'): 
            self.smoothVelocities() # calcule les vitesses si ce n'est pas fait
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.tight_layout(pad=5.0)

        # Plot 1 
        axs[0, 0].plot(self.timeVelDetached, self.vxDetached, 
                       linestyle = '-', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}, detached")
        axs[0, 0].plot(self.timeVel[self.stateVel == UNKNOWN], self.vx[self.stateVel == UNKNOWN], 
                       linestyle = '--', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}, unknown")
        axs[0, 0].plot(self.timeVel[self.stateVel == ATTACHED], self.vx[self.stateVel == ATTACHED], 
                       linestyle = ':', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}, attached")
        axs[0, 0].plot(self.timeVelDetached, self.vxDet_smooth,
                       linestyle = '-.', color =  'k' , label = f"Move median smooth, window size = {self.window_size}")
        
        axs[0, 0].set_title("x velocity")
        axs[0, 0].set_ylabel("V_x [m/s]")
        axs[0, 0].set_xlabel("t [ms]")

        # Plot 2
        axs[0, 1].plot(self.timeVelDetached, self.vAbsDetached, 
                       linestyle = '-', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}")
        axs[0, 1].plot(self.timeVel[self.stateVel == UNKNOWN], self.vAbs[self.stateVel == UNKNOWN], 
                       linestyle = '--', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}")
        axs[0, 1].plot(self.timeVel[self.stateVel == ATTACHED], self.vAbs[self.stateVel == ATTACHED], 
                       linestyle = ':', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}")
        axs[0, 1].plot(self.timeVelDetached, self.vAbsDet_smooth,
                       linestyle = '-.', color =  'k' , label = f"Move median smooth, window size = {self.window_size}")
        
        axs[0, 1].set_title("absolute velocity")
        axs[0, 1].set_ylabel("V_abs [m/s]")
        axs[0, 1].set_xlabel("t [ms]")

        # Plot 3
        axs[1, 0].plot(self.timeVelDetached, self.vyDetached,
                       linestyle = '-', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}, detached")
        axs[1, 0].plot(self.timeVel[self.stateVel == UNKNOWN], self.vy[self.stateVel == UNKNOWN], 
                       linestyle = '--', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}")
        axs[1, 0].plot(self.timeVel[self.stateVel == ATTACHED], self.vy[self.stateVel == ATTACHED], 
                       linestyle = ':', color =  _color_for_id(self.track_id) , label = f"Index: {self.track_id}")
        axs[1, 0].plot(self.timeVelDetached, self.vyDet_smooth,
                       linestyle = '-.', color =  'k' , label = f"Move median smooth, window size = {self.window_size}")

        axs[1, 0].set_title("y velocity")
        axs[1, 0].set_ylabel("V_y [m/s]")
        axs[1, 0].set_xlabel("t [ms]")

        # Case vide (pour la légende)
        axs[1, 1].axis("off")  # on désactive les axes
        axs[1, 1].legend(handles=[
            axs[0, 0].lines[0],
            axs[0, 0].lines[1],
            axs[0, 0].lines[2],
            axs[0, 0].lines[3]
            ],
            loc="center"
        )
    
    def smoothVelocities(self, window_size=5):
        """
        Lisse les vitesses vx, vy et vAbs avec une médiane mobile.

        Args:
            window_size (int): taille de la fenêtre de la médiane (doit être impair)
        """
        if not hasattr(self, 'vAbs'):
            self.calcVelocities()  # calcule les vitesses si ce n'est pas fait

        # médiane glissante
        self.window_size = window_size
        self.vxDet_smooth = median_filter(self.vxDetached, size=window_size)
        self.vyDet_smooth = median_filter(self.vyDetached, size=window_size)
        self.vAbsDet_smooth = np.sqrt(self.vxDet_smooth**2 + self.vyDet_smooth**2)
        
       
        


class CSVAnalyzer:
    def __init__(self, file_rich, file_time):
        self.df_rich = pd.read_csv(file_rich)
        self.df_time = pd.read_csv(file_time)
        # creation d'un dictionnaire pour la conversion frame2time
        self.frame2timeDictionnary = dict(zip(self.df_time["Frame"], self.df_time["Time_ms"]))
        # liste pour stocker les BubbleKinematics
        self.bubbles = []
    
    def info(self):
        print(self.df_rich.info())
    
    def extractPosition(self):
        """ Extract the positiion of each bubbles from the rich file
        L'origine est en haut à gauche"""
        
        for idx, sub_df in self.df_rich.groupby("track_id"):
            frames = sub_df["frame"]-1 # dans rich les frames commencent a 1
            state = sub_df["class_id"].to_numpy()
            time = self.frame2time(frames)
            cx = sub_df["cx_mm"].to_numpy()
            cy = -sub_df["cy_mm"].to_numpy() # minus sign to have positive velocities
            bubble = BubbleKinematics(idx, frames.to_numpy(), time, cx, cy, state)
            
            self.bubbles.append(bubble)
            
            
    def frame2time(self, frameArray):
        """Convert a the frame number in a time with the extract info of the video. 
            Warning: in the conversion the frame start at 0

        Args:
            frameArray (pd.Series): an serie with the number of frame

        Returns:
            np.array: an array with the corresponding time
        """
        timeArray = frameArray.map(self.frame2timeDictionnary).to_numpy()
        return timeArray
            
        
    









analyzer = CSVAnalyzer(r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet2\BubbleID\My_output\SaveData\rich_Test1.csv", r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet2\BubbleID\Time_Test1.csv")
analyzer.extractPosition()

# accéder à la première bulle
b0 = analyzer.bubbles[0]

print(b0.track_id, b0.startTime, b0.endTime)

# parcourir toutes les bulles
for b in analyzer.bubbles[0:4]:
    b.calcVelocities()
    b.plotVelocity()
plt.show()





