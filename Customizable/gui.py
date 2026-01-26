import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, importlib
import os

from csteDef import *

folderPath = os.path.dirname(os.path.abspath(__file__))

class BubbleIDGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BubbleID Analysis Tool")
        self.root.geometry("750x700")
        
        # Variables pour stocker les paramètres
        self.save_folder_var = tk.StringVar()
        self.extension_var = tk.StringVar()
        self.video_path_var = tk.StringVar()
        self.model_weights_var = tk.StringVar()
        self.thres_var = tk.DoubleVar(value=0.5)
        self.nFrameExtract_var = tk.IntVar(value=500)
        self.chipWidth_var = tk.DoubleVar(value=20.0)
        self.iouThresh_var = tk.DoubleVar(value=0.5)
        self.fps_var = tk.IntVar(value=4000)
        self.scoreThres_var = tk.DoubleVar(value=0.7)
        self.overlapTresh_var = tk.DoubleVar(value=0.1)
        self.minOverlapSame_var = tk.DoubleVar(value=0.7)
        self.postFusionFrames_var = tk.IntVar(value=2)
        self.nFramePreviousDisappear_var = tk.IntVar(value=3)
        self.nFramePostDisappear_var = tk.IntVar(value=2)
        self.dilateIter_var = tk.IntVar(value=1)
        
        
        
        # Liste des paramètres (clé, variable)
        self.parameters = [
            ("Threshold", self.thres_var),
            ("Max Frames", self.nFrameExtract_var),
            ("Chip Width (mm)", self.chipWidth_var),
            ("IOU Threshold", self.iouThresh_var),
            ("FPS", self.fps_var),
            ("Score Threshold", self.scoreThres_var),
            ("Overlap Threshold", self.overlapTresh_var),
            ("Min Overlap Same", self.minOverlapSame_var),
            ("Post Fusion Frames", self.postFusionFrames_var),
            ("N Frames Previous Disappear", self.nFramePreviousDisappear_var),
            ("N Frames Post Disappear", self.nFramePostDisappear_var),
            ("Dilate Iterations", self.dilateIter_var)
        ]
        
        # Instance de DataAnalysis
        self.test120 = None
        
        # Liste des modèles disponibles
        self.available_models = [
            "model_all_jpeg", 
            "model_cav_jpeg",
            "model_tip_jpeg",
            "model_all_png",
            "model_cav_png",
            "model_tip_png",
            "model_old"
        ]
        
        self.setup_ui()
    def setup_ui(self):
        style = ttk.Style()
        style.configure("TFrame", padding=10)
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # --- FILE BANDEAU ---
        file_frame = ttk.LabelFrame(main_frame, text="File", padding=10)
        file_frame.pack(fill=tk.X, pady=1)

        ttk.Label(file_frame, text="Save Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.save_folder_var, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_save_folder).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Extension:").grid(row=0, column=3, sticky="w")
        ttk.Entry(file_frame, textvariable=self.extension_var, width=20).grid(row=0, column=4, padx=5)
        
        ttk.Label(file_frame, text="Video path (for tracking):").grid(row=1, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.video_path_var, width=30).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_video_path).grid(row=1, column=2, padx=5)
        
        ttk.Label(file_frame, text="Model:").grid(row=1, column=3, sticky="w")
        model_combo = ttk.Combobox(file_frame, textvariable=self.model_weights_var, values=self.available_models, width=17)
        model_combo.grid(row=1, column=4, columnspan=2, padx=5, pady=5, sticky="ew")
        model_combo.set("model_all_jpeg")

        # --- PARAMETRES ---
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=1)

        # Génération automatique des champs
        for i, (label, var) in enumerate(self.parameters):
            ncol = 3
            irow = i//ncol
            icol = i%ncol
            ttk.Label(params_frame, text=label).grid(row=irow, column=icol*2, sticky="w", pady=5)
            ttk.Entry(params_frame, textvariable=var, width=10).grid(row=irow, column=2*icol+1, padx=5, pady=5)

        # --- 3 COLONNES ---
        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=1)

        # Tracking
        tracking_frame = ttk.LabelFrame(columns_frame, text="Tracking", padding=10)
        tracking_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tracking_buttons = [
            ("Initialize Data", self.initialize_data_analysis),
            ("Trim Video", self.trim_video),
            ("Set Scale", self.set_scale),
            ("Generate Data", self.generate_data),
            ("Tracked Video", self.tracked_video),
            ("Run All Tracking", self.run_all_tracking)
        ]
        for text, cmd in tracking_buttons:
            ttk.Button(tracking_frame, text=text, command=cmd).pack(fill=tk.X, pady=5)

        # Postprocessing
        postprocess_frame = ttk.LabelFrame(columns_frame, text="Post Process", padding=10)
        postprocess_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        postprocess_buttons = [
            ("Find Merge", self.find_merge),
            ("Compute Dwell", self.compute_dwell),
            ("Departure Diameter", self.departure_diameter),
            ("Calc Bubble Properties", self.calc_bubble_properties),
        ]
        for text, cmd in postprocess_buttons:
            ttk.Button(postprocess_frame, text=text, command=cmd).pack(fill=tk.X, pady=5)

        # Logs
        log_frame = ttk.LabelFrame(columns_frame, text="Logs", padding=10)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.log_text = tk.Text(log_frame, height=20, width=40)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
    """    
    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.configure("TFrame", padding=10)
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        
        # SECTION FILE
        file_frame = ttk.LabelFrame(main_frame, text="File", padding=15)
        file_frame.pack(fill=tk.X, pady=10)
        
        # Save Folder
        ttk.Label(file_frame, text="Save Folder:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.save_folder_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_save_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Extension
        ttk.Label(file_frame, text="Extension:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.extension_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        
        # SECTION TRACKING ET POST PROCESS (côte à côte)
        processing_frame = ttk.Frame(main_frame)
        processing_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Colonne TRACKING
        tracking_frame = ttk.LabelFrame(processing_frame, text="Tracking", padding=15)
        tracking_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video Path
        ttk.Label(tracking_frame, text="Video Path:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(tracking_frame, textvariable=self.video_path_var, width=40).grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        ttk.Button(tracking_frame, text="Browse", command=self.browse_video_path).grid(row=0, column=3, padx=5, pady=5)
        
        # Model Weights
        ttk.Label(tracking_frame, text="Model Weights:").grid(row=1, column=0, sticky="w", pady=5)
        model_combo = ttk.Combobox(tracking_frame, textvariable=self.model_weights_var, values=self.available_models, width=37)
        model_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        model_combo.set("model_all_jpeg")
        
        # Threshold
        ttk.Label(tracking_frame, text="Threshold:").grid(row=2, column=0, sticky="w", pady=5)
        threshold_frame = ttk.Frame(tracking_frame)
        threshold_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        ttk.Scale(threshold_frame, from_=0.1, to=1.0, variable=self.thres_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT)
        ttk.Label(threshold_frame, textvariable=self.thres_var, width=4).pack(side=tk.LEFT, padx=5)

        # Extension
        ttk.Label(file_frame, text="Max extract frame:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(file_frame, textvariable=self.nFrameExtract_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Boutons Tracking
        tracking_buttons = [
            ("Initialize Data", self.initialize_data_analysis),
            ("Trim Video", self.trim_video),
            ("Set Scale", self.set_scale),
            ("Generate Data", self.generate_data),
            ("Tracked Video", self.tracked_video),
            ("Run All Tracking", self.run_all_tracking)
        ]
        
        for i, (text, command) in enumerate(tracking_buttons):
            ttk.Button(tracking_frame, text=text, command=command, width=20).grid(
                row=i+3, column=0, columnspan=4, padx=10, pady=5, sticky="ew"
            )
        
        # Colonne POST PROCESS
        postprocess_frame = ttk.LabelFrame(processing_frame, text="Post Process", padding=15)
        postprocess_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Boutons Post Process
        postprocess_buttons = [
            ("Find Merge", self.find_merge),
            ("Compute Dwell", self.compute_dwell),
            ("Departure Diameter", self.departure_diameter),
            ("Calc Bubble Properties", self.calc_bubble_properties),
        ]
        
        for i, (text, command) in enumerate(postprocess_buttons):
            ttk.Button(postprocess_frame, text=text, command=command, width=20).grid(
                row=i, column=0, padx=10, pady=15, sticky="ew"
            )
        
        # SECTION LOGS
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=12, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
    """
    
    def log_message(self, message):
        """Ajoute un message aux logs"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_status(self, status):
        """Met à jour la barre de statut"""
        self.status_var.set(status)
        self.root.update()
    
    def browse_save_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.save_folder_var.set(folder)
    
    def browse_video_path(self):
        filetypes = [("Video files", "*.avi *.mp4 *.mov"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.video_path_var.set(filepath)
    
    def validate_parameters(self):
        """Valide que tous les paramètres requis sont remplis"""
        if not self.save_folder_var.get():
            messagebox.showerror("Error", "Please select a save folder")
            return False
        if not self.extension_var.get():
            messagebox.showerror("Error", "Please enter an extension")
            return False
        return True
    
    def validate_tracking_parameters(self):
        """Valide les paramètres spécifiques au tracking"""
        if not self.validate_parameters():
            return False
        if not self.video_path_var.get():
            messagebox.showerror("Error", "Please select a video file")
            return False
        if not self.model_weights_var.get():
            messagebox.showerror("Error", "Please select a model")
            return False
        return True
    
    def check_initialized(self):
        """Vérifie si DataAnalysis est initialisé"""
        if self.test120 is None:
            messagebox.showerror("Error", "Please initialize DataAnalysis first")
            return False
        return True
    
    def initialize_data_analysis(self):
        """Initialise l'instance DataAnalysis"""
        if not self.validate_tracking_parameters():
            return
        
        self.update_status("Initializing DataAnalysis...")
        try:
            import BubbleID_My as BubbleID
            from torch.cuda import is_available
            # Création des chemins
            imagesfolder = os.path.join(self.save_folder_var.get(), f"trimImages_{self.extension_var.get()}")
            model_path = os.path.join(os.path.dirname(folderPath), "MODELS", self.model_weights_var.get() + ".pth")
            device = "cuda" if is_available() else "cpu"
            
            # Créer le dossier de sauvegarde
            os.makedirs(self.save_folder_var.get(), exist_ok=True)
            
            # Instancier la classe
            self.test120 = BubbleID.DataAnalysis(
                imagesfolder,
                self.video_path_var.get(),
                self.save_folder_var.get(),
                self.extension_var.get(),
                model_path,
                device
            )
            
            self.log_message("DataAnalysis initialized successfully")
            self.log_message(f"Used device: {device}")
            self.update_status("Ready - DataAnalysis initialized")
            
        except Exception as e:
            self.log_message(f"Error initializing DataAnalysis: {str(e)}")
            self.update_status("Error")
    
    def trim_video(self):
        if not self.check_initialized():
            return
        
        self.update_status("Trimming video...")
        try:
            self.test120.trimVideo(N_frames_extr=self.nFrameExtract_var.get())
            self.log_message("Video trimmed successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error trimming video: {str(e)}")
            self.update_status("Error")
    
    def set_scale(self):
        if not self.check_initialized():
            return
        
        self.update_status("Setting scale...")
        try:
            mm = self.test120.set_scale_by_two_points(frame_idx=0, physical_mm=20.0, save=True)
            self.log_message(f"Scale set successfully: {mm} mm/px")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error setting scale: {str(e)}")
            self.update_status("Error")
    
    def generate_data(self):
        if not self.check_initialized():
            return
        
        self.update_status("Generating data...")
        try:
            self.test120.GenerateData(
                self.thres_var.get(), 
                save_rich=True, 
                save_masks=False, 
                save_contours=True, 
                iou_thresh_tid=0.5
            )
            self.log_message("Data generated successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error generating data: {str(e)}")
            self.update_status("Error")
    
    def tracked_video(self):
        if not self.check_initialized():
            return
        
        self.update_status("Creating tracked video...")
        try:
            self.test120.make_tracked_video(n_frames=self.nFrameExtract_var.get(), fps=5, score_thres=self.scoreThres_var.get())
            self.log_message("Tracked video created successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error creating tracked video: {str(e)}")
            self.update_status("Error")
    
    def run_all_tracking(self):
        """Exécute toutes les étapes du tracking à la suite"""
        if not self.validate_tracking_parameters():
            return
        
        steps = [
            (self.initialize_data_analysis, "Initialization"),
            (self.trim_video, "Video trimming"),
            (self.set_scale, "Scale setting"),
            (self.generate_data, "Data generation"),
            (self.tracked_video, "Tracked video creation")
        ]
        
        for step_func, step_name in steps:
            self.log_message(f"--- Starting {step_name} ---")
            step_func()
            # Vérifier si l'étape a réussi
            if "Error" in self.status_var.get():
                self.log_message(f"--- {step_name} failed ---")
                return
            self.log_message(f"--- {step_name} completed ---\n")
    
    def find_merge(self):   
        if not self.validate_parameters():
            return   
        self.update_status("Finding merges...")
        try:
            import parentBubble as parentBubble
            # parentBubble = importlib.reload(parentBubble)
            parentBubble.findMerge(self.save_folder_var.get(), self.extension_var.get(),
                                   score_thres=self.scoreThres_var.get(),
                                   OVERLAP_THRESH=self.overlapTresh_var.get(),
                                   MIN_OVERLAP_SAME=self.minOverlapSame_var.get(),
                                   POST_FUSION_FRAMES=self.postFusionFrames_var.get(),
                                   N_FRAMES_PREVIOUS_DISAPPEAR=self.nFramePreviousDisappear_var.get(),
                                   N_FRAMES_POST_DISAPPEAR=self.nFramePostDisappear_var.get(),
                                   DILATE_ITERS=self.dilateIter_var.get())
            self.log_message("Merge finding completed")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error finding merges: {str(e)}")
            self.update_status("Error")
        self.update_status("Finding evolution of track ID...")
        try:
            import evolution_tid as evolution_tid
            # evolution_tid = importlib.reload(evolution_tid)
            evolution_tid.evolution_tid(self.save_folder_var.get(), self.extension_var.get(), score_thres=self.scoreThres_var.get())
            self.log_message("Evolution completed")
            self.update_status("Ready")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.update_status("Error")
    
    def compute_dwell(self):  
        if not self.validate_parameters():
            return     
        self.update_status("Computing dwell time...")
        try:
            import computedwell as computedwell
            # computedwell = importlib.reload(computedwell)
            computedwell.analyze_dwell_time(self.save_folder_var.get(), self.extension_var.get(),
                                            score_thres=self.scoreThres_var.get(),
                                            n_frames_post_disappear=self.nFramePostDisappear_var.get(),
                                            fps=self.fps_var.get())
            
            self.log_message("Dwell time computed successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error computing dwell time: {str(e)}")
            self.update_status("Error")
    
    def departure_diameter(self):  
        if not self.validate_parameters():
            return
        self.update_status("Calculating departure diameter...")
        try:
            # À adapter selon votre méthode réelle
            import departureDiameter 
            # departureDiameter = importlib.reload(departureDiameter)
            departureDiameter.ComputeDepartureDiameter(self.save_folder_var.get(), self.extension_var.get())
            self.log_message("Departure diameter calculation completed")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error calculating departure diameter: {str(e)}")
            self.update_status("Error")
    
    def calc_bubble_properties(self):  
        if not self.validate_parameters():
            return
        self.update_status("Calculating bubble properties...")
        try:
            from bubbleProperties import mainProperties
            mainProperties(self.save_folder_var.get(), self.extension_var.get())
            self.log_message("Bubble properties calculation completed")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error calculating bubble properties: {str(e)}")
            self.update_status("Error")

def main():
    root = tk.Tk()
    app = BubbleIDGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()