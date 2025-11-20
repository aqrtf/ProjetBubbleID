import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
from torch.cuda import is_available
import importlib
import BubbleID_My

folderPath = os.path.dirname(os.path.abspath(__file__))

class BubbleIDGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BubbleID Analysis Tool")
        self.root.geometry("600x750")
        
        # Variables pour stocker les paramètres
        self.save_folder_var = tk.StringVar()
        self.extension_var = tk.StringVar()
        self.thres_var = tk.DoubleVar(value=0.5)
        self.video_path_var = tk.StringVar()
        self.model_weights_var = tk.StringVar()
        
        # Instance de DataAnalysis
        self.test120 = None
        
        # Liste des modèles disponibles
        self.available_models = [
            "3classes_tip_jpeg",
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
        # Style
        style = ttk.Style()
        style.configure("TFrame", padding=10)
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Titre
        title_label = ttk.Label(main_frame, text="BubbleID Analysis", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Frame pour les paramètres
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=15)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Save Folder
        ttk.Label(params_frame, text="Save Folder:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.save_folder_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(params_frame, text="Browse", command=self.browse_save_folder).grid(row=0, column=2, padx=5, pady=5)
        
        # Extension
        ttk.Label(params_frame, text="Extension:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.extension_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        
        # Threshold
        ttk.Label(params_frame, text="Threshold:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.thres_var, orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(params_frame, textvariable=self.thres_var).grid(row=2, column=2, padx=5, pady=5)
        
        # Video Path
        ttk.Label(params_frame, text="Video Path:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.video_path_var, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(params_frame, text="Browse", command=self.browse_video_path).grid(row=3, column=2, padx=5, pady=5)
        
        # Model Weights
        ttk.Label(params_frame, text="Model Weights:").grid(row=4, column=0, sticky="w", pady=5)
        model_combo = ttk.Combobox(params_frame, textvariable=self.model_weights_var, values=self.available_models, width=47)
        model_combo.grid(row=4, column=1, padx=5, pady=5)
        model_combo.set("3classes_tip_jpeg")
        
        # Bouton Initialize
        init_frame = ttk.Frame(main_frame)
        init_frame.pack(fill=tk.X, pady=10)
        ttk.Button(init_frame, text="Initialize DataAnalysis", command=self.initialize_data_analysis, 
                  style="Accent.TButton").pack(pady=5)
        
        # Frame pour les boutons d'actions
        actions_frame = ttk.LabelFrame(main_frame, text="Actions", padding=15)
        actions_frame.pack(fill=tk.X, pady=10)
        
        # Boutons d'actions
        buttons = [
            ("Reload BubbleID", self.reload_bubbleid),
            ("Trim Video", self.trim_video),
            ("Set Scale", self.set_scale),
            ("Generate Data", self.generate_data),
            ("Show Bubble Contour", self.show_bubble_contour),
            ("Tracked Video", self.tracked_video),
            ("Find Merge", self.find_merge),
            ("Compute Dwell", self.compute_dwell)
        ]
        
        for i, (text, command) in enumerate(buttons):
            ttk.Button(actions_frame, text=text, command=command, width=20).grid(
                row=i//2, column=i%2, padx=10, pady=5, sticky="ew"
            )
        
        # Frame pour les logs
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
    
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
        if not self.validate_parameters():
            return
        
        self.update_status("Initializing DataAnalysis...")
        try:
            # Recharger le module
            importlib.reload(BubbleID_My)
            import BubbleID_My as BubbleID
            
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
    
    def reload_bubbleid(self):
        """Recharge le module BubbleID"""
        self.update_status("Reloading BubbleID module...")
        try:
            importlib.reload(BubbleID_My)
            self.log_message("BubbleID module reloaded successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error reloading BubbleID: {str(e)}")
            self.update_status("Error")
    
    def trim_video(self):
        if not self.check_initialized():
            return
        
        self.update_status("Trimming video...")
        try:
            self.test120.trimVideo(N_frames_extr=100)
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
    
    def show_bubble_contour(self):
        if not self.check_initialized():
            return
        
        self.update_status("Showing bubble contours...")
        try:
            self.test120.show_bubble_contours(list(range(0, 50)), thres=0.7)
            self.log_message("Bubble contours displayed")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error showing bubble contours: {str(e)}")
            self.update_status("Error")
    
    def tracked_video(self):
        if not self.check_initialized():
            return
        
        self.update_status("Creating tracked video...")
        try:
            self.test120.make_tracked_video(n_frames=100, fps=5, score_thres=0.7)
            self.log_message("Tracked video created successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error creating tracked video: {str(e)}")
            self.update_status("Error")
    
    def find_merge(self):
        if not self.check_initialized():
            return
        
        self.update_status("Finding merges...")
        try:
            self.test120.findMerge()
            self.log_message("Merge finding completed")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error finding merges: {str(e)}")
            self.update_status("Error")
    
    def compute_dwell(self):
        if not self.check_initialized():
            return
        
        self.update_status("Computing dwell time...")
        try:
            ATTACHED, DETACHED, UNKNOWN = 2, 0, 1
            dwell = self.test120.ComputeAttachDwellWithSurface(
                fps=4000,
                class_idx_attached=ATTACHED,
                class_idx_detached=DETACHED,
                class_idx_unknown=UNKNOWN,
                tolerate_unknown_gap=1,
                min_attached_run=2,
                out_csv=os.path.join(self.save_folder_var.get(), f"dwell_{self.extension_var.get()}.csv")
            )
            
            self.log_message("Dwell time computed successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error computing dwell time: {str(e)}")
            self.update_status("Error")

def main():
    root = tk.Tk()
    app = BubbleIDGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()