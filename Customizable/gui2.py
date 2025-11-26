import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
from torch.cuda import is_available
import importlib
import BubbleID_My

class BubbleIDGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BubbleID Analysis Tool")
        self.root.geometry("750x850")
        
        # Instance de DataAnalysis
        self.test120 = None
        
        # Dictionnaire des paramètres
        self.parameters = {
            # (nom_affiché, variable_type, valeur_défaut, section, tooltip)
            "save_folder": ("Save Folder", "string", "", "file", "Dossier de sauvegarde des résultats"),
            "extension": ("Extension", "string", "", "file", "Extension pour les fichiers de sortie"),
            "video_path": ("Video Path", "string", "", "tracking", "Chemin vers la vidéo à analyser"),
            "model_weights": ("Models", "string", "3classes_tip_jpeg", "tracking", "Modèle à utiliser pour la détection"),
            "threshold": ("Threshold", "float", 0.5, "tracking", "Seuil de confiance pour la détection"),
            "n_frames_extr": ("N Frames Extract", "int", 100, "tracking", "Nombre d'images à extraire"),
            "physical_mm": ("Physical MM", "float", 20.0, "tracking", "Valeur physique pour l'étalonnage"),
            "iou_thresh_tid": ("IOU Threshold", "float", 0.5, "tracking", "Seuil IOU pour le tracking"),
            "fps": ("FPS", "int", 4000, "postprocess", "Images par seconde de la vidéo"),
            "tolerate_unknown_gap": ("Tolerate Gap", "int", 1, "postprocess", "Tolérance des gaps inconnus"),
            "min_attached_run": ("Min Attached", "int", 2, "postprocess", "Durée minimale attachée"),
        }
        
        # Liste des modèles disponibles
        self.available_models = [
            "3classes_tip_jpeg",
            "3classes_all_data", 
            "2classes_basic"
        ]
        
        # Créer les variables Tkinter dynamiquement
        self.vars = {}
        for param_name, (display_name, var_type, default, section, tooltip) in self.parameters.items():
            if var_type == "string":
                self.vars[param_name] = tk.StringVar(value=default)
            elif var_type == "int":
                self.vars[param_name] = tk.IntVar(value=default)
            elif var_type == "float":
                self.vars[param_name] = tk.DoubleVar(value=default)
        
        self.setup_ui()
    
    def create_parameter_row(self, parent, param_name, display_name, var_type, tooltip):
        """Crée une ligne de paramètre dans l'interface"""
        row_frame = ttk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(row_frame, text=display_name + ":", width=15, anchor="e").pack(side=tk.LEFT, padx=(0, 5))
        
        if param_name == "model_weights":
            combobox = ttk.Combobox(row_frame, textvariable=self.vars[param_name], 
                                   values=self.available_models, width=20, state="readonly")
            combobox.pack(side=tk.LEFT)
        elif param_name in ["save_folder", "video_path"]:
            entry_frame = ttk.Frame(row_frame)
            entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            entry = ttk.Entry(entry_frame, textvariable=self.vars[param_name], width=25)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            browse_text = "Browse" 
            ttk.Button(entry_frame, text=browse_text, width=8,
                      command=lambda p=param_name: self.browse_path(p)).pack(side=tk.LEFT, padx=(5, 0))
        else:
            entry = ttk.Entry(row_frame, textvariable=self.vars[param_name], width=10)
            entry.pack(side=tk.LEFT)
        
        # Tooltip
        if tooltip:
            self.create_tooltip(row_frame, tooltip)
        
        return row_frame
    
    def create_tooltip(self, widget, text):
        """Crée un tooltip pour un widget"""
        def enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1)
            label.pack()
            widget.tooltip = tooltip
        
        def leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
    
    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.configure("TFrame", padding=10)
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Titre
        title_label = ttk.Label(main_frame, text="BubbleID Analysis", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # SECTION FILE - Sur une seule ligne
        file_frame = ttk.LabelFrame(main_frame, text="File", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        file_content = ttk.Frame(file_frame)
        file_content.pack(fill=tk.X)
        
        # Save Folder
        ttk.Label(file_content, text="Save Folder:", width=12).pack(side=tk.LEFT, padx=(0, 5))
        save_entry = ttk.Entry(file_content, textvariable=self.vars["save_folder"], width=25)
        save_entry.pack(side=tk.LEFT)
        ttk.Button(file_content, text="Browse", command=lambda: self.browse_path("save_folder")).pack(side=tk.LEFT, padx=5)
        
        # Extension
        ttk.Label(file_content, text="Extension:", width=10).pack(side=tk.LEFT, padx=(20, 5))
        ext_entry = ttk.Entry(file_content, textvariable=self.vars["extension"], width=15)
        ext_entry.pack(side=tk.LEFT)
        
        # SECTION TRACKING ET POST PROCESS (côte à côte)
        processing_frame = ttk.Frame(main_frame)
        processing_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Colonne TRACKING (plus large)
        tracking_frame = ttk.LabelFrame(processing_frame, text="Tracking", padding=12)
        tracking_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        # Paramètres Tracking - en deux colonnes
        tracking_params_frame = ttk.Frame(tracking_frame)
        tracking_params_frame.pack(fill=tk.X, pady=5)
        
        # Colonne gauche des paramètres
        left_params = ttk.Frame(tracking_params_frame)
        left_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Colonne droite des paramètres
        right_params = ttk.Frame(tracking_params_frame)
        right_params.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Répartir les paramètres de tracking
        tracking_params = [p for p in self.parameters.items() if p[1][3] == "tracking"]
        mid_index = len(tracking_params) // 2
        
        for i, (param_name, (display_name, var_type, default, section, tooltip)) in enumerate(tracking_params):
            if i < mid_index:
                self.create_parameter_row(left_params, param_name, display_name, var_type, tooltip)
            else:
                self.create_parameter_row(right_params, param_name, display_name, var_type, tooltip)
        
        # Boutons Tracking - en deux colonnes
        tracking_buttons_frame = ttk.Frame(tracking_frame)
        tracking_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Colonne gauche des boutons
        left_buttons = ttk.Frame(tracking_buttons_frame)
        left_buttons.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Colonne droite des boutons
        right_buttons = ttk.Frame(tracking_buttons_frame)
        right_buttons.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tracking_buttons = [
            ("Load BubbleID", self.initialize_data_analysis),
            ("Trim Video", self.trim_video),
            ("Generate Data", self.generate_data),
            ("All Track", self.run_all_tracking),
            ("Scale", self.set_scale),
            ("Tracked Video", self.tracked_video),
        ]
        
        for i, (text, command) in enumerate(tracking_buttons):
            if i < 3:  # 3 premiers boutons à gauche
                ttk.Button(left_buttons, text=text, command=command).pack(fill=tk.X, pady=2)
            else:       # 3 suivants à droite
                ttk.Button(right_buttons, text=text, command=command).pack(fill=tk.X, pady=2)
        
        # Paramètres additionnels Tracking
        extra_params_frame = ttk.Frame(tracking_frame)
        extra_params_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(extra_params_frame, text="Parameters Match:").pack(side=tk.LEFT, padx=(0, 5))
        self.match_var = tk.StringVar(value="No Red Frame")
        ttk.Radiobutton(extra_params_frame, text="No Red Frame", variable=self.match_var, 
                       value="No Red Frame").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(extra_params_frame, text="No All Frame", variable=self.match_var, 
                       value="No All Frame").pack(side=tk.LEFT)
        
        # Colonne POST PROCESS
        postprocess_frame = ttk.LabelFrame(processing_frame, text="Post Process", padding=12)
        postprocess_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        
        # Paramètres Post Process
        for param_name, (display_name, var_type, default, section, tooltip) in self.parameters.items():
            if section == "postprocess":
                self.create_parameter_row(postprocess_frame, param_name, display_name, var_type, tooltip)
        
        # Boutons Post Process
        postprocess_buttons_frame = ttk.Frame(postprocess_frame)
        postprocess_buttons_frame.pack(fill=tk.X, pady=(15, 0))
        
        postprocess_buttons = [
            ("Find Merge", self.find_merge),
            ("Compute Dwell", self.compute_dwell),
            ("Departure Diameter", self.departure_diameter),
            ("Bubble Properties", self.calc_bubble_properties)
        ]
        
        for text, command in postprocess_buttons:
            ttk.Button(postprocess_buttons_frame, text=text, command=command).pack(fill=tk.X, pady=3)
        
        # SECTION LOGS
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=12, width=70, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
    
    def browse_path(self, param_name):
        """Générique pour parcourir les dossiers/fichiers"""
        if param_name == "save_folder":
            folder = filedialog.askdirectory()
            if folder:
                self.vars[param_name].set(folder)
        elif param_name == "video_path":
            filetypes = [("Video files", "*.avi *.mp4 *.mov"), ("All files", "*.*")]
            filepath = filedialog.askopenfilename(filetypes=filetypes)
            if filepath:
                self.vars[param_name].set(filepath)
    
    def log_message(self, message):
        """Ajoute un message aux logs"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_status(self, status):
        """Met à jour la barre de statut"""
        self.status_var.set(status)
        self.root.update()
    
    def validate_parameters(self, required_sections=None):
        """Valide que tous les paramètres requis sont remplis"""
        if required_sections is None:
            required_sections = ["file"]
        
        for param_name, (display_name, var_type, default, section, tooltip) in self.parameters.items():
            if section in required_sections and not self.vars[param_name].get():
                messagebox.showerror("Error", f"Please fill in: {display_name}")
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
        if not self.validate_parameters(["file", "tracking"]):
            return
        
        self.update_status("Initializing DataAnalysis...")
        try:
            # Recharger le module
            importlib.reload(BubbleID_My)
            import BubbleID_My as BubbleID
            
            # Création des chemins
            imagesfolder = os.path.join(self.vars["save_folder"].get(), 
                                      f"trimImages_{self.vars['extension'].get()}")
            model_path = os.path.join("..\\MODELS\\", self.vars["model_weights"].get(), "model_final.pth")
            device = "cuda" if is_available() else "cpu"
            
            # Créer le dossier de sauvegarde
            os.makedirs(self.vars["save_folder"].get(), exist_ok=True)
            
            # Instancier la classe
            self.test120 = BubbleID.DataAnalysis(
                imagesfolder,
                self.vars["video_path"].get(),
                self.vars["save_folder"].get(),
                self.vars["extension"].get(),
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
            self.test120.trimVideo(N_frames_extr=self.vars["n_frames_extr"].get())
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
            mm = self.test120.set_scale_by_two_points(
                frame_idx=0, 
                physical_mm=self.vars["physical_mm"].get(), 
                save=True
            )
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
                self.vars["threshold"].get(), 
                save_rich=True, 
                save_masks=False, 
                save_contours=True, 
                iou_thresh_tid=self.vars["iou_thresh_tid"].get()
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
            self.test120.make_tracked_video(n_frames=100, fps=5, score_thres=0.7)
            self.log_message("Tracked video created successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error creating tracked video: {str(e)}")
            self.update_status("Error")
    
    def run_all_tracking(self):
        """Exécute toutes les étapes du tracking à la suite"""
        if not self.validate_parameters(["file", "tracking"]):
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
            if "Error" in self.status_var.get():
                self.log_message(f"--- {step_name} failed ---")
                return
            self.log_message(f"--- {step_name} completed ---\n")
    
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
                fps=self.vars["fps"].get(),
                class_idx_attached=ATTACHED,
                class_idx_detached=DETACHED,
                class_idx_unknown=UNKNOWN,
                tolerate_unknown_gap=self.vars["tolerate_unknown_gap"].get(),
                min_attached_run=self.vars["min_attached_run"].get(),
                out_csv=os.path.join(self.vars["save_folder"].get(), 
                                   f"dwell_{self.vars['extension'].get()}.csv")
            )
            
            self.log_message("Dwell time computed successfully")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error computing dwell time: {str(e)}")
            self.update_status("Error")
    
    def departure_diameter(self):
        if not self.check_initialized():
            return
        
        self.update_status("Calculating departure diameter...")
        try:
            # À adapter selon votre méthode réelle
            self.log_message("Departure diameter calculation completed")
            self.update_status("Ready")
        except Exception as e:
            self.log_message(f"Error calculating departure diameter: {str(e)}")
            self.update_status("Error")
    
    def calc_bubble_properties(self):
        if not self.check_initialized():
            return
        
        self.update_status("Calculating bubble properties...")
        try:
            # À adapter selon votre méthode réelle
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