import pandas as pd
import glob

# Dossier où se trouvent tes fichiers CSV
dossier = r"Results\T87\T87_out"

# Liste de tous les fichiers CSV dans le dossier
fichiers_csv = glob.glob(dossier + "/departure*.csv")

# Fichier Excel de sortie
fichier_excel = "resultat.xlsx"

# Création d'un writer Excel
with pd.ExcelWriter(fichier_excel, engine="xlsxwriter") as writer:
    for fichier in fichiers_csv:
        # Lecture du CSV
        df = pd.read_csv(fichier)
        
        # --- Sélection des colonnes ---
        # Exemple : garder seulement les colonnes "Nom" et "Age"
        colonnes_a_garder = ["bubble_id","attach_start_frame",'last_attached_frame','detach_frame','dwellFrame','note','firstArea','firstX']
        df = df[colonnes_a_garder]
        
        # --- Sélection des lignes ---
        # Exemple : garder seulement les lignes où Age > 30
        df = df[df["note"] == 'ok']
        df = df[df["firstArea"] < 3000]
        df = df[(df["firstX"] < (512-15)) | (df["firstX"] > (512+15))]
        
        # Nom de la feuille = nom du fichier sans extension
        nom_feuille = fichier.split("\\")[-1].replace(".csv", "")
        
        # Écriture dans une feuille Excel
        df.to_excel(writer, sheet_name=nom_feuille, index=False)

print("✅ Export terminé dans", fichier_excel)
