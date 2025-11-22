import os, math, csv, ast
import numpy as np, pandas as pd
from csteDef import *
MIN_ATTACH_FRAME = 4
fps = 4000
chip = "T"
tension = 50

# Choix de la methode de calcul du diametre
# Liste des méthodes valides
valid_methods = {"area", "feret_max", "feret_min", "ell_maj", "ell_min", "perim", "mix"}

diameterMethod = ["perim", "area"]   # peut contenir plusieurs methodes
# Vérifier que toutes les méthodes sont valides
invalid = [m for m in diameterMethod if m not in valid_methods]
if invalid:
    raise ValueError(f"Méthodes invalides: {invalid}. "
                        f"Les méthodes valides sont: {sorted(valid_methods)}")
interp = True                            # True = interp, False = discr

# Construire les noms de colonnes dynamiquement
suffix = "interp" if interp else "discr"
colonnes = [f"D_{method}_mm_{suffix}" for method in diameterMethod]

class Myclass:
    savefolder = r"C:\Users\afara\Documents\EPFL\cours\MA3\Projet\ProjetBubbleID\My_output\Test6"
    extension = "Test6"
    mm_per_px = 0.023730276134122288
    
self = Myclass()


# Chemins vers les fichiers de données
departure_csv = os.path.join(self.savefolder, f"departure_{self.extension}.csv")          # Diametre et depart des bulles
evolution_csv = os.path.join(self.savefolder, f"evolutionID_{self.extension}.csv") 


out_csv = os.path.join(self.savefolder, f"xxx_{self.extension}.csv") # TODO choose name of exit

# VÉRIFICATIONS DE SÉCURITÉ: les fichiers existent-ils?
if not os.path.isfile(departure_csv):
    raise FileNotFoundError(f"{departure_csv} non trovato.")
if not os.path.isfile(evolution_csv):
    raise FileNotFoundError(f"{evolution_csv} non trovato.")

df_depart = pd.read_csv(departure_csv)
df_depart.columns = df_depart.columns.str.strip() # on suppr les espace qui peuvent apparaitre si on align les csv
departDiameters = []
growingTimes = []
for bubble in df_depart.itertuples():
    if bubble.detach_frame is not None:
        n_attach_frame = (bubble.detach_frame-bubble.attach_start_frame+1)
        if n_attach_frame>=MIN_ATTACH_FRAME:
            # la bulle se detache, il ne s'agit pas d'une erreur
            # Extraction du diametre de depart
            departDiameters.append(df_depart[colonnes].mean(axis=1)) 
            
            if bubble.birth:
                # On a toute la croissance de la bulle
                growingTimes.append(n_attach_frame/fps) #TODO fps
            else:
                growingTimes.append(np.nan)

departDiameters = np.array(departDiameters)
growingTimes = np.array(growingTimes)

departDiameterMean = np.nanmean(departDiameters) if departDiameters.size > 0 else np.nan
growingTimesMean = np.nanmean(growingTimes) if growingTimes.size > 0 else np.nan

results = pd.DataFrame([{
    "chip": chip,
    "tension": tension,
    "departDiameter": departDiameterMean,
    "growingTime": growingTimesMean,
    "elevationVelocity": None, #TODO
    "growingVelocity": None,
}])
# Si le fichier n'existe pas encore → créer avec header
results.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)


            

            

