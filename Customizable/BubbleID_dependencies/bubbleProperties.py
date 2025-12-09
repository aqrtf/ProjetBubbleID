import os, csv, ast, json
import numpy as np, pandas as pd
from csteDef import *

 
# Liste des méthodes valides pour le calcul du diamètre
valid_methods = {"area", "feret_max", "feret_min", "ell_maj", "ell_min", "perim", "mix"}
valid_suffix = {"interp", "discr", "mean"}

def mainProperties(savefolder, extension,
                      diameterMethod=["mix"],
                      interp="mean",
                      chipName="T2", tension=50, 
                      fps=4000, min_attach_frame=4,
                      maxBirthSize = 3000,
                      xCenter = [512-15, 512+15],
                      xEdge = [224, 1024-224]): # TODO chipname/ tension
    """
    Analyse les diamètres de départ et les temps de croissance des bulles.
    Retourne un DataFrame avec les résultats et met à jour un fichier CSV.
    
    maxBirthSize:  Taille maximale (en pixels carrés) pour considérer une bulle comme 'nouvelle' si elle apparaît après les premiers frames

    """

    # Vérifier que toutes les méthodes sont valides
    invalid = [m for m in diameterMethod if m not in valid_methods]
    if invalid:
        raise ValueError(f"Méthodes invalides: {invalid}. "
                         f"Les méthodes valides sont: {sorted(valid_methods)}")

    # Vérifier que toutes les méthodes sont valides
    invalid = [m for m in interp if m not in valid_suffix]
    if interp not in valid_suffix:
        raise ValueError(f"suffixe invalides: {invalid}. "
                         f"Les méthodes valides sont: {sorted(valid_suffix)}")
    colonnes = [f"D_{method}_mm_{interp}" for method in diameterMethod]

    # Chemins vers les fichiers
    departure_csv = os.path.join(savefolder, f"departure_{extension}.csv")
    evolution_csv = os.path.join(savefolder, f"evolutionID_{extension}.csv")
    out_csv = os.path.join(savefolder, f"mainProperties.csv")  

    # Vérifications de sécurité
    for path in [departure_csv, evolution_csv]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} non trovato.")

    # Chargement du CSV de départ
    df_depart = pd.read_csv(departure_csv)
    df_depart.columns = df_depart.columns.str.strip()  # nettoyage des colonnes

    departDiameters = []
    growingTimes = []

    # Parcours des bulles
    for bubble in df_depart.itertuples():
        if bubble.note == "ok":
            n_attach_frame = ((bubble.last_attached_frame+1 + bubble.detach_frame)/2 - bubble.attach_start_frame + 1)
            if n_attach_frame >= min_attach_frame:
                # La bulle se détache, ce n'est pas une erreur
                # Extraction du diamètre de départ
                departDiameters.append(df_depart[colonnes].mean(axis=1))

                if (bubble.firstArea < maxBirthSize) and (bubble.attach_start_frame > 1) :
                    # On a toute la croissance de la bulle
                    if (xCenter[0] < bubble.firstX < xCenter[1]) or (bubble.firstX < xEdge[0]) or (bubble.firstX > xEdge[1]):
                        # La bulle est centrée, on l'exclu car il y a interference avec les bulle devant
                        # pareil si elle est proche des bords
                        growingTimes.append(np.nan)
                    else:
                        growingTimes.append(n_attach_frame / fps)
                else:
                    growingTimes.append(np.nan)

    # Conversion en arrays
    departDiameters = np.array(departDiameters)
    growingTimes = np.array(growingTimes)
    frequencies = 1/growingTimes

    # retrait des outliers
    from fonctions import rmoutliers, rmmissing
    departDiameters, _ = rmmissing(departDiameters)
    frequencies, _ = rmmissing(frequencies)
    departDiameters, _ = rmoutliers(departDiameters)
    frequencies, _ = rmoutliers(frequencies)

    # Calcul des statistiques
    departDiameterMean = np.mean(departDiameters) if departDiameters.size > 0 else np.nan
    departDiameterStd = np.std(departDiameters) if departDiameters.size > 0 else np.nan
    frequencyMean = np.mean(frequencies) if frequencies.size > 0 else np.nan
    frequencyStd = np.std(frequencies) if frequencies.size > 0 else np.nan

    # Calcul des vitesses via la fonction bubble_velocities
    from BubbleID_dependencies.velocities import bubble_velocities
    attach_vel, detach_vel = bubble_velocities(savefolder, extension,
                                               minPointForVelocity=2, fps=fps)
    
    # Construction du DataFrame résultat
    results = pd.DataFrame([{
        "chip": chipName,
        "tension": tension,
        "extension": extension,
        "departDiameter": departDiameterMean,
        "departDiameter_std": departDiameterStd,
        "frequency": frequencyMean,
        "frequency_std": frequencyStd,
        "elevationVelocity": detach_vel.vMean_mm,
        "elevationVelocity_std": detach_vel.vStd_mm,
        "growingVelocity": attach_vel.vMean_mm,
        "growingVelocity_std": attach_vel.vStd_mm,
    }])

    # Sauvegarde dans le CSV (append)
    results.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    dict_json = {"attachV": [arr.tolist() for arr in attach_vel.vy_mm],
                 "detachV": [arr.tolist() for arr in detach_vel.vy_mm]}
    # Sauvegarde avec indentation
    outJsonPath = os.path.join(savefolder, f"velocities_{extension}.json")
    with open(outJsonPath, "w", encoding="utf-8") as f:
        json.dump(dict_json, f, indent=4, ensure_ascii=False)

    print(f"File salvato: {out_csv}")
    return results

# mainProperties(r"Inputs\T87_out", "T87_60V1")