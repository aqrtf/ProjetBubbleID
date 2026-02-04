
"""
bubbleProperties.py
-------------------
Analyse et calcule les propriétés principales des bulles (diamètre de départ, temps de croissance, fréquences, vitesses)
à partir de fichiers CSV issus du traitement vidéo. Les résultats sont sauvegardés dans un CSV et un JSON.

Fonction principale :
    - mainProperties(savefolder, extension, ...):
        Calcule et sauvegarde les propriétés principales des bulles pour une expérience donnée.

Fichiers lus/écrits (suffixe = extension) :
    - departure_<suffixe>.csv (lecture)
    - evolutionID_<suffixe>.csv (lecture)
    - mainProperties.csv (écriture, append)
    - velocities_<suffixe>.json (écriture)

Dépendances internes :
    - csteDef
    - velocities (fonction bubble_velocities)
    - frequency (fonction count_detachment_transitions)
    - functions.rmmissing, functions.rmoutliers

Auteurs : [à compléter]
Date : [à compléter]
"""

import os, csv, ast, json
import numpy as np, pandas as pd

from csteDef import *
from functions.rmmissing import rmmissing
from functions.rmoutliers import rmoutliers

 
# Liste des méthodes valides pour le calcul du diamètre
valid_methods = {"area", "feret_max", "feret_min", "ell_maj", "ell_min", "perim", "mix"}
valid_suffix = {"interp", "discr", "mean"}

def mainProperties(savefolder, extension,
                      diameterMethod=["mix"],
                      interp="mean",
                      chipName="-", tension=0, 
                      fps=4000, min_attach_frame=4,
                      maxBirthSize = 3000,
                      xCenter = [512-15, 512+15],
                      xEdge = [224, 1024-224]): # TODO chipname/ tension
    """
    Analyse les diamètres de départ et les temps de croissance des bulles.
    Retourne un DataFrame avec les résultats et met à jour un fichier CSV.
    
    maxBirthSize:  Taille maximale (en pixels carrés) pour considérer une bulle comme 'nouvelle' si elle apparaît après les premiers frames
    diameterMethod: Methode d'evaluation du diametre parmi "area", "feret_max", "feret_min", "ell_maj", "ell_min", "perim", "mix"
    interp: Methode d'interpolation parmi "interp", "discr", "mean"
    chipName: Nom de la chip (renseigne dans le csv de sortie et par le code multipleAnalysis)
    tension: Tension appliquée (renseigne dans le csv de sortie et par le code multipleAnalysis)
    fps: Frames par seconde de la vidéo (utilisé pour le calcul des temps)
    min_attach_frame: Nombre minimum de frames d'attachement pour considérer une bulle comme valide
    xCenter: Zone centrale à exclure pour le calcul du temps de croissance (interférence avec les bulles en avant)
    xEdge: Zone proche des bords à exclure pour le calcul du temps de croissance (interférence avec les bords)

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
    out_csv = os.path.join(os.path.dirname(savefolder), f"mainProperties.csv")  

    # Vérifications de sécurité
    for path in [departure_csv, evolution_csv]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} non trovato.")

    # Chargement du CSV de départ
    df_depart = pd.read_csv(departure_csv)
    df_depart.columns = df_depart.columns.str.strip()  # nettoyage des colonnes

    # Nouveau df des bulles valides pour verif
    bulleCroissanceValide = []
    bulleDepartValide = []
    departDiameters = []
    growingTimes = []

    # Parcours des bulles
    for bubble in df_depart.itertuples():
        if bubble.note == "ok":
            # n_attach_frame = ((bubble.last_attached_frame+1 + bubble.detach_frame)/2 - bubble.attach_start_frame + 1)
            # Plus restrictif:
            n_attach_frame = (bubble.detach_frame - bubble.attach_start_frame + 1)
            if n_attach_frame >= min_attach_frame:
                # La bulle se détache, ce n'est pas une erreur
                # Extraction du diamètre de départ
                departDiameters.append(df_depart.loc[bubble.Index, colonnes].mean())
                bulleDepartValide.append(bubble)

                if (bubble.firstArea < maxBirthSize) and (bubble.attach_start_frame > 1) :
                    # On a toute la croissance de la bulle
                    if (xCenter[0] < bubble.firstX < xCenter[1]) or (bubble.firstX < xEdge[0]) or (bubble.firstX > xEdge[1]):
                        # La bulle est centrée, on l'exclu car il y a interference avec les bulle devant
                        # pareil si elle est proche des bords
                        growingTimes.append(np.nan)
                    else:
                        growingTimes.append(n_attach_frame / fps)
                        bulleCroissanceValide.append(bubble)
                else:
                    growingTimes.append(np.nan)

    # Save a csv of valid bubbles for debug
    debug_csv = os.path.join(savefolder, f"validGrowthBubbles_{extension}.csv")
    with open(debug_csv, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df_depart.columns)
        for bubble in bulleCroissanceValide:
            writer.writerow(bubble[1:])  # Exclure l'index
    print(f"Valid growth bubbles saved to: {debug_csv}")
    # Save a csv of valid bubbles for debug
    debug_csv = os.path.join(savefolder, f"validDepartBubbles_{extension}.csv")
    with open(debug_csv, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df_depart.columns)
        for bubble in bulleDepartValide:
            writer.writerow(bubble[1:])  # Exclure l'index
    print(f"Valid departure bubbles saved to: {debug_csv}")
    
    # Conversion en arrays
    departDiameters = np.array(departDiameters)
    growingTimes = np.array(growingTimes)
    frequencies = 1/growingTimes

    # retrait des outliers
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
    from velocities import bubble_velocities
    attach_vel, detach_vel = bubble_velocities(savefolder, extension,
                                               minPointForVelocity=2, fps=fps)
    
    from frequency import count_detachment_transitions
    _, frequency2 = count_detachment_transitions(savefolder, extension)

    # Construction du DataFrame résultat
    results = pd.DataFrame([{
        "chip": chipName,
        "tension": tension,
        "extension": extension,
        "departDiameter": departDiameterMean,
        "departDiameter_std": departDiameterStd,
        "frequency": frequencyMean,
        "frequency_std": frequencyStd,
        "frequency2": frequency2,
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


def mainPropertiesMean(savefolder, globalExtension,
                      diameterMethod=["mix"],
                      interp="mean",
                      chipName="-", tension=0, 
                      fps=4000, min_attach_frame=4,
                      maxBirthSize = 3000,
                      xCenter = [512-15, 512+15],
                      xEdge = [224, 1024-224]): # TODO chipname/ tension
    """
    Analyse les diamètres de départ et les temps de croissance des bulles en prenant la moyenne entre 1 et 2.
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


    departDiameters = []
    growingTimes = []
    for iExp in [1,2]:  # 1 et 2
        extension = globalExtension + str(iExp)
        # Chemins vers les fichiers
        departure_csv = os.path.join(savefolder, f"departure_{extension}.csv")
        evolution_csv = os.path.join(savefolder, f"evolutionID_{extension}.csv")
        out_csv = os.path.join(os.path.dirname(savefolder), f"mainProperties.csv")  

        # Vérifications de sécurité
        for path in [departure_csv, evolution_csv]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} non trovato.")

        # Chargement du CSV de départ
        df_depart = pd.read_csv(departure_csv)
        df_depart.columns = df_depart.columns.str.strip()  # nettoyage des colonnes

        

        # Parcours des bulles
        for bubble in df_depart.itertuples():
            if bubble.note == "ok":
                # n_attach_frame = ((bubble.last_attached_frame+1 + bubble.detach_frame)/2 - bubble.attach_start_frame + 1)
                # Plus restrictif:
                n_attach_frame = (bubble.detach_frame - bubble.attach_start_frame + 1)
                if n_attach_frame >= min_attach_frame:
                    # La bulle se détache, ce n'est pas une erreur
                    # Extraction du diamètre de départ pour la bulle courante
                    departDiameters.append(df_depart.loc[bubble.Index, colonnes].mean())

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
    departDiameters, _ = rmmissing(departDiameters)
    frequencies, _ = rmmissing(frequencies)
    departDiameters, _ = rmoutliers(departDiameters)
    frequencies, _ = rmoutliers(frequencies)

    # Calcul des statistiques
    departDiameterMean = np.mean(departDiameters) if departDiameters.size > 0 else np.nan
    departDiameterStd = np.std(departDiameters) if departDiameters.size > 0 else np.nan
    frequencyMean = np.mean(frequencies) if frequencies.size > 0 else np.nan
    frequencyStd = np.std(frequencies) if frequencies.size > 0 else np.nan

    #  TODO adapter pour la moyenne
    # # Calcul des vitesses via la fonction bubble_velocities
    from velocities import bubble_velocities, velocities
    attach_vel_tot = velocities()
    detach_vel_tot = velocities()
    
    attach_vel1, detach_vel1 = bubble_velocities(savefolder, globalExtension+"1",
                                                minPointForVelocity=2, fps=fps)
    attach_vel2, detach_vel2 = bubble_velocities(savefolder, globalExtension+"2",
                                                minPointForVelocity=2, fps=fps)
    
    attach_vel_tot.vy = attach_vel1.vy + attach_vel2.vy
    attach_vel_tot.diameter = attach_vel1.diameter + attach_vel2.diameter
    detach_vel_tot.vy = detach_vel1.vy + detach_vel2.vy
    detach_vel_tot.diameter = detach_vel1.diameter + detach_vel2.diameter
    attach_vel_tot.computeMean()
    detach_vel_tot.computeMean()
    # read scale
    scale_path = os.path.join(savefolder, f"scale_{extension}.json")
    # Conversion pixels → mm
    with open(scale_path, "r") as f:
        dataScale = json.load(f)
    mm_per_px = float(dataScale["mm_per_px"])
    attach_vel_tot.convert2mm(mm_per_px)
    detach_vel_tot.convert2mm(mm_per_px)
    
    # from frequency import count_detachment_transitions
    # _, frequency2 = count_detachment_transitions(savefolder, extension)

    # Construction du DataFrame résultat
    results = pd.DataFrame([{
        "chip": chipName,
        "tension": tension,
        "extension": globalExtension,
        "departDiameter": departDiameterMean,
        "departDiameter_std": departDiameterStd,
        "frequency": frequencyMean,
        "frequency_std": frequencyStd,
        "frequency2": None, # frequency2,
        "elevationVelocity": detach_vel_tot.vMean_mm,
        "elevationVelocity_std": detach_vel_tot.vStd_mm,
        "growingVelocity": attach_vel_tot.vMean_mm,
        "growingVelocity_std": attach_vel_tot.vStd_mm,
    }])

    # Sauvegarde dans le CSV (append)
    results.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    # dict_json = {"attachV": [arr.tolist() for arr in attach_vel.vy_mm],
    #              "detachV": [arr.tolist() for arr in detach_vel.vy_mm]}
    # # Sauvegarde avec indentation
    # outJsonPath = os.path.join(savefolder, f"velocities_{extension}.json")
    # with open(outJsonPath, "w", encoding="utf-8") as f:
    #     json.dump(dict_json, f, indent=4, ensure_ascii=False)

    print(f"File salvato: {out_csv}")
    return results




if __name__ == "__main__":
    # Example usage for testing purposes
    savefolder = r"Inputs\T87_out"
    globalExtension = "T87_60V"

    # Test mainProperties
    mainPropertiesMean(savefolder, globalExtension)