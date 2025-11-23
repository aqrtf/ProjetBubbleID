import os, csv, ast
import numpy as np, pandas as pd
from csteDef import *


# Liste des méthodes valides pour le calcul du diamètre
valid_methods = {"area", "feret_max", "feret_min", "ell_maj", "ell_min", "perim", "mix"}


def mainProperties(savefolder, extension,
                      diameterMethod=["mix"],
                      interp=True,
                      chipName="T", tension=50,
                      fps=4000, min_attach_frame=4):
    """
    Analyse les diamètres de départ et les temps de croissance des bulles.
    Retourne un DataFrame avec les résultats et met à jour un fichier CSV.
    """

    # Vérifier que toutes les méthodes sont valides
    invalid = [m for m in diameterMethod if m not in valid_methods]
    if invalid:
        raise ValueError(f"Méthodes invalides: {invalid}. "
                         f"Les méthodes valides sont: {sorted(valid_methods)}")

    # Construire les noms de colonnes dynamiquement
    suffix = "interp" if interp else "discr"
    colonnes = [f"D_{method}_mm_{suffix}" for method in diameterMethod]

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
        if bubble.detach_frame is not None:
            n_attach_frame = (bubble.detach_frame - bubble.attach_start_frame + 1)
            if n_attach_frame >= min_attach_frame:
                # La bulle se détache, ce n'est pas une erreur
                # Extraction du diamètre de départ
                departDiameters.append(df_depart[colonnes].mean(axis=1))

                if bubble.birth:
                    # On a toute la croissance de la bulle
                    growingTimes.append(n_attach_frame / fps)
                else:
                    growingTimes.append(np.nan)

    # Conversion en arrays
    departDiameters = np.array(departDiameters)
    growingTimes = np.array(growingTimes)

    # Calcul des statistiques
    departDiameterMean = np.nanmean(departDiameters) if departDiameters.size > 0 else np.nan
    departDiameterStd = np.nanstd(departDiameters) if departDiameters.size > 0 else np.nan
    growingTimeMean = np.nanmean(growingTimes) if growingTimes.size > 0 else np.nan
    growingTimeStd = np.nanstd(growingTimes) if growingTimes.size > 0 else np.nan

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
        "growingTime": growingTimeMean,
        "growingTime_std": growingTimeStd,
        "elevationVelocity": detach_vel.vMean_mm,
        "elevationVelocity_std": detach_vel.vStd_mm,
        "growingVelocity": attach_vel.vMean_mm,
        "growingVelocity_std": attach_vel.vStd_mm,
    }])

    # Sauvegarde dans le CSV (append)
    results.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)

    return results
