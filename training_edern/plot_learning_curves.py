
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_learning_curves(metrics_path: str):
    """
    Lit un fichier metrics.json de Detectron2 et trace les courbes d'apprentissage.

    Args:
        metrics_path: Chemin vers le fichier metrics.json.
    """
    # Vérifier si le fichier existe
    if not os.path.exists(metrics_path):
        print(f"Erreur : Le fichier '{metrics_path}' n'a pas été trouvé.")
        return

    # Charger les métriques
    metrics_data = []
    with open(metrics_path, 'r') as f:
        for line in f:
            try:
                metrics_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Avertissement : Impossible de lire une ligne dans {metrics_path}")

    if not metrics_data:
        print("Aucune donnée de métrique trouvée.")
        return

    # Séparer les métriques d'entraînement et d'évaluation
    train_metrics = [m for m in metrics_data if 'total_loss' in m]
    eval_metrics = [m for m in metrics_data if 'bbox/AP' in m]

    # Créer un DataFrame pandas pour une manipulation facile
    df = pd.DataFrame(train_metrics)

    # Créer la figure et les axes pour les graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Courbes d'apprentissage pour {os.path.basename(metrics_path)}", fontsize=16)

    # --- Graphique 1 : Total Loss ---
    ax1.plot(df['iteration'], df['total_loss'], label='Total Loss (Entraînement)')
    # Calculer une moyenne mobile pour lisser la courbe
    if len(df) > 10:
        ax1.plot(df['iteration'], df['total_loss'].rolling(window=10).mean(), label='Moyenne mobile (10 it.)', linestyle='--')
    ax1.set_ylabel('Perte (Loss)')
    ax1.set_title("Perte Totale durant l'Entraînement")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    
    # --- Graphique 2 : Détailler les pertes ---
    loss_components = [key for key in df.columns if key.startswith('loss_')]
    for loss_name in loss_components:
        ax2.plot(df['iteration'], df[loss_name], label=loss_name)
    
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Perte (Loss)')
    ax2.set_title('Composantes de la Perte')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend()
    
    # Afficher les métriques d'évaluation finales si elles existent
    if eval_metrics:
        final_eval = eval_metrics[-1]
        print("\n--- Métriques d'Évaluation Finales ---")
        for key, value in final_eval.items():
            if 'AP' in key:
                print(f"{key}: {value:.4f}")
        print("------------------------------------")


    # Afficher le graphique
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace les courbes d'apprentissage à partir d'un fichier metrics.json de Detectron2."
    )
    parser.add_argument(
        "metrics_file",
        type=str,
        help="Chemin vers le fichier metrics.json à analyser.",
    )
    args = parser.parse_args()
    
    plot_learning_curves(args.metrics_file)
