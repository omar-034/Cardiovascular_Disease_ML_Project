import os
import re
import subprocess
from pathlib import Path


DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def download_kaggle_data(dataset_id, target_dir=None):
    """
    Télécharge un dataset Kaggle et le dézippe.
    
    Args:
        dataset_id (str): ID du dataset Kaggle (format: username/dataset-name)
        target_dir (str, optional): Répertoire de destination. 
                                    Si None, utilise le chemin par défaut: data/
    
    Returns:
        str: Chemin absolu du répertoire de téléchargement
    """
    
    if not isinstance(dataset_id, str) or not DATASET_ID_PATTERN.fullmatch(dataset_id.strip()):
        raise ValueError("dataset_id doit respecter le format 'owner/dataset-name'.")

    dataset_id = dataset_id.strip()

    # Si target_dir n'est pas spécifié, utiliser le répertoire par défaut
    if target_dir is None:
        # Trouver le répertoire racine du projet
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # src/components/data_download.py -> racine
        target_dir = project_root / 'data'
    else:
        # Convertir en chemin absolu si chemin relatif fourni
        target_dir = Path(target_dir)
        if not target_dir.is_absolute():
            # Si chemin relatif, le résoudre par rapport à la racine du projet
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            target_dir = project_root / target_dir
    
    # Créer le répertoire s'il n'existe pas
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Téléchargement du dataset '{dataset_id}'...")
    print(f"📂 Destination: {target_dir}")
    
    try:
        # Utiliser subprocess pour télécharger avec kaggle CLI
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_id,
                "-p",
                str(target_dir),
                "--unzip",
            ],
            shell=False,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

        print(f"✅ Dataset téléchargé avec succès dans {target_dir}")

    except subprocess.TimeoutExpired as e:
        print("❌ Erreur: délai dépassé pendant le téléchargement Kaggle.")
        raise RuntimeError("Le téléchargement Kaggle a dépassé le délai autorisé.") from e
    except subprocess.CalledProcessError as e:
        print("❌ Erreur lors du téléchargement:")
        print(f"   {e.stderr}")
        raise RuntimeError(f"Erreur Kaggle: {e.stderr}") from e
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        raise
    
    return str(target_dir.resolve())


def get_data_path(relative_path=None):
    """
    Obtient le chemin absolu du répertoire de données.
    
    Args:
        relative_path (str, optional): Chemin relatif dans le répertoire data/
    
    Returns:
        str: Chemin absolu complet
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / 'data'
    
    if relative_path:
        return str((data_dir / relative_path).resolve())
    return str(data_dir.resolve())

