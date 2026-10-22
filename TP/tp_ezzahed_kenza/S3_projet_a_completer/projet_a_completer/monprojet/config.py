"""Configuration du projet : c'est le seul fichier à adapter à votre sujet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Vos données -------------------------------------------------------------
DATA_PATH = ROOT / "data" / "cantine_repas.csv"
TARGET = "repas_servis"                 # nombre de repas servis le jour J
FEATURES = [                            # explicite : on exclut "date" et la cible
    "jour", "inscrits", "menu", "temperature_c", "pluie_mm",
    "examens", "sortie_scolaire", "veille_vacances",
    "repas_semaine_precedente",
]
TEXT_COLUMN = None

# --- Type de problème --------------------------------------------------------
TASK = "regression"
MAIN_METRIC = "mae"                     # en nombre de repas ; "rmse" et "r2" possibles

# --- Découpage entraînement / test -------------------------------------------
SPLIT = "chronologique"                 # série temporelle : pas de mélange aléatoire
TIME_COLUMN = "date"
TEST_SIZE = 0.2                         # ≈ les 68 derniers jours d'école
SEED = 42

MODELS_DIR = ROOT / "models"