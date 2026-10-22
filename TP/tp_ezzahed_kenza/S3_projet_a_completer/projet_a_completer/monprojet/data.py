"""Chargement et découpage des données.

À vous d'écrire load_data() et split(). Les tests de tests/test_data.py disent ce qui est attendu.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from monprojet import config


def load_data(path=None) -> pd.DataFrame:
    """Lit le fichier CSV et renvoie un DataFrame.

    Attendu :
    - lire config.DATA_PATH (ou `path` s'il est donné) ;
    - si la colonne config.TARGET n'existe pas, lever une ValueError avec un message clair ;
    - retirer les lignes où la cible est manquante.

    Pistes : pd.read_csv, df.columns, df.dropna(subset=[...]).
    """
    df = pd.read_csv(path if path is not None else config.DATA_PATH)
    if config.TARGET not in df.columns:
        raise ValueError(
            f"Colonne cible '{config.TARGET}' introuvable. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    return df.dropna(subset=[config.TARGET]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    """Les colonnes d'entrée : celles de config.FEATURES, ou toutes sauf la cible et la date. (Fournie.)"""
    if config.FEATURES is not None:
        return list(config.FEATURES)
    excluded = {config.TARGET, config.TIME_COLUMN}
    return [c for c in df.columns if c not in excluded]


def split(df: pd.DataFrame):
    """Renvoie X_train, X_test, y_train, y_test.

    Attendu :
    - X = les colonnes de feature_columns(df) ; y = la colonne cible ;
    - si config.SPLIT == "random" : un découpage aléatoire de taille config.TEST_SIZE,
      avec la graine config.SEED, et stratifié sur y pour une classification ;
    - si config.SPLIT == "chronologique" : trier selon config.TIME_COLUMN et garder
      les lignes les plus récentes pour le test (le modèle apprend le passé, il est testé sur le futur).

    Pistes : train_test_split(..., test_size=..., random_state=..., stratify=...) ;
    pd.to_datetime(...).argsort() ; X.iloc[...].
    """
    X = df[feature_columns(df)]
    y = df[config.TARGET]

    if config.SPLIT == "chronologique":
        order = pd.to_datetime(df[config.TIME_COLUMN]).argsort().to_numpy()
        X = X.iloc[order]
        y = y.iloc[order]
        n_test = max(1, int(round(len(df) * config.TEST_SIZE)))
        cut = len(df) - n_test
        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y if config.TASK == "classification" else None,
    )