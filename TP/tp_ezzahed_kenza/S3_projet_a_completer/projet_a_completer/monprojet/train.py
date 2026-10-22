"""Entraîne chaque modèle candidat, l'évalue sur le jeu de test et les compare.

Usage, depuis la racine du projet : python -m monprojet.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from monprojet import config
from monprojet.data import load_data, split
from monprojet.models import candidates

HIGHER_IS_BETTER = {"accuracy": True, "recall": True, "precision": True, "f1": True,
                    "mae": False, "rmse": False, "r2": True}


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Prépare les colonnes : nombres mis à l'échelle, catégories encodées, texte vectorisé. (Fournie.)"""
    text = [config.TEXT_COLUMN] if config.TEXT_COLUMN else []
    numeric = [c for c in X.columns if c not in text and pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in text and c not in numeric]
    steps = []
    if numeric:
        steps.append(("nombres", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
    if categorical:
        steps.append(("categories", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                  OneHotEncoder(handle_unknown="ignore")), categorical))
    if text:
        steps.append(("texte", TfidfVectorizer(), config.TEXT_COLUMN))
    return ColumnTransformer(steps, sparse_threshold=0)

def evaluate(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    if config.TASK == "classification":
        average = "binary" if y_test.nunique() == 2 else "macro"
        scores = {
            "accuracy": accuracy_score(y_test, predictions),
            "recall": recall_score(y_test, predictions, average=average, zero_division=0),
            "precision": precision_score(y_test, predictions, average=average, zero_division=0),
            "f1": f1_score(y_test, predictions, average=average, zero_division=0),
        }
    else:
        scores = {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "r2": r2_score(y_test, predictions),
        }

    return {name: round(float(value), 3) for name, value in scores.items()}


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    results = {}
    fitted = {}
    for name, estimator in candidates().items():
        model = Pipeline([
            ("preparation", build_preprocessing(X_train)),
            ("modele", estimator),
        ])
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        fitted[name] = model

    higher_is_better = HIGHER_IS_BETTER[config.MAIN_METRIC]
    table = pd.DataFrame(results).T.sort_values(
        config.MAIN_METRIC, ascending=not higher_is_better, kind="stable"
    )
    print(table)

    best = table.index[0]

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], config.MODELS_DIR / "model.joblib")
    with open(config.MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"main_metric": config.MAIN_METRIC, "best": best, "results": results},
            f,
            indent=2,
        )

    print(f"\nMeilleur modele ({config.MAIN_METRIC}) : {best}")