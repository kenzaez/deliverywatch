"""Les modèles à comparer.

À vous d'écrire candidates(). Les tests de tests/test_models.py disent ce qui est attendu.
"""
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from monprojet import config


def candidates() -> dict:
    """Renvoie un dictionnaire {nom: modèle scikit-learn non entraîné}.

    Attendu :
    - le PREMIER modèle est la baseline : le modèle le plus simple possible
      (DummyClassifier pour une classification, DummyRegressor pour une régression) ;
    - au moins trois vrais modèles ensuite, adaptés à config.TASK ;
    - random_state=config.SEED pour les modèles qui ont du hasard.

    Pistes : sklearn.dummy, sklearn.linear_model, sklearn.ensemble, sklearn.neighbors, sklearn.tree.
    """
    seed = config.SEED

    if config.TASK == "classification":
        return {
            "baseline": DummyClassifier(strategy="most_frequent"),
            "logreg": make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
            ),
            "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "random_forest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=seed
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        }

    return {
        "baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "knn": make_pipeline(StandardScaler(), KNeighborsRegressor()),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }