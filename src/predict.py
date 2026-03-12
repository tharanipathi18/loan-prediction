from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
COLUMNS_PATH = MODEL_DIR / "model_columns.pkl"


def _build_drive_url(url_env: str | None, id_env: str | None, default_url: str | None) -> str | None:
    if url_env:
        return url_env
    if id_env:
        return f"https://drive.google.com/uc?id={id_env}"
    return default_url


MODEL_URL = _build_drive_url(
    url_env=os.getenv("MODEL_URL"),
    id_env=os.getenv("MODEL_FILE_ID"),
    default_url="https://drive.google.com/uc?id=1yUHf5QEQAUK-VsVrMFg_koiO0JFlTVWp",
)
COLUMNS_URL = _build_drive_url(
    url_env=os.getenv("MODEL_COLUMNS_URL"),
    id_env=os.getenv("MODEL_COLUMNS_FILE_ID"),
    default_url=None,
)


def _download_if_missing(path: Path, url: str | None, label: str) -> None:
    if path.exists():
        return

    if not url:
        raise FileNotFoundError(
            f"{label} not found at {path}. Set a Google Drive URL/ID in env vars."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("gdown is required to download missing model artifacts.") from exc

    gdown.download(url=url, output=str(path), quiet=True, fuzzy=True)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to download {label} from Google Drive.")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _normalize_columns(columns_obj: Any) -> list[str]:
    if isinstance(columns_obj, pd.Index):
        columns = columns_obj.tolist()
    elif isinstance(columns_obj, Sequence) and not isinstance(columns_obj, (str, bytes)):
        columns = list(columns_obj)
    else:
        raise TypeError("model_columns.pkl must contain a list-like set of feature names.")

    columns = [str(col) for col in columns]
    if not columns:
        raise ValueError("model_columns.pkl is empty.")
    if len(columns) != len(set(columns)):
        raise ValueError("model_columns.pkl contains duplicate feature names.")
    return columns


def _model_feature_names(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(col) for col in model.feature_names_in_]
    return []


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, list[str]]:
    _download_if_missing(MODEL_PATH, MODEL_URL, "Model file")
    model = _load_pickle(MODEL_PATH)

    model_columns: list[str] = []

    if not COLUMNS_PATH.exists():
        if COLUMNS_URL:
            _download_if_missing(COLUMNS_PATH, COLUMNS_URL, "Model columns file")
    if COLUMNS_PATH.exists():
        try:
            model_columns = _normalize_columns(_load_pickle(COLUMNS_PATH))
        except Exception:
            model_columns = []

    if not model_columns:
        model_columns = _model_feature_names(model)
    if not model_columns:
        raise FileNotFoundError(
            "Training columns are unavailable. Provide model/model_columns.pkl "
            "or ensure the model has feature_names_in_."
        )

    training_columns = _model_feature_names(model)
    if training_columns:
        model_columns = training_columns

    return model, model_columns


def _prepare_input_frame(data: Mapping[str, Any], model_columns: list[str]) -> pd.DataFrame:
    if isinstance(data, Mapping):
        data_dict = dict(data)
    else:
        try:
            data_dict = dict(data)
        except Exception:
            data_dict = {}

    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=model_columns, fill_value=0)
    df = df.where(df.notna(), 0)
    null_tokens = {"", "NA", "N/A", "NAN", "NONE", "NULL"}
    df = df.apply(
        lambda column: column.map(
            lambda value: 0
            if isinstance(value, str) and value.strip().upper() in null_tokens
            else value
        )
    )
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df.astype(float)


def _extract_probability(model: Any, features: pd.DataFrame) -> float:
    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is None:
        return float("nan")

    try:
        probabilities = predict_proba(features)
    except Exception:
        return float("nan")

    if len(probabilities) == 0:
        return float("nan")

    row_probabilities = probabilities[0]
    classes = list(getattr(model, "classes_", []))

    if classes and 1 in classes:
        return float(row_probabilities[classes.index(1)])
    if len(row_probabilities) > 1:
        return float(row_probabilities[1])
    return float(row_probabilities[0])


def predict_loan(data: Mapping[str, Any]) -> tuple[Any, float]:
    model, model_columns = _load_artifacts()
    features = _prepare_input_frame(data, model_columns)

    try:
        prediction = model.predict(features)[0]
    except Exception:
        classes = list(getattr(model, "classes_", []))
        fallback_prediction = classes[0] if classes else 0
        return fallback_prediction, float("nan")

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    probability = _extract_probability(model, features)
    return prediction, probability
