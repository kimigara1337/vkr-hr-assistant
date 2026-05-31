
import os
import numpy as np
import pandas as pd
import joblib


from pathlib import Path

PROJECT_DIR = os.environ.get("VKR_PROJECT_DIR")
if PROJECT_DIR is None:
    PROJECT_DIR = str(Path(__file__).resolve().parents[1])

DATA_DIR = f"{PROJECT_DIR}/data"
MODELS_DIR = f"{PROJECT_DIR}/models"
REPORTS_DIR = f"{PROJECT_DIR}/reports"

FINAL_MODEL_PATH = f"{MODELS_DIR}/final_hybrid_e5_tfidf_xgb_model.joblib"
FINAL_FEATURES_PATH = f"{MODELS_DIR}/final_hybrid_e5_tfidf_features.joblib"
PAIR_TEXT_MODEL_PATH = f"{MODELS_DIR}/pair_text_tfidf_oof_logreg.joblib"

E5_MODEL_NAME = "intfloat/multilingual-e5-base"


REQUIRED_BASE_COLUMNS = [
    "idCv",
    "idVacancy",
    "vacancyName",
    "positionName",
    "cv_text",
    "vacancy_text",
]


def load_models(load_e5=False):
    """
    Загружает финальную модель, список признаков и TF-IDF модель.
    E5-модель загружается только при необходимости, потому что она тяжёлая.
    """
    missing_paths = []

    for path in [FINAL_MODEL_PATH, FINAL_FEATURES_PATH, PAIR_TEXT_MODEL_PATH]:
        if not os.path.exists(path):
            missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(
            "Не найдены необходимые файлы модели:\n" + "\n".join(missing_paths)
        )

    final_model = joblib.load(FINAL_MODEL_PATH)
    feature_cols = joblib.load(FINAL_FEATURES_PATH)
    pair_text_model = joblib.load(PAIR_TEXT_MODEL_PATH)

    models = {
        "final_model": final_model,
        "feature_cols": feature_cols,
        "pair_text_model": pair_text_model,
        "e5_model": None,
    }

    if load_e5:
        try:
            from sentence_transformers import SentenceTransformer
            models["e5_model"] = SentenceTransformer(E5_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                "Не удалось загрузить E5-модель. "
                "Проверь установку sentence-transformers."
            ) from e

    return models


def check_input_columns(df):
    """
    Проверяет, что во входном CSV есть базовые колонки.
    """
    missing_cols = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "Во входных данных не хватает обязательных колонок: "
            + ", ".join(missing_cols)
        )


def build_pair_text(df):
    """
    Создаёт общий текст пары вакансия-резюме, если он не был передан во входном CSV.
    """
    df = df.copy()

    if "pair_text" not in df.columns:
        df["pair_text"] = (
            "ВАКАНСИЯ:\n"
            + "vacancyName: " + df["vacancyName"].fillna("").astype(str) + "\n"
            + "vacancy_text: " + df["vacancy_text"].fillna("").astype(str) + "\n\n"
            + "РЕЗЮМЕ:\n"
            + "positionName: " + df["positionName"].fillna("").astype(str) + "\n"
            + "cv_text: " + df["cv_text"].fillna("").astype(str)
        )
    else:
        df["pair_text"] = df["pair_text"].fillna("").astype(str)

    return df


def add_text_lengths(df):
    """
    Добавляет длины текстов, если их нет.
    """
    df = df.copy()

    if "cv_text_len" not in df.columns:
        df["cv_text_len"] = df["cv_text"].fillna("").astype(str).str.len()

    if "vacancy_text_len" not in df.columns:
        df["vacancy_text_len"] = df["vacancy_text"].fillna("").astype(str).str.len()

    if "pair_text_len" not in df.columns:
        df["pair_text_len"] = df["pair_text"].fillna("").astype(str).str.len()

    return df


def add_pair_text_score(df, pair_text_model):
    """
    Добавляет TF-IDF score для пары вакансия-резюме.
    """
    df = df.copy()

    if "pair_text_score_oof" not in df.columns:
        df["pair_text_score_oof"] = pair_text_model.predict_proba(
            df["pair_text"].fillna("").astype(str)
        )[:, 1]

    return df


def add_e5_similarity(df, e5_model=None, batch_size=64):
    """
    Добавляет E5 similarity, если её нет во входном CSV.

    Если колонка e5_similarity уже есть, повторно ничего не считает.
    """
    df = df.copy()

    if "e5_similarity" in df.columns:
        df["e5_similarity"] = pd.to_numeric(df["e5_similarity"], errors="coerce")
        return df

    if e5_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            e5_model = SentenceTransformer(E5_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                "Во входном CSV нет колонки e5_similarity, "
                "а E5-модель загрузить не удалось. "
                "Установи sentence-transformers или передай готовую колонку e5_similarity."
            ) from e

    cv_texts = ("passage: " + df["cv_text"].fillna("").astype(str)).tolist()
    vacancy_texts = ("query: " + df["vacancy_text"].fillna("").astype(str)).tolist()

    cv_emb = e5_model.encode(
        cv_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    vacancy_emb = e5_model.encode(
        vacancy_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    df["e5_similarity"] = np.sum(cv_emb * vacancy_emb, axis=1)

    return df


def prepare_numeric_features(df):
    """
    Готовит числовые признаки так же, как при обучении финальной модели.
    """
    df = df.copy()

    default_numeric_cols = [
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
    ]

    for col in default_numeric_cols:
        if col not in df.columns:
            df[col] = 0

    numeric_cols = [
        "pair_text_len",
        "cv_text_len",
        "vacancy_text_len",
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
        "e5_similarity",
        "pair_text_score_oof",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["length_ratio"] = (
        df["cv_text_len"] / df["vacancy_text_len"].replace(0, np.nan)
    )

    df["experience_diff"] = (
        df["experience"] - df["experienceRequirements"]
    )

    df["experience_enough"] = (
        df["experience"] >= df["experienceRequirements"]
    ).astype(int)

    df["salary_match"] = (
        (df["salary_cv"] >= df["salaryMin_vacancy"]) &
        (df["salary_cv"] <= df["salaryMax_vacancy"])
    ).astype(int)

    df["salary_above_max"] = (
        df["salary_cv"] > df["salaryMax_vacancy"]
    ).astype(int)

    df["salary_below_min"] = (
        df["salary_cv"] < df["salaryMin_vacancy"]
    ).astype(int)

    return df


def score_to_priority(score):
    """
    Переводит score модели в человекочитаемый приоритет.
    """
    if score >= 0.70:
        return "Высокий приоритет"
    elif score >= 0.40:
        return "Средний приоритет"
    else:
        return "Низкий приоритет"


def make_explanation(row):
    """
    Формирует простое HR-объяснение результата.
    Это не SHAP, а компактное бизнес-объяснение для интерфейса.
    """
    positive_factors = []
    negative_factors = []

    if row.get("pair_text_score_oof", 0) >= 0.6:
        positive_factors.append("текст резюме имеет заметное совпадение с текстом вакансии")
    else:
        negative_factors.append("низкий общий текстовый score резюме и вакансии")

    if row.get("e5_similarity", 0) >= 0.86:
        positive_factors.append("семантическая близость резюме и вакансии высокая")
    else:
        negative_factors.append("семантическая близость резюме и вакансии ниже ожидаемой")

    if row.get("experience_enough", 0) == 1:
        positive_factors.append("опыт кандидата соответствует минимальному требованию")
    else:
        negative_factors.append("опыт кандидата может быть ниже требования вакансии")

    if row.get("salary_match", 0) == 1:
        positive_factors.append("зарплатные ожидания попадают в вилку вакансии")
    else:
        if row.get("salary_above_max", 0) == 1:
            negative_factors.append("зарплатные ожидания выше максимальной вилки вакансии")
        elif row.get("salary_below_min", 0) == 1:
            negative_factors.append("зарплатные ожидания ниже минимальной вилки вакансии")

    explanation = "Факторы, повышающие оценку:\n"

    if positive_factors:
        explanation += "\n".join([f"- {factor}" for factor in positive_factors])
    else:
        explanation += "- выраженных положительных факторов не найдено"

    explanation += "\n\nФакторы, понижающие оценку:\n"

    if negative_factors:
        explanation += "\n".join([f"- {factor}" for factor in negative_factors])
    else:
        explanation += "- выраженных отрицательных факторов не найдено"

    explanation += "\n\nИтоговая рекомендация требует проверки HR-специалистом."

    return explanation


def predict_candidates(input_df, models=None, compute_e5_if_missing=True):
    """
    Основная функция инференса.

    На вход принимает DataFrame с вакансиями и резюме.
    На выход возвращает DataFrame со score, priority, rank и explanation.
    """
    df = input_df.copy()

    check_input_columns(df)

    if models is None:
        models = load_models(load_e5=False)

    final_model = models["final_model"]
    feature_cols = models["feature_cols"]
    pair_text_model = models["pair_text_model"]
    e5_model = models.get("e5_model")

    df = build_pair_text(df)
    df = add_text_lengths(df)
    df = add_pair_text_score(df, pair_text_model)

    if compute_e5_if_missing:
        df = add_e5_similarity(df, e5_model=e5_model)

    df = prepare_numeric_features(df)

    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        raise ValueError(
            "Не хватает признаков для финальной модели: "
            + ", ".join(missing_features)
        )

    X = df[feature_cols].copy()

    df["score"] = final_model.predict_proba(X)[:, 1]
    df["priority"] = df["score"].apply(score_to_priority)

    df = df.sort_values(
        ["idVacancy", "score"],
        ascending=[True, False]
    ).copy()

    df["rank"] = df.groupby("idVacancy").cumcount() + 1

    df["explanation"] = df.apply(make_explanation, axis=1)

    output_cols = [
        "rank",
        "idVacancy",
        "idCv",
        "vacancyName",
        "positionName",
        "score",
        "priority",
        "explanation",
    ]

    optional_cols = [
        "target",
        "e5_similarity",
        "pair_text_score_oof",
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
    ]

    output_cols += [col for col in optional_cols if col in df.columns]

    return df[output_cols].reset_index(drop=True)
