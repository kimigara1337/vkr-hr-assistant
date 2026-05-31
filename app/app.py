
import os
import sys
import html
import base64
import pandas as pd
import streamlit as st
import csv


def prepare_csv_for_excel(df: pd.DataFrame) -> bytes:
    """
    Готовит CSV для корректного открытия в Excel:
    - разделитель ';' для русской локали Excel;
    - UTF-8 с BOM;
    - экранирование всех текстовых полей;
    - переносы строк в explanation заменяются на одинарный текст;
    - защита от формул Excel.
    """
    export_df = df.copy()

    # Оставляем удобный порядок колонок, если они есть
    preferred_cols = [
        "rank",
        "idVacancy",
        "idCv",
        "vacancyName",
        "positionName",
        "score",
        "priority",
        "explanation",
        "target",
        "e5_similarity",
        "pair_text_score_oof",
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
    ]

    existing_cols = [col for col in preferred_cols if col in export_df.columns]
    other_cols = [col for col in export_df.columns if col not in existing_cols]
    export_df = export_df[existing_cols + other_cols]

    # Чистим текстовые колонки, чтобы Excel не устраивал цирк с #ИМЯ?
    text_cols = export_df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        export_df[col] = (
            export_df[col]
            .fillna("")
            .astype(str)
            .str.replace("\r", " ", regex=False)
            .str.replace("\n", " | ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.strip()
        )

        # Защита от строк, которые Excel может принять за формулы
        export_df[col] = export_df[col].apply(
            lambda x: "'" + x if x.startswith(("=", "+", "-", "@")) else x
        )

    csv_text = export_df.to_csv(
        index=False,
        sep=";",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    return csv_text.encode("utf-8-sig")


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.inference import load_models, predict_candidates


# ============================================================
# Настройки страницы
# ============================================================

st.set_page_config(
    page_title="HR Candidate Ranking Assistant",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# Стили
# ============================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 850 !important;
        margin-bottom: 0.4rem !important;
    }

    h2 {
        font-size: 1.65rem !important;
        margin-top: 2rem !important;
        margin-bottom: 0.9rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
        margin-top: 1.4rem !important;
    }

    .subtitle {
        color: rgba(250,250,250,0.72);
        font-size: 1rem;
        line-height: 1.65;
        max-width: 980px;
        margin-bottom: 1.4rem;
    }

    .note {
        color: rgba(250,250,250,0.68);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(180px, 1fr));
        gap: 14px;
        margin: 1.2rem 0 1.8rem 0;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.035));
        border: 1px solid rgba(250,250,250,0.12);
        border-radius: 16px;
        padding: 18px 18px;
    }

    .metric-label {
        color: rgba(250,250,250,0.62);
        font-size: 0.86rem;
        margin-bottom: 6px;
    }

    .metric-value {
        color: white;
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.02em;
    }

    .model-card {
        border: 1px solid rgba(250,250,250,0.13);
        border-radius: 16px;
        padding: 18px 20px;
        background: rgba(255,255,255,0.035);
        margin: 1rem 0 1.2rem 0;
    }

    .model-title {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.78rem;
        font-weight: 800;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.14);
    }

    .badge-green {
        color: #4ade80;
        background: rgba(74,222,128,0.10);
    }

    .badge-yellow {
        color: #facc15;
        background: rgba(250,204,21,0.10);
    }

    .badge-red {
        color: #f87171;
        background: rgba(248,113,113,0.10);
    }

    .badge-blue {
        color: #93c5fd;
        background: rgba(147,197,253,0.10);
    }

    .candidate-card {
        border: 1px solid rgba(250,250,250,0.13);
        border-radius: 16px;
        padding: 18px 20px;
        background: rgba(255,255,255,0.035);
        margin-bottom: 14px;
    }

    .candidate-head {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        align-items: flex-start;
        margin-bottom: 10px;
    }

    .candidate-title {
        font-size: 1.05rem;
        font-weight: 850;
        line-height: 1.35;
    }

    .candidate-subtitle {
        color: rgba(250,250,250,0.62);
        font-size: 0.88rem;
        margin-top: 3px;
    }

    .score-box {
        min-width: 115px;
        text-align: right;
    }

    .score-value {
        font-size: 1.35rem;
        font-weight: 900;
    }

    .score-label {
        color: rgba(250,250,250,0.58);
        font-size: 0.8rem;
    }

    .bar {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
        margin: 10px 0 14px 0;
    }

    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #4ade80);
        border-radius: 999px;
    }

    .candidate-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(150px, 1fr));
        gap: 10px;
        margin: 12px 0;
    }

    .mini-cell {
        border: 1px solid rgba(250,250,250,0.10);
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(0,0,0,0.12);
    }

    .mini-label {
        color: rgba(250,250,250,0.55);
        font-size: 0.76rem;
        margin-bottom: 3px;
    }

    .mini-value {
        color: white;
        font-size: 0.92rem;
        font-weight: 750;
    }

    .explanation-box {
        border-top: 1px solid rgba(250,250,250,0.10);
        margin-top: 12px;
        padding-top: 12px;
        color: rgba(250,250,250,0.78);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .safe-table-wrapper {
        overflow-x: auto;
        border: 1px solid rgba(250,250,250,0.12);
        border-radius: 14px;
        margin: 12px 0 20px 0;
        max-height: 360px;
    }

    table.safe-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
    }

    table.safe-table th {
        background: #20242d;
        color: white;
        padding: 9px 10px;
        border-bottom: 1px solid rgba(250,250,250,0.14);
        text-align: left;
        white-space: nowrap;
    }

    table.safe-table td {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(250,250,250,0.07);
        vertical-align: top;
        max-width: 280px;
        word-break: break-word;
    }

    table.safe-table tr:nth-child(even) {
        background: rgba(255,255,255,0.035);
    }

    .download-link {
        display: inline-block;
        padding: 10px 14px;
        border-radius: 10px;
        border: 1px solid rgba(250,250,250,0.14);
        background: #1f6feb;
        color: white !important;
        text-decoration: none;
        font-weight: 800;
        margin-top: 8px;
    }

    .divider {
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 24px 0;
    }

    [data-testid="stSidebar"] {
        min-width: 260px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Функции
# ============================================================

def esc(value, max_len=180):
    if pd.isna(value):
        return ""
    text = str(value)
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return html.escape(text)


def fmt_num(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.000"


def priority_badge(priority):
    text = str(priority)
    lower = text.lower()

    if "высок" in lower:
        css = "badge-green"
    elif "сред" in lower:
        css = "badge-yellow"
    else:
        css = "badge-red"

    return f'<span class="badge {css}">{html.escape(text)}</span>'


def static_table(df, columns=None, max_rows=20):
    if df is None or df.empty:
        return "<p class='note'>Нет данных для отображения.</p>"

    temp = df.copy()

    if columns is not None:
        existing_cols = [col for col in columns if col in temp.columns]
        temp = temp[existing_cols]

    temp = temp.head(max_rows).copy()

    for col in temp.columns:
        temp[col] = temp[col].apply(lambda x: esc(x, max_len=120))

    table = temp.to_html(
        index=False,
        escape=False,
        classes="safe-table",
        border=0
    )

    return f"<div class='safe-table-wrapper'>{table}</div>"


def prepare_csv_for_excel(df: pd.DataFrame) -> bytes:
    """
    Подготавливает CSV для нормального открытия в Excel:
    - UTF-8 с BOM;
    - разделитель ';' для русской локали;
    - все поля берутся в кавычки;
    - переносы строк внутри explanation убираются;
    - строки, похожие на формулы Excel, защищаются.
    """
    export_df = df.copy()

    preferred_cols = [
        "rank",
        "idVacancy",
        "idCv",
        "vacancyName",
        "positionName",
        "score",
        "priority",
        "explanation",
        "target",
        "e5_similarity",
        "pair_text_score_oof",
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
    ]

    existing_cols = [col for col in preferred_cols if col in export_df.columns]
    other_cols = [col for col in export_df.columns if col not in existing_cols]
    export_df = export_df[existing_cols + other_cols]

    text_cols = export_df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        export_df[col] = (
            export_df[col]
            .fillna("")
            .astype(str)
            .str.replace("\r", " ", regex=False)
            .str.replace("\n", " | ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.strip()
        )

        export_df[col] = export_df[col].apply(
            lambda x: "'" + x if x.startswith(("=", "+", "-", "@")) else x
        )

    csv_text = export_df.to_csv(
        index=False,
        sep=";",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    return csv_text.encode("utf-8-sig")


def download_link(df, filename, label):
    csv_bytes = prepare_csv_for_excel(df)
    b64 = base64.b64encode(csv_bytes).decode()

    return (
        f'<a class="download-link" '
        f'href="data:text/csv;base64,{b64}" '
        f'download="{html.escape(filename)}">{html.escape(label)}</a>'
    )


@st.cache_resource
def cached_models():
    return load_models()


@st.cache_data
def cached_demo_data():
    paths = [
        f"{PROJECT_DIR}/data/processed_resume_vacancy_matching_e5_tfidf_oof.csv",
        f"{PROJECT_DIR}/data/processed_resume_vacancy_matching_e5.csv",
        f"{PROJECT_DIR}/data/sample_new_input.csv",
    ]

    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path), path

    raise FileNotFoundError("Не найден демонстрационный CSV.")


@st.cache_data(show_spinner=True)
def cached_predictions(df):
    models = cached_models()
    return predict_candidates(df, models=models)


def candidate_card(row):
    rank = esc(row.get("rank", ""))
    position = esc(row.get("positionName", ""))
    vacancy = esc(row.get("vacancyName", ""))
    candidate_id = esc(row.get("idCv", ""), max_len=80)

    score = float(row.get("score", 0) or 0)
    score_pct = max(0, min(100, score * 100))

    priority = row.get("priority", "")
    target = esc(row.get("target", ""))

    e5 = fmt_num(row.get("e5_similarity", 0))
    tfidf_oof = fmt_num(row.get("pair_text_score_oof", 0))
    exp = esc(row.get("experience", ""))
    exp_req = esc(row.get("experienceRequirements", ""))
    salary = esc(row.get("salary_cv", ""))
    sal_min = esc(row.get("salaryMin_vacancy", ""))
    sal_max = esc(row.get("salaryMax_vacancy", ""))

    explanation = esc(row.get("explanation", "Объяснение отсутствует."), max_len=1000)

    return f"""
    <div class="candidate-card">
        <div class="candidate-head">
            <div>
                <div class="candidate-title">#{rank} — {position}</div>
                <div class="candidate-subtitle">Вакансия: {vacancy}</div>
                <div class="candidate-subtitle">idCv: <code>{candidate_id}</code></div>
            </div>
            <div class="score-box">
                <div class="score-value">{score:.3f}</div>
                <div class="score-label">score</div>
            </div>
        </div>

        <div class="bar">
            <div class="bar-fill" style="width:{score_pct:.1f}%"></div>
        </div>

        <div>
            {priority_badge(priority)}
            <span class="badge badge-blue">target: {target}</span>
        </div>

        <div class="candidate-grid">
            <div class="mini-cell">
                <div class="mini-label">E5 similarity</div>
                <div class="mini-value">{e5}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">TF-IDF OOF score</div>
                <div class="mini-value">{tfidf_oof}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">Опыт кандидата</div>
                <div class="mini-value">{exp}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">Требуемый опыт</div>
                <div class="mini-value">{exp_req}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">Зарплата кандидата</div>
                <div class="mini-value">{salary}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">Минимум вакансии</div>
                <div class="mini-value">{sal_min}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">Максимум вакансии</div>
                <div class="mini-value">{sal_max}</div>
            </div>
            <div class="mini-cell">
                <div class="mini-label">Решение</div>
                <div class="mini-value">проверка HR</div>
            </div>
        </div>

        <div class="explanation-box">
            <b>Объяснение рекомендации:</b><br>
            {explanation}
        </div>
    </div>
    """


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown("## Режим работы")
mode = st.sidebar.radio(
    "Выберите источник данных",
    ["Demo CSV", "Upload CSV"],
    index=0
)

st.sidebar.success("Модели загружены")

with st.sidebar.expander("Информация о модели"):
    st.markdown("""
    **Финальная модель:**  
    E5 semantic similarity + TF-IDF OOF score + matching features + XGBoost.

    **Назначение:**  
    предварительное ранжирование кандидатов внутри вакансии.

    **Ограничение:**  
    результат требует проверки HR-специалистом.
    """)


# ============================================================
# Header
# ============================================================

st.markdown("# HR Candidate Ranking Assistant")

st.markdown("""
<div class="subtitle">
MVP NLP-приложения для предварительной оценки соответствия кандидатов вакансиям.
Система рассчитывает <b>score соответствия</b>, формирует рейтинг кандидатов внутри выбранной вакансии,
присваивает приоритет обработки и показывает объяснение рекомендации.
<br><br>
<b>Важно:</b> система не принимает финальное кадровое решение. Результат предназначен для предварительной
приоритизации и должен проверяться HR-специалистом.
</div>
""", unsafe_allow_html=True)


# ============================================================
# Данные
# ============================================================

try:
    if mode == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Загрузите CSV", type=["csv"])
        if uploaded_file is None:
            st.info("Загрузите CSV или выберите Demo CSV.")
            st.stop()
        df_input = pd.read_csv(uploaded_file)
        input_path = "uploaded file"
    else:
        df_input, input_path = cached_demo_data()
        st.sidebar.success("Demo CSV загружен")

    with st.spinner("Считаем рекомендации..."):
        predictions = cached_predictions(df_input)

except Exception as e:
    st.error("Ошибка при загрузке данных или расчете рекомендаций.")
    st.exception(e)
    st.stop()


# ============================================================
# Метрики
# ============================================================

total_rows = len(predictions)
total_input = len(df_input)
total_vacancies = predictions["idVacancy"].nunique() if "idVacancy" in predictions.columns else 0
total_candidates = predictions["idCv"].nunique() if "idCv" in predictions.columns else 0
avg_score = predictions["score"].mean() if "score" in predictions.columns else 0

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Входных пар резюме-вакансия</div>
        <div class="metric-value">{total_input:,}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Уникальных вакансий</div>
        <div class="metric-value">{total_vacancies:,}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Уникальных кандидатов</div>
        <div class="metric-value">{total_candidates:,}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Средний score</div>
        <div class="metric-value">{avg_score:.3f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="model-card">
    <div class="model-title">Используемый подход</div>
    <span class="badge badge-blue">multilingual E5 embeddings</span>
    <span class="badge badge-blue">TF-IDF OOF score</span>
    <span class="badge badge-blue">matching features</span>
    <span class="badge badge-blue">XGBoost</span>
    <div class="note">
        Модель объединяет семантическую близость резюме и вакансии, текстовый score на основе TF-IDF,
        числовые признаки опыта, зарплаты и длины описаний. Итоговый score используется для ранжирования кандидатов.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Входные данные
# ============================================================

with st.expander("Показать пример входных данных", expanded=False):
    short_input_cols = [
        "idVacancy",
        "idCv",
        "vacancyName",
        "positionName",
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
        "e5_similarity",
        "pair_text_score_oof",
        "target",
    ]

    st.markdown(
        f"<p class='note'>Источник: <code>{html.escape(str(input_path))}</code></p>",
        unsafe_allow_html=True
    )
    st.markdown(static_table(df_input, short_input_cols, max_rows=8), unsafe_allow_html=True)


# ============================================================
# Распределение приоритетов
# ============================================================

st.markdown("## Результаты ранжирования")

if "priority" in predictions.columns:
    priority_counts = predictions["priority"].value_counts()

    high = int(priority_counts.get("Высокий приоритет", 0))
    mid = int(priority_counts.get("Средний приоритет", 0))
    low = int(priority_counts.get("Низкий приоритет", 0))

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Высокий приоритет</div>
            <div class="metric-value">{high:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Средний приоритет</div>
            <div class="metric-value">{mid:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Низкий приоритет</div>
            <div class="metric-value">{low:,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Всего рекомендаций</div>
            <div class="metric-value">{len(predictions):,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Выбор вакансии
# ============================================================

st.markdown("## Подбор кандидатов под вакансию")

vacancy_df = (
    predictions[["idVacancy", "vacancyName"]]
    .drop_duplicates()
    .sort_values("vacancyName")
    .reset_index(drop=True)
)

search_query = st.text_input(
    "Поиск вакансии",
    value="",
    placeholder="Например: бухгалтер, экономист, программист..."
)

filtered_vacancy_df = vacancy_df.copy()

if search_query.strip():
    q = search_query.strip().lower()
    filtered_vacancy_df = filtered_vacancy_df[
        filtered_vacancy_df["vacancyName"].fillna("").str.lower().str.contains(q, regex=False)
    ]

if filtered_vacancy_df.empty:
    st.warning("По такому запросу вакансии не найдены.")
    st.stop()

filtered_vacancy_df = filtered_vacancy_df.head(150).copy()

vacancy_options = (
    filtered_vacancy_df["vacancyName"].astype(str)
    + " | "
    + filtered_vacancy_df["idVacancy"].astype(str)
).tolist()

selected_label = st.selectbox(
    "Выберите вакансию",
    vacancy_options,
    index=0
)

selected_vacancy = selected_label.split(" | ")[-1]

top_k = st.select_slider(
    "Сколько кандидатов показать",
    options=[3, 5, 10, 15, 20],
    value=5
)


# ============================================================
# Топ кандидатов
# ============================================================

selected_results = predictions[
    predictions["idVacancy"].astype(str) == str(selected_vacancy)
].copy()

selected_results = selected_results.sort_values("score", ascending=False).head(top_k).copy()

st.markdown("### Топ кандидатов")

if selected_results.empty:
    st.info("Для выбранной вакансии нет кандидатов.")
else:
    for _, row in selected_results.iterrows():
        
        clean_card_html = "\n".join(
            line.strip() for line in candidate_card(row).splitlines()
            if line.strip()
        )
        st.markdown(clean_card_html, unsafe_allow_html=True)


# ============================================================
# Короткая таблица
# ============================================================

with st.expander("Показать компактную таблицу результата", expanded=False):
    result_cols = [
        "rank",
        "idCv",
        "vacancyName",
        "positionName",
        "score",
        "priority",
        "target",
        "e5_similarity",
        "pair_text_score_oof",
    ]
    st.markdown(static_table(selected_results, result_cols, max_rows=top_k), unsafe_allow_html=True)


# =========================================================
# Экспорт
# =========================================================

st.markdown("## Экспорт результата")

try:
    export_cols = [
        "rank",
        "idVacancy",
        "idCv",
        "vacancyName",
        "positionName",
        "score",
        "priority",
        "explanation",
        "target",
        "e5_similarity",
        "pair_text_score_oof",
        "experience",
        "experienceRequirements",
        "salary_cv",
        "salaryMin_vacancy",
        "salaryMax_vacancy",
    ]

    # Берём только те колонки, которые реально есть в selected_results
    export_cols = [col for col in export_cols if col in selected_results.columns]

    export_df = selected_results[export_cols].copy()

    # Переименовываем колонки для нормального человеческого вида
    export_df = export_df.rename(columns={
        "rank": "Место",
        "idVacancy": "ID вакансии",
        "idCv": "ID кандидата",
        "vacancyName": "Вакансия",
        "positionName": "Кандидат / должность",
        "score": "Итоговый score",
        "priority": "Приоритет",
        "explanation": "Объяснение рекомендации",
        "target": "Истинная метка",
        "e5_similarity": "Семантическая близость E5",
        "pair_text_score_oof": "Текстовый TF-IDF score",
        "experience": "Опыт кандидата",
        "experienceRequirements": "Требуемый опыт",
        "salary_cv": "Ожидания кандидата",
        "salaryMin_vacancy": "Минимум вакансии",
        "salaryMax_vacancy": "Максимум вакансии",
    })

    # Округляем числовые признаки, чтобы CSV не выглядел как выброс из лаборатории
    numeric_cols = [
        "Итоговый score",
        "Семантическая близость E5",
        "Текстовый TF-IDF score",
        "Опыт кандидата",
        "Требуемый опыт",
        "Ожидания кандидата",
        "Минимум вакансии",
        "Максимум вакансии",
    ]

    for col in numeric_cols:
        if col in export_df.columns:
            export_df[col] = pd.to_numeric(export_df[col], errors="coerce").round(3)

    # Убираем переносы строк из объяснения, чтобы Excel не превращал файл в кашу
    if "Объяснение рекомендации" in export_df.columns:
        export_df["Объяснение рекомендации"] = (
            export_df["Объяснение рекомендации"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
            .str.replace("  ", " ", regex=False)
        )

    # CSV для Excel: utf-8-sig + разделитель ;
    csv_bytes = export_df.to_csv(
        index=False,
        sep=";",
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="Скачать рекомендации CSV",
        data=csv_bytes,
        file_name="selected_vacancy_recommendations.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error("Ошибка при подготовке файла экспорта.")
    st.exception(e)

st.markdown("""
<div class="note">
<br>
Демонстрационный интерфейс показывает результат модели в человекочитаемом виде:
выбор вакансии, топ кандидатов, score, приоритет и объяснение факторов.
</div>
""", unsafe_allow_html=True)
