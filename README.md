# HR Candidate Ranking Assistant

Локальная демонстрационная версия NLP-приложения для предварительной оценки и ранжирования кандидатов под вакансии.

## Что делает приложение

Приложение:

- загружает данные по парам «резюме-вакансия»;
- рассчитывает score соответствия кандидата вакансии;
- формирует рейтинг кандидатов внутри вакансии;
- присваивает приоритет обработки;
- показывает объяснение рекомендации;
- позволяет экспортировать результат в CSV.

Система не принимает финальное кадровое решение. Результат предназначен для предварительной приоритизации и должен проверяться HR-специалистом.

## Состав проекта

## Состав проекта

```text
VKR_HR_Assistant_LOCAL_DEMO/
├── app/
│   └── app.py
├── src/
│   └── inference.py
├── data/
│   └── processed_resume_vacancy_matching_e5_tfidf_oof.csv
├── models/
│   ├── final_hybrid_e5_tfidf_xgb_model.joblib
│   ├── final_hybrid_e5_tfidf_features.joblib
│   └── pair_text_tfidf_oof_logreg.joblib
├── reports/
│   ├── final_hybrid_e5_tfidf_results.csv
│   └── inference_test_predictions.csv
├── requirements.txt
├── setup_and_run.bat
└── run_app.bat
```

## Быстрый запуск на Windows через VS Code

1. Распаковать архив.
2. Открыть папку VKR_HR_Assistant_LOCAL_DEMO в VS Code.
3. Открыть терминал в VS Code.
4. Выполнить команды:

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app\app.py

После запуска приложение будет доступно по адресу:

http://localhost:8501

## Самый простой запуск

Можно дважды нажать файл:

setup_and_run.bat

Он создаст виртуальное окружение, установит зависимости и запустит приложение.

## Повторный запуск

Если зависимости уже установлены, можно запускать:

run_app.bat

или через терминал:

.\.venv\Scripts\activate
python -m streamlit run app\app.py

## Используемый подход

Финальная модель использует гибридный подход:

- E5 semantic similarity;
- TF-IDF OOF score;
- matching features;
- XGBoost.

Модель предназначена для предварительного ранжирования кандидатов внутри выбранной вакансии.
