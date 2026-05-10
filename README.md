# Sentiment model service

Локальный сервис для:
- регистрации моделей и версий;
- выбора активной версии;
- предсказания тональности по одному отзыву;
- пакетной проверки CSV;
- простого веб-интерфейса;
- работы через API.

## Запуск

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Откройте:
- `http://127.0.0.1:8000/` — интерфейс;
- `http://127.0.0.1:8000/docs` — Swagger API.

## Примеры моделей
Подходящие sentiment-модели для русского языка:
- `cointegrated/rubert-tiny2` — лёгкая базовая модель;
- `blanchefort/rubert-base-cased-sentiment` — более тяжёлая;

## Как использовать
1. Зарегистрируйте модель через UI или `POST /api/v1/models/register`.
2. Активируйте нужную версию через `POST /api/v1/models/{family}/activate/{version}`.
3. Отправьте текст на `POST /api/v1/predict`.

## Версионирование
Каждая регистрация создаёт новую версию внутри family.
Например:
- family: `base-rubert`
- source: `cointegrated/rubert-tiny2`
- version: `1`
Потом можно зарегистрировать дообученную локальную модель в той же family как version `2`.
