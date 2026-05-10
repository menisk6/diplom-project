"""
Загрузка подготовленных датасетов в ODS-слой БД diplom.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table, Text, create_engine, text
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus


# ---------------------------------------------------------------------
# 1. НАСТРОЙКИ ПОДКЛЮЧЕНИЯ К БД
# ---------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "diplom")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
os.environ['PGCLIENTENCODING'] = 'utf-8'

safe_user = quote_plus(DB_USER)
safe_password = quote_plus(DB_PASSWORD)

ENGINE_URL = f"postgresql+psycopg2://{safe_user}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ---------------------------------------------------------------------
# 2. ПУТИ К ДАННЫМ
# ---------------------------------------------------------------------
VK_JSON_PATH = Path("../datasets/vk_comments.json")

# Kaggle/GitHub.
KAGGLE_laytsw_PATH = Path("../datasets/laytsw.csv")
KAGGLE_senylar_PATH = Path("../datasets/senylar.csv")
GITHUB_cleaned_kaspi_reviews_PATH = Path("../datasets/cleaned_kaspi_reviews.csv")


# ---------------------------------------------------------------------
# 3. СОЗДАНИЕ ENGINE
# ---------------------------------------------------------------------
def get_engine() -> Engine:
    return create_engine(ENGINE_URL, future=True, connect_args={'client_encoding': 'utf8'})


# ---------------------------------------------------------------------
# 4. СОЗДАНИЕ SCHEMA И ODS-ТАБЛИЦ
# ---------------------------------------------------------------------
def ensure_schema(engine: Engine, schema_name: str = "ods") -> None:
    """Создаёт schema, если её ещё нет."""
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))


def get_common_columns():
    """Возвращает новый список базовых колонок для каждой таблицы."""
    return [
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("source_record_key", String(255), nullable=True),
        # Column("author_name", Text, nullable=True),
        Column("comment_datetime", DateTime, nullable=True),
        Column("comment_text_raw", Text, nullable=False),
        Column("language_text_raw", String(255), nullable=True),
        Column("category_text_raw", String(255), nullable=True),
        Column("shop_raw", Text, nullable=True),
        Column("label_raw", Text, nullable=True),
        Column("source_file", String(512), nullable=False),
        Column("raw_payload", Text, nullable=True),
        Column("load_datetime", DateTime, nullable=False),
    ]


def build_tables(metadata: MetaData, schema_name: str = "ods") -> Dict[str, Table]:
    """
    Описание ODS-таблиц.

    Все таблицы имеют одинаковую базовую структуру,
    чтобы можно было удобно хранить разные источники данных.
    """

    vk_reviews_raw = Table(
        "vk_reviews_raw",
        metadata,
        *get_common_columns(),
        schema=schema_name,
        extend_existing=True,
    )

    kaggle_laytsw_raw = Table(
        "kaggle_laytsw_raw",
        metadata,
        *get_common_columns(),
        schema=schema_name,
        extend_existing=True,
    )

    kaggle_senylar_raw = Table(
        "kaggle_senylar_raw",
        metadata,
        *get_common_columns(),
        schema=schema_name,
        extend_existing=True,
    )

    github_kaspi_raw = Table(
        "github_kaspi_raw",
        metadata,
        *get_common_columns(),
        schema=schema_name,
        extend_existing=True,
    )

    return {
        "vk_reviews_raw": vk_reviews_raw,
        "kaggle_laytsw_raw": kaggle_laytsw_raw,
        "kaggle_senylar_raw": kaggle_senylar_raw,
        "github_kaspi_raw": github_kaspi_raw
    }


def ensure_tables(engine: Engine) -> None:
    """Создаёт ODS-таблицы, если их ещё нет."""
    metadata = MetaData()
    build_tables(metadata, schema_name="ods")
    metadata.create_all(engine, checkfirst=True)


# ---------------------------------------------------------------------
# 5. ЧТЕНИЕ ИСХОДНЫХ ДАННЫХ
# ---------------------------------------------------------------------
def read_json_any(path: Path) -> pd.DataFrame:
    """
    Читает JSON в одном из форматов:
      - массив объектов JSON: [ {...}, {...} ]
      - JSON Lines: по одной записи в строке
      - один JSON-объект
    """
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return pd.DataFrame()

    # 1) Пытаемся как обычный JSON
    try:
        data = json.loads(content)

        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass

    # 2) Пытаемся как JSON Lines
    try:
        return pd.read_json(path, lines=True, encoding="utf-8")
    except ValueError as exc:
        raise ValueError(f"Не удалось прочитать JSON-файл: {path}") from exc


def read_source_file(path: Path, file_format: str, sep: str = ",") -> pd.DataFrame:
    """
    Универсальное чтение входного файла.
    file_format: 'json', 'csv', 'tsv'
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    fmt = file_format.lower().strip()

    if fmt == "json":
        return read_json_any(path)

    if fmt == "csv":
        data = pd.DataFrame()
        try:
            data = pd.read_csv(path, sep=sep)
        except:
            data = pd.read_csv(path, sep='\t')
        return data

    raise ValueError(f"Неподдерживаемый формат файла: {file_format}")


# ---------------------------------------------------------------------
# 6. ПРЕОБРАЗОВАНИЕ ДАННЫХ В УНИФИЦИРОВАННЫЙ ВИД ODS
# ---------------------------------------------------------------------
def build_ods_dataframe(
    df: pd.DataFrame,
    *,
    source_file: str,
    column_map: Dict[str, str],
    record_key_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Приводит исходный DataFrame к ODS-формату.

    column_map:
        Сопоставление исходных колонок -> ODS-колонок

    record_key_col:
        Имя исходной колонки, по которой можно хранить ключ записи источника.

    label_col:
        Имя исходной колонки с меткой, если она есть.
    """
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)

    # Переносим нужные колонки
    for src_col, dst_col in column_map.items():
        if src_col in df.columns:
            out[dst_col] = df[src_col]
        else:
            out[dst_col] = None

    # Ключ записи источника
    if record_key_col and record_key_col in df.columns:
        out["source_record_key"] = df[record_key_col].astype("string")
    else:
        # Если в источнике нет явного ключа, используем номер строки
        out["source_record_key"] = df.index.astype("string")

    # Метка, если есть
    if label_col and label_col in df.columns:
        out["label_raw"] = df[label_col].astype("string")
    else:
        out["label_raw"] = None

    # Маппинг для заполнения даты
    date_map = {
    ' янв ': '.01.',
    ' фев ': '.02.',
    ' мар ': '.03.',
    ' апр ': '.04.',
    ' мая ': '.05.',
    ' июн ': '.06.',
    ' июл ': '.07.',
    ' авг ': '.08.',
    ' сен ': '.09.',
    ' окт ': '.10.',
    ' ноя ': '.11.',
    ' дек ': '.12.',
    ' в ': ' '
}

    # Дата/время комментария
    if "comment_datetime" in out.columns:
        out["comment_datetime"] = pd.to_datetime(out["comment_datetime"].replace(date_map, regex=True), dayfirst=True, errors="coerce")
    else:
        out["comment_datetime"] = None

    # Текст комментария обязателен для ODS-raw
    if "comment_text_raw" not in out.columns:
        out["comment_text_raw"] = None

    # Язык, если есть
    if "language_text_raw" not in out.columns:
        out["language_text_raw"] = None

    # Категория, если есть
    if 'category_text_raw' not in out.columns:
        out["category_text_raw"] = None

    # Магазин, если есть
    if 'shop_raw' not in out.columns:
        out["shop_raw"] = None

    # Приведение строковых полей
    # if "author_name" in out.columns:
    #     out["author_name"] = out["author_name"].astype("string")
    # else: 
    #     out["author_name"] = None

    out["comment_text_raw"] = out["comment_text_raw"].astype("string")

    # Служебные поля
    out["source_file"] = source_file
    out["load_datetime"] = datetime.now()

    # raw_payload — полная исходная строка как JSON
    out["raw_payload"] = df.apply(
        lambda row: json.dumps(
            row.where(pd.notnull(row), None).to_dict(),
            ensure_ascii=False,
            default=str,
        ),
        axis=1,
    )

    # Убираем строки без текста
    out = out[out["comment_text_raw"].notna() & (out["comment_text_raw"].str.len() > 0)].copy()

    return out


# ---------------------------------------------------------------------
# 7. ЗАГРУЗКА В ODS
# ---------------------------------------------------------------------
def load_dataframe_to_ods(
    engine: Engine,
    table_name: str,
    dataframe: pd.DataFrame,
    schema_name: str = "ods",
) -> int:
    """
    Загружает DataFrame в ODS-таблицу.
    Возвращает количество загруженных строк.
    """
    if dataframe.empty:
        print(f"[INFO] Нет строк для загрузки в {schema_name}.{table_name}")
        return 0

    # Порядок и состав колонок должны совпадать со схемой таблицы
    columns_order = [
        "source_record_key",
        # "author_name",
        "comment_datetime",
        "comment_text_raw",
        "language_text_raw",
        "category_text_raw",
        "shop_raw",
        "label_raw",
        "source_file",
        "raw_payload",
        "load_datetime",
    ]
    dataframe = dataframe[columns_order].copy()

    dataframe.to_sql(
        name=table_name,
        con=engine,
        schema=schema_name,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print(f"[OK] Загружено строк в {schema_name}.{table_name}: {len(dataframe)}")
    return len(dataframe)


# ---------------------------------------------------------------------
# 8. ЗАГРУЗЧИКИ ПО ИСТОЧНИКАМ
# ---------------------------------------------------------------------
def load_vk_reviews(engine: Engine, path: Path) -> int:
    """
    Загружает VK JSON с полями:
        author, date, text

    В ODS попадает в таблицу ods.vk_reviews_raw
    """
    df = read_source_file(path, file_format="json")

    ods_df = build_ods_dataframe(
        df,
        source_file=path.name,
        column_map={
            # "author": "author_name",
            "date": "comment_datetime",
            "text": "comment_text_raw",
            "shop": "shop_raw"
        },
        record_key_col=None,
        label_col=None,
    )

    return load_dataframe_to_ods(engine, "vk_reviews_raw", ods_df)


def load_kaggle_reviews(engine: Engine, path: Path, table_name: str = 'kaggle_reviews_raw', column_map: dict = None, record_key_col: str = None, label_col: str = 'label', file_format: str = "csv") -> int:
    """
    Загружает Kaggle-датасет.
    """
    df = read_source_file(path, file_format=file_format)

    ods_df = build_ods_dataframe(
        df,
        source_file=path.name,
        column_map=column_map,
        record_key_col=record_key_col if record_key_col in df.columns else None,
        label_col=label_col,
    )

    return load_dataframe_to_ods(engine, table_name, ods_df)


def load_github_reviews(engine: Engine, path: Path, table_name: str = "github_reviews_raw", column_map: dict = None, record_key_col: str = None, label_col: str = 'label', file_format: str = "csv") -> int:
    """
    Загружает датасет из GitHub.
    """
    df = read_source_file(path, file_format=file_format)

    ods_df = build_ods_dataframe(
        df,
        source_file=path.name,
        column_map=column_map,
        record_key_col=record_key_col if record_key_col in df.columns else None,
        label_col=label_col,
    )

    return load_dataframe_to_ods(engine, table_name, ods_df)


# ---------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------
def main() -> None:
    engine = get_engine()

    # Создаём schema и таблицы
    ensure_schema(engine, "ods")
    ensure_tables(engine)

    total_loaded = 0
    # VK JSON
    if VK_JSON_PATH.exists():
        total_loaded += load_vk_reviews(engine, VK_JSON_PATH)
    else:
        print(f"[WARN] VK файл не найден: {VK_JSON_PATH}")

    # Kaggle
    if KAGGLE_laytsw_PATH.exists():
        total_loaded += load_kaggle_reviews(engine, 
                                            KAGGLE_laytsw_PATH, 
                                            table_name='kaggle_laytsw_raw', 
                                            column_map={
                                                "review": "comment_text_raw", 
                                                "sentiment": "label_raw"
                                                }, 
                                            label_col = 'sentiment',
                                            file_format="csv")
    else:
        print(f"[WARN] Kaggle файл не найден: {KAGGLE_laytsw_PATH}")

    if KAGGLE_senylar_PATH.exists():
        total_loaded += load_kaggle_reviews(engine, 
                                            KAGGLE_senylar_PATH, 
                                            table_name='kaggle_senylar_raw', 
                                            column_map={
                                                "text": "comment_text_raw", 
                                                "sentiment": "label_raw"
                                                }, 
                                            label_col = 'sentiment',
                                            file_format="csv")
    else:
        print(f"[WARN] Kaggle файл не найден: {KAGGLE_senylar_PATH}")

    # GitHub
    if GITHUB_cleaned_kaspi_reviews_PATH.exists():
        total_loaded += load_github_reviews(engine, 
                                            GITHUB_cleaned_kaspi_reviews_PATH, 
                                            table_name='github_kaspi_raw', 
                                            column_map={
                                                    "combined_text": "comment_text_raw",
                                                    "category": "category_text_raw",
                                                    "language": "language_text_raw",
                                                    "rating": "label_raw"
                                                }, 
                                            label_col = 'rating',
                                            file_format="csv")
    else:
        print(f"[WARN] GitHub файл не найден: {GITHUB_cleaned_kaspi_reviews_PATH}")

    print(f"\n[INFO] Всего загружено строк: {total_loaded}")


if __name__ == "__main__":
    main()