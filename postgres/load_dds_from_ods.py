from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dds_schema import ensure_schema, ensure_tables, get_engine


# ---------------------------------------------------------------------
# Настройки ODS-источников
# ---------------------------------------------------------------------
ODS_TABLES_META = {
    "vk_reviews_raw": {
        "source_name": "vk",
        "source_type": "social_network",
        "description": "VK comments/reviews dataset",
    },
    "kaggle_laytsw_raw": {
        "source_name": "kaggle_laytsw",
        "source_type": "open_dataset",
        "description": "Kaggle dataset: laytsw",
    },
    "kaggle_senylar_raw": {
        "source_name": "kaggle_senylar",
        "source_type": "open_dataset",
        "description": "Kaggle dataset: senylar",
    },
    "github_kaspi_raw": {
        "source_name": "github_kaspi",
        "source_type": "git_repository",
        "description": "GitHub dataset: cleaned kaspi reviews",
    },
}

SOURCE_TABLE_ORDER = list(ODS_TABLES_META.keys())


# ---------------------------------------------------------------------
# Утилиты нормализации
# ---------------------------------------------------------------------
def normalize_key(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or None


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value)
    s = html.unescape(s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    s = series.astype("string").str.strip()
    s = s.replace(
        {
            " янв ": ".01.",
            " фев ": ".02.",
            " мар ": ".03.",
            " апр ": ".04.",
            " мая ": ".05.",
            " июн ": ".06.",
            " июл ": ".07.",
            " авг ": ".08.",
            " сен ": ".09.",
            " окт ": ".10.",
            " ноя ": ".11.",
            " дек ": ".12.",
            " в ": " ",
        },
        regex=True,
    )
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def normalize_label(value: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Возвращает:
        label_name, label_code, label_description
    """
    if value is None or pd.isna(value):
        return None, None, None

    raw = str(value).strip().lower()

    # Текстовые метки
    text_map = {
        "positive": ("positive", "pos", "Положительная тональность"),
        "pos": ("positive", "pos", "Положительная тональность"),
        "положительный": ("positive", "pos", "Положительная тональность"),
        "положительная": ("positive", "pos", "Положительная тональность"),
        "negative": ("negative", "neg", "Отрицательная тональность"),
        "neg": ("negative", "neg", "Отрицательная тональность"),
        "отрицательный": ("negative", "neg", "Отрицательная тональность"),
        "отрицательная": ("negative", "neg", "Отрицательная тональность"),
        "neutral": ("neutral", "neu", "Нейтральная тональность"),
        "neautral": ("neutral", "neu", "Нейтральная тональность"),
        "neu": ("neutral", "neu", "Нейтральная тональность"),
        "нейтральный": ("neutral", "neu", "Нейтральная тональность"),
        "нейтральная": ("neutral", "neu", "Нейтральная тональность"),
    }
    if raw in text_map:
        return text_map[raw]

    # Числовые оценки / рейтинги
    try:
        num = float(raw.replace(",", "."))
        if num >= 4:
            return "positive", "pos", "Положительная тональность"
        if num == 3:
            return "neutral", "neu", "Нейтральная тональность"
        if num <= 2:
            return "negative", "neg", "Отрицательная тональность"
    except ValueError:
        pass

    # Фолбэк
    return "unknown", "unk", "Не удалось нормализовать исходную метку"


def make_comment_uid(source_name: str, source_record_key: Optional[str], text_raw: Optional[str], dt_value) -> str:
    base = "|".join(
        [
            str(source_name or ""),
            str(source_record_key or ""),
            str(text_raw or ""),
            str(dt_value if pd.notna(dt_value) else ""),
        ]
    )
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Чтение ODS
# ---------------------------------------------------------------------
def read_ods_table(engine: Engine, table_name: str) -> pd.DataFrame:
    query = text(f"""SELECT * FROM "ods"."{table_name}" WHERE language_text_raw is null or language_text_raw = 'russian'""")
    return pd.read_sql_query(query, con=engine)


def load_all_ods(engine: Engine) -> pd.DataFrame:
    frames = []
    for table_name in SOURCE_TABLE_ORDER:
        if not engine.dialect.has_table(engine.connect(), table_name, schema="ods"):
            print(f"[WARN] Таблица ods.{table_name} не найдена, пропуск.")
            continue

        meta = ODS_TABLES_META[table_name]
        df = read_ods_table(engine, table_name)
        if df.empty:
            continue

        # Приведение к единому виду
        unified = pd.DataFrame()

        unified["source_record_key"] = df["source_record_key"] if "source_record_key" in df.columns else None
        # unified["author_name"] = df["author_name"] if "author_name" in df.columns else None
        unified["shop_raw"] = df["shop_raw"] if "shop_raw" in df.columns else None
        unified["comment_text_raw"] = df["comment_text_raw"] if "comment_text_raw" in df.columns else None
        unified["comment_datetime"] = parse_datetime_series(df["comment_datetime"]) if "comment_datetime" in df.columns else pd.NaT
        unified["label_raw"] = df["label_raw"] if "label_raw" in df.columns else None
        unified["category_text_raw"] = df["category_text_raw"] if "category_text_raw" in df.columns else None
        unified["source_file"] = df["source_file"] if "source_file" in df.columns else table_name
        unified["load_datetime"] = pd.to_datetime(df["load_datetime"], errors="coerce") if "load_datetime" in df.columns else datetime.now()

        unified["source_table"] = table_name
        unified["source_name"] = meta["source_name"]
        unified["source_type"] = meta["source_type"]
        unified["source_description"] = meta["description"]

        frames.append(unified)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Вспомогательные вставки в DDS
# ---------------------------------------------------------------------
def _read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    return pd.read_sql_query(text(f'SELECT * FROM "{schema}"."{table}"'), con=engine)


def insert_missing_rows(
    engine: Engine,
    schema: str,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Вставляет только отсутствующие строки по natural key.
    Возвращает актуальное содержимое таблицы.
    """
    if df.empty:
        return _read_table(engine, schema, table)

    df = df.dropna(subset=key_cols).drop_duplicates(subset=key_cols).copy()

    current = _read_table(engine, schema, table)
    if current.empty:
        new_rows = df
    else:
        current_keys = current[key_cols].drop_duplicates()
        merged = df.merge(current_keys, on=key_cols, how="left", indicator=True)
        new_rows = merged[merged["_merge"] == "left_only"][df.columns].copy()

    if not new_rows.empty:
        new_rows.to_sql(
            name=table,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    return _read_table(engine, schema, table)


# ---------------------------------------------------------------------
# Подготовка справочников
# ---------------------------------------------------------------------
def build_data_source_dim(staging: pd.DataFrame) -> pd.DataFrame:
    if staging.empty:
        return pd.DataFrame(columns=["source_name", "source_type", "description", "created_at"])

    src = (
        staging[["source_name", "source_type", "source_description"]]
        .drop_duplicates()
        .rename(columns={"source_description": "description"})
        .copy()
    )
    src["created_at"] = datetime.now()
    return src


# def build_author_dim(staging: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
#     rows = staging[["source_name", "author_name"]].copy()
#     rows["source_author_key"] = rows["author_name"].map(normalize_key)
#     rows = rows.dropna(subset=["source_author_key"]).drop_duplicates(subset=["source_name", "source_author_key"])

#     dim = rows.merge(source_map[["source_id", "source_name"]], on="source_name", how="left")
#     dim["created_at"] = datetime.now()
#     return dim[["author_name", "source_author_key", "source_id", "created_at"]]


def build_shop_dim(staging: pd.DataFrame, source_map: pd.DataFrame) -> pd.DataFrame:
    rows = staging[["source_name", "shop_raw"]].copy()
    rows["shop_name"] = rows["shop_raw"]
    rows["shop_source_key"] = rows["shop_name"].map(normalize_key)
    rows = rows.dropna(subset=["shop_source_key"]).drop_duplicates(subset=["source_name", "shop_source_key"])

    dim = rows.merge(source_map[["source_id", "source_name"]], on="source_name", how="left")
    dim["shop_url"] = None
    dim["created_at"] = datetime.now()
    return dim[["shop_name", "shop_source_key", "shop_url", "source_id", "created_at"]]


def build_label_dim() -> pd.DataFrame:
    rows = [
        {"label_name": "positive", "label_code": "pos", "label_description": "Положительная тональность"},
        {"label_name": "neutral", "label_code": "neu", "label_description": "Нейтральная тональность"},
        {"label_name": "negative", "label_code": "neg", "label_description": "Отрицательная тональность"},
        {"label_name": "unknown", "label_code": "unk", "label_description": "Не удалось нормализовать исходную метку"},
    ]
    df = pd.DataFrame(rows)
    df["created_at"] = datetime.now()
    return df


# ---------------------------------------------------------------------
# Подготовка факта
# ---------------------------------------------------------------------
def build_comment_fact(
    staging: pd.DataFrame,
    source_map: pd.DataFrame,
    # author_map: pd.DataFrame,
    shop_map: pd.DataFrame,
    label_map: pd.DataFrame,
) -> pd.DataFrame:
    if staging.empty:
        return pd.DataFrame()

    fact = staging.copy()

    # fact["author_name"] = fact["author_name"].map(clean_text)
    fact["shop_raw"] = fact["shop_raw"].map(clean_text)
    fact["label_raw"] = fact["label_raw"].map(lambda x: None if pd.isna(x) else str(x).strip())

    # Нормализация дат
    fact["comment_datetime"] = pd.to_datetime(fact["comment_datetime"], errors="coerce", dayfirst=False)
    fact["load_datetime"] = pd.to_datetime(fact["load_datetime"], errors="coerce")

    # Нормализуем метки
    normalized = fact["label_raw"].apply(normalize_label)
    fact["label_name"] = normalized.apply(lambda x: x[0])
    fact["label_code"] = normalized.apply(lambda x: x[1])

    # Убираем пустые тексты
    fact = fact[fact["comment_text_raw"].notna() & (fact["comment_text_raw"].str.len() > 0)].copy()

    # IDs
    fact = fact.merge(source_map[["source_id", "source_name"]], on="source_name", how="left")

    # author_key = fact["author_name"].map(normalize_key)
    # fact["source_author_key"] = author_key
    # fact = fact.merge(
    #     author_map[["author_id", "source_id", "source_author_key"]],
    #     on=["source_id", "source_author_key"],
    #     how="left",
    # )

    shop_key = fact["shop_raw"].map(normalize_key)
    fact["shop_source_key"] = shop_key
    fact = fact.merge(
        shop_map[["shop_id", "source_id", "shop_source_key"]],
        on=["source_id", "shop_source_key"],
        how="left",
    )

    fact = fact.merge(
        label_map[["label_id", "label_code"]],
        on="label_code",
        how="left",
    )

    # comment_uid для идемпотентности
    fact["comment_uid"] = fact.apply(
        lambda r: make_comment_uid(
            r["source_name"],
            r.get("source_record_key"),
            r.get("comment_text_raw"),
            r.get("comment_datetime"),
        ),
        axis=1,
    )

    fact["is_valid"] = True

    columns = [
        "comment_uid",
        "source_id",
        "source_record_key",
        # "author_id",
        "shop_id",
        "label_id",
        "category_text_raw",
        "comment_text_raw",
        "comment_datetime",
        "load_datetime",
        "is_valid",
    ]
    return fact[columns].copy()


# ---------------------------------------------------------------------
# Запись в DDS
# ---------------------------------------------------------------------
def load_dds(engine: Engine) -> None:
    staging = load_all_ods(engine)
    if staging.empty:
        print("[INFO] В ODS нет данных для загрузки в DDS.")
        return

    # 1) data_source
    data_source_df = build_data_source_dim(staging)
    data_source_df = insert_missing_rows(
        engine,
        schema="dds",
        table="data_source",
        df=data_source_df,
        key_cols=["source_name"],
    )
    source_map = data_source_df if "source_id" in data_source_df.columns else _read_table(engine, "dds", "data_source")
    print("data_source загружен")

    # 2) sentiment_label
    label_df = build_label_dim()
    label_df = insert_missing_rows(
        engine,
        schema="dds",
        table="sentiment_label",
        df=label_df,
        key_cols=["label_code"],
    )
    label_map = _read_table(engine, "dds", "sentiment_label")
    print("sentiment_label загружен")

    # 3) author
    # author_df = build_author_dim(staging, source_map)
    # author_df = insert_missing_rows(
    #     engine,
    #     schema="dds",
    #     table="author",
    #     df=author_df,
    #     key_cols=["source_id", "source_author_key"],
    # )
    # author_map = _read_table(engine, "dds", "author")
    # print("author загружен")

    # 4) shop
    shop_df = build_shop_dim(staging, source_map)
    shop_df = insert_missing_rows(
        engine,
        schema="dds",
        table="shop",
        df=shop_df,
        key_cols=["source_id", "shop_source_key"],
    )
    shop_map = _read_table(engine, "dds", "shop")
    print("shop загружен")

    # 5) fact
    fact_df = build_comment_fact(staging, source_map, shop_map, label_map)
    if fact_df.empty:
        print("[INFO] Нет строк для загрузки в DDS.fact.")
        return
    # Идемпотентная загрузка по comment_uid
    existing_fact = _read_table(engine, "dds", "comment_fact")
    if not existing_fact.empty and "comment_uid" in existing_fact.columns:
        fact_df = fact_df.merge(existing_fact[["comment_uid"]], on="comment_uid", how="left", indicator=True)
        fact_df = fact_df[fact_df["_merge"] == "left_only"].drop(columns=["_merge"]).copy()

    if fact_df.empty:
        print("[INFO] Все записи уже есть в dds.comment_fact.")
        return

    fact_df.to_sql(
        name="comment_fact",
        con=engine,
        schema="dds",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print("comment_fact загружен")

    print(f"[OK] DDS загружен. Добавлено записей в comment_fact: {len(fact_df)}")


def main() -> None:
    engine = get_engine()
    ensure_schema(engine, "dds")
    ensure_tables(engine, "dds")
    load_dds(engine)


if __name__ == "__main__":
    main()