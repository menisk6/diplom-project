from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import inspect, text, create_engine
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "diplom")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

safe_user = quote_plus(DB_USER)
safe_password = quote_plus(DB_PASSWORD)
ENGINE_URL = f"postgresql+psycopg2://{safe_user}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

DDS_SCHEMA = "dds"
DDS_TABLE = "comment_fact"
ODS_SCHEMA = "ods"
ODS_TABLE = "vk_labeled_data_raw"


def get_engine() -> Engine:
    return create_engine(ENGINE_URL, future=True, connect_args={"client_encoding": "utf8"})


def ensure_schema(engine: Engine, schema_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))


def table_exists(engine: Engine, schema: str, table: str) -> bool:
    return inspect(engine).has_table(table_name=table, schema=schema)


def read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    return pd.read_sql_query(text(f'SELECT * FROM "{schema}"."{table}"'), con=engine)


def delete_rows_by_keys(
    engine: Engine,
    schema: str,
    table: str,
    key_col: str,
    keys: list,
    chunk_size: int = 1000,
) -> None:
    if not keys:
        return

    unique_keys = [k for k in pd.Series(keys).dropna().astype(str).unique().tolist() if k]
    if not unique_keys:
        return

    with engine.begin() as conn:
        for i in range(0, len(unique_keys), chunk_size):
            chunk = unique_keys[i : i + chunk_size]
            conn.execute(
                text(f'DELETE FROM "{schema}"."{table}" WHERE "{key_col}" = ANY(:keys)'),
                {"keys": chunk},
            )


def clean_text(value):
    if value is None or pd.isna(value):
        return None
    s = str(value)
    s = html.unescape(s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_label_code(value):
    if value is None or pd.isna(value):
        return None

    raw = str(value).strip().lower()

    mapping = {
        "positive": "pos",
        "pos": "pos",
        "положительный": "pos",
        "положительная": "pos",
        "negative": "neg",
        "neg": "neg",
        "отрицательный": "neg",
        "отрицательная": "neg",
        "neutral": "neu",
        "neautral": "neu",
        "neu": "neu",
        "нейтральный": "neu",
        "нейтральная": "neu",
    }
    if raw in mapping:
        return mapping[raw]

    try:
        num = float(raw.replace(",", "."))
        if num >= 4:
            return "pos"
        if num == 3:
            return "neu"
        if num <= 2:
            return "neg"
    except ValueError:
        pass

    return "unk"


def make_comment_uid(source_id, comment_id) -> str:
    base = f"vk|{source_id}|{comment_id}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def build_dds_dataframe(engine: Engine) -> pd.DataFrame:
    if not table_exists(engine, ODS_SCHEMA, ODS_TABLE):
        raise FileNotFoundError(f"Таблица {ODS_SCHEMA}.{ODS_TABLE} не найдена. Сначала загрузи CSV в ODS.")

    df = read_table(engine, ODS_SCHEMA, ODS_TABLE)
    if df.empty:
        return df

    required = {"comment_id", "comment_text_raw", "comment_datetime", "sentiment", "shop_id", "source_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"В ODS-таблице не хватает колонок: {missing}")

    df = df.copy()

    # базовая очистка
    df["comment_text_raw"] = df["comment_text_raw"].map(clean_text)
    df["comment_text_clean"] = df["comment_text_raw"].map(clean_text)
    df["comment_datetime"] = pd.to_datetime(df["comment_datetime"], errors="coerce")
    df["load_datetime"] = pd.to_datetime(df["load_datetime"], errors="coerce")
    df["is_valid"] = df["is_valid"].fillna(True).astype(bool)

    # если source_id пустой, подставим VK source_id из dds.data_source
    if "source_id" in df.columns and df["source_id"].isna().all():
        source_lookup = read_table(engine, DDS_SCHEMA, "data_source")
        vk_row = source_lookup[source_lookup["source_name"].astype(str).str.lower() == "vk"]
        if vk_row.empty:
            raise ValueError("Не найден source_id для VK в dds.data_source")
        df["source_id"] = vk_row.iloc[0]["source_id"]

    # нормализация тональности
    df["label_raw"] = df["sentiment"].astype("string")
    df["label_code"] = df["label_raw"].map(normalize_label_code)

    labels = read_table(engine, DDS_SCHEMA, "sentiment_label")[["label_id", "label_code"]]
    if labels.empty:
        raise ValueError("dds.sentiment_label пустая. Сначала заполни справочник тональностей.")

    df = df.merge(labels, on="label_code", how="left")
    if df["label_id"].isna().any():
        bad = df[df["label_id"].isna()]["label_raw"].dropna().unique().tolist()
        raise ValueError(f"Не удалось сопоставить sentiment с label_id для значений: {bad}")

    # комментарий-ключ для идемпотентности
    df["source_record_key"] = df["comment_id"].astype("string")
    df["comment_uid"] = df.apply(lambda r: make_comment_uid(r["source_id"], r["comment_id"]), axis=1)

    # поля, которых нет в этом файле
    df["language_text_raw"] = None
    df["category_text_raw"] = None
    df["shop_raw"] = None
    df["source_table"] = ODS_TABLE
    df["source_file"] = df["source_file"].astype("string") if "source_file" in df.columns else f"{ODS_TABLE}.csv"
    df["raw_payload"] = df["raw_payload"] if "raw_payload" in df.columns else None
    df["label_source"] = "manual"
    df["processing_status"] = "labeled"

    out = df[
        [
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
    ].copy()

    out = out.dropna(subset=["comment_uid", "comment_text_raw"]).drop_duplicates(subset=["comment_uid"], keep="last")
    return out


def load_to_dds(engine: Engine, df: pd.DataFrame) -> None:
    if df.empty:
        print("[INFO] Нет строк для загрузки в DDS.")
        return

    if table_exists(engine, DDS_SCHEMA, DDS_TABLE):
        delete_rows_by_keys(
            engine=engine,
            schema=DDS_SCHEMA,
            table=DDS_TABLE,
            key_col="comment_uid",
            keys=df["comment_uid"].tolist(),
        )

    df.to_sql(
        name=DDS_TABLE,
        con=engine,
        schema=DDS_SCHEMA,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print(f"[OK] Загружено в {DDS_SCHEMA}.{DDS_TABLE}: {len(df)} строк")


def main() -> None:
    engine = get_engine()
    ensure_schema(engine, DDS_SCHEMA)

    dds_df = build_dds_dataframe(engine)
    load_to_dds(engine, dds_df)


if __name__ == "__main__":
    main()