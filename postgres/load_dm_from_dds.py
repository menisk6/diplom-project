from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from dm_schema import ensure_schema, ensure_tables, get_engine


# ---------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "diplom")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


# ---------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------
def normalize_key(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or None


def ns_key(source_name: str, key: str | None) -> str | None:
    if key is None or pd.isna(key):
        return None
    return f"{source_name.strip().lower()}|{str(key).strip().lower()}"


def clean_text(value):
    if value is None or pd.isna(value):
        return None
    s = str(value)
    s = html.unescape(s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    return pd.read_sql_query(text(f'SELECT * FROM "{schema}"."{table}"'), con=engine)


def has_table(engine: Engine, schema: str, table: str) -> bool:
    return inspect(engine).has_table(table_name=table, schema=schema)


def insert_missing_rows(
    engine: Engine,
    schema: str,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Вставляет в таблицу только новые строки по natural key.
    Возвращает актуальное содержимое таблицы после вставки.
    """
    if df.empty:
        return read_table(engine, schema, table)

    df = df.dropna(subset=key_cols).drop_duplicates(subset=key_cols).copy()

    current = read_table(engine, schema, table)
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

    return read_table(engine, schema, table)


def make_fact_key(*parts) -> str:
    base = "|".join("" if p is None or pd.isna(p) else str(p) for p in parts)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def filter_new_fact_rows(
    engine: Engine,
    schema: str,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """
    Оставляет только те строки, которых ещё нет в целевой fact-таблице.
    """
    if df.empty:
        return df

    df = df.dropna(subset=["comment_text_raw"]).copy()
    df["__fact_key__"] = df.apply(lambda r: make_fact_key(*[r.get(c) for c in key_cols]), axis=1)
    df = df.drop_duplicates(subset=["__fact_key__"])

    existing = read_table(engine, schema, table)
    if existing.empty:
        return df.drop(columns=["__fact_key__"])

    existing["__fact_key__"] = existing.apply(lambda r: make_fact_key(*[r.get(c) for c in key_cols]), axis=1)
    merged = df.merge(existing[["__fact_key__"]], on="__fact_key__", how="left", indicator=True)
    out = merged[merged["_merge"] == "left_only"].drop(columns=["_merge", "__fact_key__"]).copy()
    return out


# ---------------------------------------------------------------------
# Загрузка справочников DM из DDS
# ---------------------------------------------------------------------
def load_data_source_dim(engine: Engine) -> pd.DataFrame:
    dds = read_table(engine, "dds", "data_source")
    if dds.empty:
        return dds

    dm = dds[["source_name", "source_type", "description", "created_at"]].copy()
    dm = dm.drop_duplicates(subset=["source_name"]).copy()
    insert_missing_rows(engine, "dm", "data_source", dm, ["source_name"])
    return read_table(engine, "dm", "data_source")


def load_sentiment_label_dim(engine: Engine) -> pd.DataFrame:
    dds = read_table(engine, "dds", "sentiment_label")
    if dds.empty:
        return dds

    dm = dds[["label_name", "label_code", "label_description"]].copy()
    if "created_at" not in dds.columns:
        dm["created_at"] = datetime.now()
    else:
        dm["created_at"] = datetime.now()

    dm = dm.drop_duplicates(subset=["label_code"]).copy()
    insert_missing_rows(engine, "dm", "sentiment_label", dm, ["label_code"])
    return read_table(engine, "dm", "sentiment_label")


# def load_author_dim(engine: Engine) -> pd.DataFrame:
#     dds_author = read_table(engine, "dds", "author")
#     dds_source = read_table(engine, "dds", "data_source")
#     if dds_author.empty or dds_source.empty:
#         return pd.DataFrame()

#     df = dds_author.merge(dds_source[["source_id", "source_name"]], on="source_id", how="left")
#     df["author_name"] = df["author_name"].map(clean_text)
#     # df["source_author_key"] = df.apply(lambda r: ns_key(r["source_name"], r["source_author_key"]), axis=1)
#     dm = df[["author_name", "created_at"]].copy()

#     dm = dm.dropna(subset=["author_name"]).drop_duplicates(subset=["author_name"]).copy()
#     insert_missing_rows(engine, "dm", "author", dm, ["author_name"])
#     return read_table(engine, "dm", "author")


def load_shop_dim(engine: Engine) -> pd.DataFrame:
    dds_shop = read_table(engine, "dds", "shop")
    dds_source = read_table(engine, "dds", "data_source")
    if dds_shop.empty or dds_source.empty:
        return pd.DataFrame()

    df = dds_shop.merge(dds_source[["source_id", "source_name"]], on="source_id", how="left")
    df["shop_name"] = df["shop_name"].map(clean_text)
    # df["shop_source_key"] = df.apply(lambda r: ns_key(r["source_name"], r["shop_source_key"]), axis=1)
    dm = df[["shop_name", "shop_url", "created_at"]].copy()

    dm = dm.dropna(subset=["shop_name"]).drop_duplicates(subset=["shop_name"]).copy()
    insert_missing_rows(engine, "dm", "shop", dm, ["shop_name"])
    return read_table(engine, "dm", "shop")


# ---------------------------------------------------------------------
# Подготовка fact-слоёв
# ---------------------------------------------------------------------
def build_fact_source_frame(engine: Engine) -> pd.DataFrame:
    """
    Собирает единый staging из dds.comment_fact с присоединением справочников DDS.
    """
    fact = read_table(engine, "dds", "comment_fact")
    if fact.empty:
        return fact

    dds_source = read_table(engine, "dds", "data_source")[["source_id", "source_name", "source_type", "description"]]
    # dds_author = read_table(engine, "dds", "author")[["author_id", "author_name", "source_id"]]
    dds_shop = read_table(engine, "dds", "shop")[["shop_id", "shop_name", "shop_url", "source_id"]]
    dds_label = read_table(engine, "dds", "sentiment_label")[["label_id", "label_name", "label_code"]]

    fact = fact.merge(dds_source, on="source_id", how="left", suffixes=("", "_src"))

    # if "author_id" in fact.columns:
    #     fact = fact.merge(dds_author, on=["author_id", "source_id"], how="left", suffixes=("", "_author"))
    if "shop_id" in fact.columns:
        fact = fact.merge(dds_shop, on=["shop_id", "source_id"], how="left", suffixes=("", "_shop"))
    if "label_id" in fact.columns:
        fact = fact.merge(dds_label, on="label_id", how="left", suffixes=("", "_label"))

    return fact


def build_labeled_fact(engine: Engine, dm_sources: pd.DataFrame, dm_shops: pd.DataFrame, dm_labels: pd.DataFrame) -> pd.DataFrame:
    fact = build_fact_source_frame(engine)
    if fact.empty:
        return fact

    fact = fact[fact["label_id"].notna()].copy()

    fact["comment_text_raw"] = fact["comment_text_raw"].map(clean_text)
    # fact["comment_text_clean"] = fact["comment_text_raw"].map(clean_text)
    fact["comment_datetime"] = pd.to_datetime(fact["comment_datetime"], errors="coerce")
    fact["load_datetime"] = pd.to_datetime(fact["load_datetime"], errors="coerce")
    fact = fact[fact["comment_text_raw"].notna() & (fact["comment_text_raw"].str.len() > 0)].copy()

    # fact["source_author_key"] = fact.apply(lambda r: ns_key(r["source_name"], r["source_author_key"]), axis=1)
    # fact["shop_source_key"] = fact.apply(lambda r: ns_key(r["source_name"], r["shop_source_key"]), axis=1)

    fact = fact.merge(dm_sources[["source_id", "source_name"]], on="source_name", how="left", suffixes=("", "_dm"))
    # fact = fact.merge(
    #     dm_authors[["author_id", "source_author_key"]],
    #     on="source_author_key",
    #     how="left",
    #     suffixes=("", "_dm_author"),
    # )
    # fact = fact.merge(
    #     dm_shops[["shop_id", "shop_source_key"]],
    #     on="shop_source_key",
    #     how="left",
    #     suffixes=("", "_dm_shop"),
    # )
    fact = fact.merge(
        dm_labels[["label_id", "label_code"]],
        on="label_code",
        how="left",
        suffixes=("", "_dm_label"),
    )

    # fact["label_source"] = fact["source_name"].apply(lambda x: "manual" if str(x).lower() == "vk" else "dataset")
    fact["is_valid"] = True

    out = fact[
        [
            "comment_text_raw",
            # "comment_text_clean",
            "comment_datetime",
            # "author_id",
            "shop_id",
            "source_id",
            "label_id",
            # "label_source",
            "load_datetime",
            "is_valid",
        ]
    ].copy()

    out = filter_new_fact_rows(
        engine,
        "dm",
        "comment_fact_labeled",
        out,
        key_cols=["source_id", "shop_id", "label_id", "comment_datetime", "comment_text_raw"],
    )
    return out


def build_unlabeled_fact(engine: Engine, dm_sources: pd.DataFrame, dm_shops: pd.DataFrame) -> pd.DataFrame:
    fact = build_fact_source_frame(engine)
    if fact.empty:
        return fact

    # Только VK и только без метки
    fact = fact[(fact["source_name"].astype("string").str.lower() == "vk") & (fact["label_id"].isna())].copy()

    fact["comment_text_raw"] = fact["comment_text_raw"].map(clean_text)
    # fact["comment_text_clean"] = fact["comment_text_raw"].map(clean_text)
    fact["comment_datetime"] = pd.to_datetime(fact["comment_datetime"], errors="coerce")
    fact["load_datetime"] = pd.to_datetime(fact["load_datetime"], errors="coerce")
    fact = fact[fact["comment_text_raw"].notna() & (fact["comment_text_raw"].str.len() > 0)].copy()

    # fact["source_author_key"] = fact.apply(lambda r: ns_key(r["source_name"], r["source_author_key"]), axis=1)
    # fact["shop_source_key"] = fact.apply(lambda r: ns_key(r["source_name"], r["shop_source_key"]), axis=1)

    fact = fact.merge(dm_sources[["source_id", "source_name"]], on="source_name", how="left", suffixes=("", "_dm"))
    # fact = fact.merge(
    #     dm_authors[["author_id", "source_author_key"]],
    #     on="source_author_key",
    #     how="left",
    # )
    # fact = fact.merge(
    #     dm_shops[["shop_id", "shop_source_key"]],
    #     on="shop_source_key",
    #     how="left",
    # )

    # fact["processing_status"] = "ready_for_labeling"
    fact["is_valid"] = True

    out = fact[
        [
            "comment_text_raw",
            # "comment_text_clean",
            "comment_datetime",
            # "author_id",
            "shop_id",
            "source_id",
            "load_datetime",
            # "processing_status",
            "is_valid",
        ]
    ].copy()

    out = filter_new_fact_rows(
        engine,
        "dm",
        "comment_fact_unlabeled",
        out,
        key_cols=["source_id", "shop_id", "comment_datetime", "comment_text_raw"],
    )
    return out


# ---------------------------------------------------------------------
# Основной загрузчик
# ---------------------------------------------------------------------
def load_dm(engine: Engine) -> None:
    dm_sources = load_data_source_dim(engine)
    print("dm.data_source загружен")

    dm_labels = load_sentiment_label_dim(engine)
    print("dm.sentiment_label загружен")

    # dm_authors = load_author_dim(engine)
    # print("dm.author загружен")

    dm_shops = load_shop_dim(engine)
    print("dm.shop загружен")

    labeled_fact = build_labeled_fact(engine, dm_sources, dm_shops, dm_labels)
    if not labeled_fact.empty:
        labeled_fact.to_sql(
            name="comment_fact_labeled",
            con=engine,
            schema="dm",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"comment_fact_labeled загружен: {len(labeled_fact)} строк")
    else:
        print("comment_fact_labeled: нет новых строк")

    unlabeled_fact = build_unlabeled_fact(engine, dm_sources, dm_shops)
    if not unlabeled_fact.empty:
        unlabeled_fact.to_sql(
            name="comment_fact_unlabeled",
            con=engine,
            schema="dm",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"comment_fact_unlabeled загружен: {len(unlabeled_fact)} строк")
    else:
        print("comment_fact_unlabeled: нет новых строк")


def main() -> None:
    engine = get_engine()
    ensure_schema(engine, "dm")
    ensure_tables(engine, "dm")
    load_dm(engine)


if __name__ == "__main__":
    main()