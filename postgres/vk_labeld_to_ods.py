from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import sha1
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

CSV_PATH = Path("../datasets/vk_labeld_data.csv")
ODS_SCHEMA = "ods"
ODS_TABLE = "vk_labeled_data_raw"


def get_engine() -> Engine:
    return create_engine(ENGINE_URL, future=True, connect_args={"client_encoding": "utf8"})


def ensure_schema(engine: Engine, schema_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))


def table_exists(engine: Engine, schema: str, table: str) -> bool:
    return inspect(engine).has_table(table_name=table, schema=schema)


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


def build_ods_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="Windows-1251", sep=';')

    required = {
        # "author_id",
        "comment_datetime",
        "comment_id",
        "comment_text_raw",
        "created_at",
        "id",
        "is_valid",
        "load_datetime",
        "sentiment",
        "shop_id",
        "source_id",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"В CSV не хватает колонок: {missing}")

    df = df.copy()
    df["source_record_key"] = df["comment_id"].astype("string")
    df["source_file"] = csv_path.name
    df["raw_payload"] = df.apply(
        lambda r: json.dumps(
            r.where(pd.notnull(r), None).to_dict(),
            ensure_ascii=False,
            default=str,
        ),
        axis=1,
    )
    df["ingest_datetime"] = datetime.now()

    df = df.dropna(subset=["source_record_key", "comment_text_raw"]).drop_duplicates(
        subset=["source_record_key"],
        keep="last",
    )

    # порядок колонок можно оставить таким
    columns = [
        "source_record_key",
        # "author_id",
        "comment_datetime",
        "comment_id",
        "comment_text_raw",
        "created_at",
        "id",
        "is_valid",
        "load_datetime",
        "sentiment",
        "shop_id",
        "source_id",
        "source_file",
        "raw_payload",
        "ingest_datetime",
    ]
    return df[columns].copy()


def load_to_ods(engine: Engine, df: pd.DataFrame) -> None:
    if df.empty:
        print("[INFO] Нет строк для загрузки в ODS.")
        return

    if table_exists(engine, ODS_SCHEMA, ODS_TABLE):
        delete_rows_by_keys(
            engine=engine,
            schema=ODS_SCHEMA,
            table=ODS_TABLE,
            key_col="source_record_key",
            keys=df["source_record_key"].tolist(),
        )

    df.to_sql(
        name=ODS_TABLE,
        con=engine,
        schema=ODS_SCHEMA,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print(f"[OK] Загружено в {ODS_SCHEMA}.{ODS_TABLE}: {len(df)} строк")


def main() -> None:
    engine = get_engine()
    ensure_schema(engine, ODS_SCHEMA)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {CSV_PATH}")

    ods_df = build_ods_dataframe(CSV_PATH)
    load_to_ods(engine, ods_df)


if __name__ == "__main__":
    main()