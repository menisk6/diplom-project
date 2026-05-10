from __future__ import annotations

import os
from urllib.parse import quote_plus

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------
# Подключение к БД
# ---------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "diplom")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

safe_user = quote_plus(DB_USER)
safe_password = quote_plus(DB_PASSWORD)
ENGINE_URL = f"postgresql+psycopg2://{safe_user}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine() -> Engine:
    return create_engine(
        ENGINE_URL,
        future=True,
        connect_args={"client_encoding": "utf8"},
    )


def ensure_schema(engine: Engine, schema_name: str = "dm") -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))


def build_dm_tables(metadata: MetaData, schema_name: str = "dm") -> dict[str, Table]:
    # -----------------------------------------------------------------
    # 1. author
    # -----------------------------------------------------------------
    # author = Table(
    #     "author",
    #     metadata,
    #     Column("author_id", BigInteger, primary_key=True, autoincrement=True),
    #     Column("author_name", Text, nullable=True),
    #     # Column("source_author_key", String(255), nullable=True),
    #     Column("created_at", DateTime, nullable=False),
    #     UniqueConstraint("author_name", name="uq_dm_author_source_author_key"),
    #     schema=schema_name,
    #     extend_existing=True,
    # )

    # -----------------------------------------------------------------
    # 2. shop
    # -----------------------------------------------------------------
    shop = Table(
        "shop",
        metadata,
        Column("shop_id", BigInteger, primary_key=True, autoincrement=True),
        Column("shop_name", Text, nullable=True),
        # Column("shop_source_key", String(255), nullable=True),
        Column("shop_url", Text, nullable=True),
        Column("created_at", DateTime, nullable=False),
        UniqueConstraint("shop_name", name="uq_dm_shop_source_key"),
        schema=schema_name,
        extend_existing=True,
    )

    # -----------------------------------------------------------------
    # 3. data_source
    # -----------------------------------------------------------------
    data_source = Table(
        "data_source",
        metadata,
        Column("source_id", BigInteger, primary_key=True, autoincrement=True),
        Column("source_name", String(128), nullable=False, unique=True),
        Column("source_type", String(64), nullable=False),
        Column("description", Text, nullable=True),
        Column("created_at", DateTime, nullable=False),
        schema=schema_name,
        extend_existing=True,
    )

    # -----------------------------------------------------------------
    # 4. sentiment_label
    # -----------------------------------------------------------------
    sentiment_label = Table(
        "sentiment_label",
        metadata,
        Column("label_id", BigInteger, primary_key=True, autoincrement=True),
        Column("label_name", String(64), nullable=False),
        Column("label_code", String(32), nullable=False, unique=True),
        Column("label_description", Text, nullable=True),
        Column("created_at", DateTime, nullable=False),
        schema=schema_name,
        extend_existing=True,
    )

    # -----------------------------------------------------------------
    # 5. comment_fact_labeled
    # -----------------------------------------------------------------
    comment_fact_labeled = Table(
        "comment_fact_labeled",
        metadata,
        Column("comment_id", BigInteger, primary_key=True, autoincrement=True),
        Column("comment_text_raw", Text, nullable=False),
        # Column("comment_text_clean", Text, nullable=False),
        Column("comment_datetime", DateTime, nullable=True),
        # Column("author_id", BigInteger, ForeignKey(f"{schema_name}.author.author_id"), nullable=True),
        Column("shop_id", BigInteger, ForeignKey(f"{schema_name}.shop.shop_id"), nullable=True),
        Column("source_id", BigInteger, ForeignKey(f"{schema_name}.data_source.source_id"), nullable=False),
        Column("label_id", BigInteger, ForeignKey(f"{schema_name}.sentiment_label.label_id"), nullable=False),
        # Column("label_source", String(64), nullable=False),
        Column("load_datetime", DateTime, nullable=False),
        Column("is_valid", Boolean, nullable=False),
        schema=schema_name,
        extend_existing=True,
    )

    # -----------------------------------------------------------------
    # 6. comment_fact_unlabeled
    # -----------------------------------------------------------------
    comment_fact_unlabeled = Table(
        "comment_fact_unlabeled",
        metadata,
        Column("comment_id", BigInteger, primary_key=True, autoincrement=True),
        Column("comment_text_raw", Text, nullable=False),
        # Column("comment_text_clean", Text, nullable=False),
        Column("comment_datetime", DateTime, nullable=True),
        # Column("author_id", BigInteger, ForeignKey(f"{schema_name}.author.author_id"), nullable=True),
        Column("shop_id", BigInteger, ForeignKey(f"{schema_name}.shop.shop_id"), nullable=True),
        Column("source_id", BigInteger, ForeignKey(f"{schema_name}.data_source.source_id"), nullable=False),
        Column("load_datetime", DateTime, nullable=False),
        # Column("processing_status", String(64), nullable=False),
        Column("is_valid", Boolean, nullable=False),
        schema=schema_name,
        extend_existing=True,
    )

    return {
        # "author": author,
        "shop": shop,
        "data_source": data_source,
        "sentiment_label": sentiment_label,
        "comment_fact_labeled": comment_fact_labeled,
        "comment_fact_unlabeled": comment_fact_unlabeled,
    }


def ensure_tables(engine: Engine, schema_name: str = "dm") -> None:
    metadata = MetaData()
    build_dm_tables(metadata, schema_name=schema_name)
    metadata.create_all(engine, checkfirst=True)


def main() -> None:
    engine = get_engine()
    ensure_schema(engine, "dm")
    ensure_tables(engine, "dm")
    print("[OK] DM schema and tables created successfully.")


if __name__ == "__main__":
    main()