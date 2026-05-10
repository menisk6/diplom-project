from __future__ import annotations

import os
from urllib.parse import quote_plus

from sqlalchemy import (
    Boolean,
    BigInteger,
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
    return create_engine(ENGINE_URL, future=True, connect_args={"client_encoding": "utf8"})


def ensure_schema(engine: Engine, schema_name: str = "dds") -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))


def build_dds_tables(metadata: MetaData, schema_name: str = "dds") -> dict[str, Table]:
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

    # author = Table(
    #     "author",
    #     metadata,
    #     Column("author_id", BigInteger, primary_key=True, autoincrement=True),
    #     Column("author_name", Text, nullable=True),
    #     Column("source_author_key", String(255), nullable=False),
    #     Column("source_id", BigInteger, ForeignKey(f"{schema_name}.data_source.source_id"), nullable=False),
    #     Column("created_at", DateTime, nullable=False),
    #     UniqueConstraint("source_id", "source_author_key", name="uq_dds_author_source_key"),
    #     schema=schema_name,
    #     extend_existing=True,
    # )

    shop = Table(
        "shop",
        metadata,
        Column("shop_id", BigInteger, primary_key=True, autoincrement=True),
        Column("shop_name", Text, nullable=True),
        Column("shop_source_key", String(255), nullable=False),
        Column("shop_url", Text, nullable=True),
        Column("source_id", BigInteger, ForeignKey(f"{schema_name}.data_source.source_id"), nullable=False),
        Column("created_at", DateTime, nullable=False),
        UniqueConstraint("source_id", "shop_source_key", name="uq_dds_shop_source_key"),
        schema=schema_name,
        extend_existing=True,
    )

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

    comment_fact = Table(
        "comment_fact",
        metadata,
        Column("comment_id", BigInteger, primary_key=True, autoincrement=True),
        Column("comment_uid", String(128), nullable=False, unique=True),
        Column("source_id", BigInteger, ForeignKey(f"{schema_name}.data_source.source_id"), nullable=False),
        Column("source_record_key", String(255), nullable=True),
        # Column("author_id", BigInteger, ForeignKey(f"{schema_name}.author.author_id"), nullable=True),
        Column("shop_id", BigInteger, ForeignKey(f"{schema_name}.shop.shop_id"), nullable=True),
        Column("label_id", BigInteger, ForeignKey(f"{schema_name}.sentiment_label.label_id"), nullable=True),
        Column("category_text_raw", String(255), nullable=True),
        Column("comment_text_raw", Text, nullable=False),
        Column("comment_datetime", DateTime, nullable=True),
        Column("load_datetime", DateTime, nullable=False),
        Column("is_valid", Boolean, nullable=False),
        UniqueConstraint("comment_uid", name="uq_dds_comment_uid"),
        schema=schema_name,
        extend_existing=True,
    )

    return {
        "data_source": data_source,
        # "author": author,
        "shop": shop,
        "sentiment_label": sentiment_label,
        "comment_fact": comment_fact,
    }


def ensure_tables(engine: Engine, schema_name: str = "dds") -> None:
    metadata = MetaData()
    build_dds_tables(metadata, schema_name=schema_name)
    metadata.create_all(engine, checkfirst=True)
    print("[OK] DDS schema and tables created successfully.")