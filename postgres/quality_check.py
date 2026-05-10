from __future__ import annotations

import os
import uuid
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.sql.sqltypes import String, Text


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

TARGET_SCHEMAS = ["dds", "dm"]
DQ_SCHEMA = "dq"
DQ_TABLE = "quality_check"


def get_engine() -> Engine:
    return create_engine(
        ENGINE_URL,
        future=True,
        connect_args={"client_encoding": "utf8"},
    )


# ---------------------------------------------------------------------
# DQ-таблица
# ---------------------------------------------------------------------
def ensure_dq_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{DQ_SCHEMA}"'))


def ensure_dq_table(engine: Engine) -> None:
    """
    Таблица для записи результатов DQ-проверок.
    Используем to_sql для загрузки логов, но таблицу создаём заранее.
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS "{DQ_SCHEMA}"."{DQ_TABLE}" (
        check_id BIGSERIAL PRIMARY KEY,
        check_run_id VARCHAR(64) NOT NULL,
        checked_at TIMESTAMP NOT NULL,
        schema_name VARCHAR(64) NOT NULL,
        table_name VARCHAR(128) NOT NULL,
        error_type VARCHAR(64) NOT NULL,
        error_detail TEXT NULL,
        affected_rows BIGINT NULL
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ---------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------
def read_table(engine: Engine, schema: str, table: str) -> pd.DataFrame:
    return pd.read_sql_query(text(f'SELECT * FROM "{schema}"."{table}"'), con=engine)


def table_exists(inspector, schema: str, table: str) -> bool:
    return inspector.has_table(table_name=table, schema=schema)


def is_textual_type(col_type) -> bool:
    return isinstance(col_type, (String, Text))


def log_issue(
    issues: list[dict],
    check_run_id: str,
    schema_name: str,
    table_name: str,
    error_type: str,
    error_detail: str,
    affected_rows: int | None = None,
) -> None:
    issues.append(
        {
            "check_run_id": check_run_id,
            "checked_at": datetime.now(),
            "schema_name": schema_name,
            "table_name": table_name,
            "error_type": error_type,
            "error_detail": error_detail,
            "affected_rows": affected_rows,
        }
    )


# ---------------------------------------------------------------------
# Проверки
# ---------------------------------------------------------------------
def check_duplicate_pk(engine: Engine, schema: str, table: str, pk_cols: list[str]) -> tuple[int, str]:
    """
    Возвращает:
      - число строк, участвующих в дублях по PK
      - текстовое описание нарушения
    """
    if not pk_cols:
        return 0, "PK отсутствует, проверка дублей невозможна"

    cols_sql = ", ".join(f'"{c}"' for c in pk_cols)

    sql = f"""
        SELECT COALESCE(SUM(cnt - 1), 0) AS duplicate_rows
        FROM (
            SELECT {cols_sql}, COUNT(*) AS cnt
            FROM "{schema}"."{table}"
            GROUP BY {cols_sql}
            HAVING COUNT(*) > 1
        ) t
    """
    dup_rows = int(pd.read_sql_query(text(sql), con=engine).iloc[0]["duplicate_rows"])
    detail = f"PK columns: {', '.join(pk_cols)}"
    return dup_rows, detail


def check_required_fields(engine: Engine, schema: str, table: str, columns_meta: list[dict]) -> list[tuple[str, int]]:
    """
    Проверяет обязательные поля:
      - NOT NULL
      - для текстовых полей дополнительно пустую строку

    Возвращает список нарушений: [(detail, affected_rows), ...]
    """
    violations: list[tuple[str, int]] = []

    required_cols = [c for c in columns_meta if c.get("nullable") is False]

    for col in required_cols:
        col_name = col["name"]
        col_type = col["type"]

        if is_textual_type(col_type):
            sql = f"""
                SELECT COUNT(*) AS bad_rows
                FROM "{schema}"."{table}"
                WHERE "{col_name}" IS NULL
                   OR BTRIM(CAST("{col_name}" AS TEXT)) = ''
            """
        else:
            sql = f"""
                SELECT COUNT(*) AS bad_rows
                FROM "{schema}"."{table}"
                WHERE "{col_name}" IS NULL
            """

        bad_rows = int(pd.read_sql_query(text(sql), con=engine).iloc[0]["bad_rows"])
        if bad_rows > 0:
            violations.append((f"required field: {col_name}", bad_rows))

    return violations


def check_fk_integrity(engine: Engine, schema: str, table: str, fk_list: list[dict]) -> list[tuple[str, int]]:
    """
    Проверяет внешние ключи таблицы.
    Возвращает список нарушений: [(detail, affected_rows), ...]
    """
    violations: list[tuple[str, int]] = []

    for fk in fk_list:
        constrained_cols = fk.get("constrained_columns") or []
        referred_schema = fk.get("referred_schema") or schema
        referred_table = fk.get("referred_table")
        referred_cols = fk.get("referred_columns") or []

        if not constrained_cols or not referred_table or not referred_cols:
            continue

        join_cond = " AND ".join(
            f't."{lc}" = r."{rc}"'
            for lc, rc in zip(constrained_cols, referred_cols)
        )

        non_null_cond = " AND ".join(
            f't."{lc}" IS NOT NULL'
            for lc in constrained_cols
        )

        # Если FK не заполнен полностью, он не считается нарушением целостности
        sql = f"""
            SELECT COUNT(*) AS bad_rows
            FROM "{schema}"."{table}" t
            LEFT JOIN "{referred_schema}"."{referred_table}" r
                ON {join_cond}
            WHERE {non_null_cond}
              AND r."{referred_cols[0]}" IS NULL
        """
        bad_rows = int(pd.read_sql_query(text(sql), con=engine).iloc[0]["bad_rows"])
        if bad_rows > 0:
            detail = (
                f'FK {fk.get("name") or "(unnamed)"} -> '
                f'"{referred_schema}"."{referred_table}" '
                f'({", ".join(constrained_cols)} -> {", ".join(referred_cols)})'
            )
            violations.append((detail, bad_rows))

    return violations


def validate_table(engine: Engine, inspector, schema: str, table: str, check_run_id: str) -> list[dict]:
    issues: list[dict] = []

    cols_meta = inspector.get_columns(table_name=table, schema=schema)
    pk_info = inspector.get_pk_constraint(table_name=table, schema=schema)
    fk_list = inspector.get_foreign_keys(table_name=table, schema=schema)

    if not table_exists(inspector, schema, table):
        log_issue(
            issues,
            check_run_id,
            schema,
            table,
            "table_missing",
            f'Table "{schema}"."{table}" not found',
            None,
        )
        return issues

    # Если таблица пустая, считаем её корректной по DQ-логике:
    # пустая таблица не нарушает PK/FK/NOT NULL.
    row_count_sql = f'SELECT COUNT(*) AS n FROM "{schema}"."{table}"'
    row_count = int(pd.read_sql_query(text(row_count_sql), con=engine).iloc[0]["n"])
    if row_count == 0:
        return issues

    # 1) Дубли по PK
    pk_cols = pk_info.get("constrained_columns") or []
    dup_rows, pk_detail = check_duplicate_pk(engine, schema, table, pk_cols)
    if dup_rows > 0:
        log_issue(
            issues,
            check_run_id,
            schema,
            table,
            "duplicate_pk",
            pk_detail,
            dup_rows,
        )

    # 2) Обязательные поля
    required_violations = check_required_fields(engine, schema, table, cols_meta)
    for detail, bad_rows in required_violations:
        log_issue(
            issues,
            check_run_id,
            schema,
            table,
            "missing_required_fields",
            detail,
            bad_rows,
        )

    # 3) Ссылочная целостность
    fk_violations = check_fk_integrity(engine, schema, table, fk_list)
    for detail, bad_rows in fk_violations:
        log_issue(
            issues,
            check_run_id,
            schema,
            table,
            "foreign_key_violation",
            detail,
            bad_rows,
        )

    return issues


def collect_quality_issues(engine: Engine) -> pd.DataFrame:
    inspector = inspect(engine)
    check_run_id = uuid.uuid4().hex
    all_issues: list[dict] = []

    for schema in TARGET_SCHEMAS:
        if not inspector.has_schema(schema):
            continue

        tables = inspector.get_table_names(schema=schema)
        for table in tables:
            # пропускаем, если в будущем появится DQ в списке схем
            if schema == DQ_SCHEMA.lower() and table == DQ_TABLE:
                continue
            table_issues = validate_table(engine, inspector, schema, table, check_run_id)
            all_issues.extend(table_issues)

    if not all_issues:
        return pd.DataFrame(
            columns=[
                "check_run_id",
                "checked_at",
                "schema_name",
                "table_name",
                "error_type",
                "error_detail",
                "affected_rows",
            ]
        )

    return pd.DataFrame(all_issues)


def save_quality_issues(engine: Engine, issues_df: pd.DataFrame) -> None:
    if issues_df.empty:
        print("[OK] Нарушений качества данных не найдено.")
        return

    issues_df.to_sql(
        name=DQ_TABLE,
        con=engine,
        schema=DQ_SCHEMA,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print(f"[OK] Найдено и записано нарушений DQ: {len(issues_df)}")


def main() -> None:
    engine = get_engine()
    ensure_dq_schema(engine)
    ensure_dq_table(engine)

    issues_df = collect_quality_issues(engine)
    save_quality_issues(engine, issues_df)


if __name__ == "__main__":
    main()