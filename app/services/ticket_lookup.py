"""Read the company coding catalog without allocating numbers."""
from __future__ import annotations

import argparse
import getpass
import json
import sys

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.pool import NullPool


TEAM_CODES = text('''
    SELECT child_theme_name AS subtheme_name, child_theme_prefix AS abbreviation,
           theme_name, theme_prefix, theme_type, theme_type_prefix,
           COUNT(DISTINCT product_id) AS product_count,
           GROUP_CONCAT(DISTINCT CASE WHEN product_name LIKE '%帽衫%' OR product_name LIKE '%连帽%'
                                      THEN product_name END ORDER BY product_name) AS hoodie_products
    FROM theme
    WHERE LOWER(TRIM(child_theme_name)) IN :names OR LOWER(TRIM(theme_name)) IN :names
    GROUP BY child_theme_name, child_theme_prefix, theme_name, theme_prefix, theme_type, theme_type_prefix
    ORDER BY theme_name, child_theme_name, child_theme_prefix
    LIMIT 100
''').bindparams(bindparam('names', expanding=True))

LEAGUE_TEAM_CODES = text('''
    SELECT DISTINCT child_theme_name AS subtheme_name, child_theme_prefix AS abbreviation
    FROM theme
    WHERE LOWER(TRIM(child_theme_name)) IN :names
      AND LOWER(TRIM(theme_type)) = :theme_type
      AND LOWER(TRIM(theme_name)) IN :league_names
''').bindparams(bindparam('names', expanding=True), bindparam('league_names', expanding=True))

CATALOG_TABLES = text('''
    SELECT TABLE_SCHEMA AS database_name, TABLE_NAME AS table_name,
           GROUP_CONCAT(COLUMN_NAME ORDER BY ORDINAL_POSITION) AS column_names
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME IN ('theme', 'match', 'product', 'child_theme', 'theme_type', 'product_theme_type')
    GROUP BY TABLE_SCHEMA, TABLE_NAME
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    LIMIT 100
''')

PRODUCT_CODES = text('''
    SELECT id, product_name, prefix, current_number
    FROM product
    WHERE LOWER(TRIM(product_name)) = :product_name
''')

PRODUCT_THEME_CODES = text('''
    SELECT DISTINCT theme_name, medium_code, child_theme_name, child_theme_prefix
    FROM theme
    WHERE product_id = :product_id AND LOWER(TRIM(theme_type)) = :theme_type
    ORDER BY theme_name, child_theme_name
''')

PRODUCT_TICKET_HISTORY = text('''
    SELECT MAX(end_number) AS last_issued_number, MAX(created_at) AS last_issued_at
    FROM ticket
    WHERE product_id = :product_id
''')


def catalog_engine(url: URL):
    return create_engine(
        url, poolclass=NullPool, hide_parameters=True,
        connect_args={'connect_timeout': 5, 'read_timeout': 15, 'write_timeout': 5},
    )


def read_catalog(url: URL, statement, parameters=None) -> list[dict]:
    engine = catalog_engine(url)
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('START TRANSACTION READ ONLY')
            rows = connection.execute(statement, parameters or {}).mappings().all()
            return [dict(row) for row in rows]
    finally:
        engine.dispose()


def lookup_product_catalog(url: URL, *, product_name: str, theme_type: str):
    if url.drivername != 'mysql+pymysql' or url.database != 'ticket_system':
        raise ValueError('公司编码目录应配置为 MySQL ticket_system。')
    engine = catalog_engine(url)
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql('START TRANSACTION READ ONLY')
            products = connection.execute(PRODUCT_CODES, {'product_name': product_name}).mappings().all()
            if len(products) != 1:
                return None, [], None
            product = dict(products[0])
            product_id = product['id']
            rows = connection.execute(PRODUCT_THEME_CODES, {
                'product_id': product_id, 'theme_type': theme_type,
            }).mappings().all()
            history = connection.execute(PRODUCT_TICKET_HISTORY, {'product_id': product_id}).mappings().one()
            return product, [dict(row) for row in rows], dict(history)
    finally:
        engine.dispose()


def lookup_team_codes(url: URL, names: list[str]) -> list[dict]:
    terms = sorted({value.strip().lower() for value in names if value.strip()})
    if not terms:
        raise ValueError('请提供球队全名。')
    return read_catalog(url, TEAM_CODES, {'names': terms})


def lookup_league_team_codes(url: URL, names, *, theme_type: str, league: str) -> list[dict]:
    league = league.strip().lower().removesuffix('球队').strip()
    return read_catalog(url, LEAGUE_TEAM_CODES, {
        'names': sorted(names), 'theme_type': theme_type.strip().lower(), 'league_names': [league, league + '球队'],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description='只读查询领号库球队缩写')
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=3306)
    parser.add_argument('--database', required=True)
    parser.add_argument('--user', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--name', action='append', help='球队全名，可重复传入')
    action.add_argument('--catalog-info', action='store_true', help='列出领号库的表和字段')
    args = parser.parse_args()
    password = getpass.getpass('MySQL 密码：') if sys.stdin.isatty() else sys.stdin.readline().rstrip('\r\n')
    url = URL.create('mysql+pymysql', username=args.user, password=password,
                     host=args.host, port=args.port, database=args.database, query={'charset': 'utf8mb4'})
    try:
        rows = read_catalog(url, CATALOG_TABLES) if args.catalog_info else lookup_team_codes(url, args.name)
    except DBAPIError as exc:
        number = exc.orig.args[0] if exc.orig.args and isinstance(exc.orig.args[0], int) else None
        print(json.dumps({'ok': False, 'mysql_error': number}, ensure_ascii=False))
        return 1
    print(json.dumps({'ok': True, 'database': args.database, 'rows': rows}, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
