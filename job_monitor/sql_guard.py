"""SQL guardrails for the agent's read-only query tool.

Uses sqlglot to parse the query rather than regex matching, so we can reason
about set operations, CTEs, subqueries, and qualified identifiers properly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.AlterTable,
    exp.TruncateTable,
    exp.Merge,
    exp.Command,
)

_ALLOWED_TOP_LEVEL: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.Subquery,
)


class UnsafeSQLError(ValueError):
    """Raised when the SQL guard rejects a query."""


def _table_aliases(table: exp.Table) -> set[str]:
    """Return all the ways `table` could be addressed (bare, schema.table, catalog.schema.table)."""
    name = table.name.lower()
    aliases = {name}
    if table.db:
        aliases.add(f"{table.db}.{name}".lower())
    if table.catalog:
        aliases.add(f"{table.catalog}.{table.db}.{name}".lower())
    return aliases


def validate_select(sql: str, allowed_tables: Iterable[str]) -> None:
    """Raise UnsafeSQLError if `sql` is not a single safe read-only query.

    A query is accepted when:
      - It parses as exactly one statement.
      - The top-level is a SELECT, set operation, or subquery.
      - It contains no DDL/DML nodes anywhere in the tree.
      - Every referenced table matches at least one entry in `allowed_tables`
        (case-insensitive; bare/schema-qualified/fully-qualified all accepted).
    """
    allowed = {t.lower() for t in allowed_tables}

    try:
        statements = sqlglot.parse(sql, dialect="databricks")
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"Could not parse SQL: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Exactly one statement allowed; got {len(statements)}."
        )

    tree = statements[0]
    if not isinstance(tree, _ALLOWED_TOP_LEVEL):
        raise UnsafeSQLError(
            f"Only read-only SELECT-style queries allowed; got {type(tree).__name__}."
        )

    for forbidden_type in _FORBIDDEN_NODES:
        hit = next(iter(tree.find_all(forbidden_type)), None)
        if hit is not None:
            raise UnsafeSQLError(f"Forbidden statement type in query: {type(hit).__name__}")

    for table in tree.find_all(exp.Table):
        if not _table_aliases(table) & allowed:
            ref = table.sql(dialect="databricks")
            raise UnsafeSQLError(
                f"Table '{ref}' is not allowed. Allowed tables: {sorted(allowed)}"
            )
