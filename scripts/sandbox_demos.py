"""Four ways to give each run its own state, from cheapest to heaviest (chapter 7).

Usage:
    uv run python scripts/sandbox_demos.py
    uv run --extra containers python scripts/sandbox_demos.py --container

The container demo needs Docker running and the `containers` extra.
"""

import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def temp_directory_per_run() -> None:
    with tempfile.TemporaryDirectory() as a:
        (Path(a) / "refund_receipt.txt").write_text("84.50")
        print("run A wrote:", sorted(p.name for p in Path(a).iterdir()))
    with tempfile.TemporaryDirectory() as b:
        print("run B sees: ", sorted(p.name for p in Path(b).iterdir()))
    print("run A's directory still exists:", Path(a).exists())


def issue_receipt(amount: float, now: datetime) -> str:
    """The tool is handed the clock, so a test can hand it a fixed one."""
    return f"refund {amount} at {now.isoformat()}"


def controlled_clock() -> None:
    fixed = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    first = issue_receipt(84.5, fixed)
    second = issue_receipt(84.5, fixed)
    print(first)
    print("same output twice:", first == second)


def new_run_database() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("create table refunds (customer text, amount real)")
    return db


def refunds_in(db: sqlite3.Connection) -> int:
    return db.execute("select count(*) from refunds").fetchone()[0]


def database_per_run() -> None:
    a = new_run_database()
    a.execute("insert into refunds values ('cust-42', 84.5)")
    print("run A sees", refunds_in(a), "refund")
    b = new_run_database()
    print("run B sees", refunds_in(b), "refunds")


def throwaway_postgres() -> None:
    import psycopg
    from testcontainers.community.postgres import PostgresContainer

    def refunds_after(insert: bool) -> int:
        with (
            PostgresContainer("postgres:18") as pg,
            psycopg.connect(
                host=pg.get_container_host_ip(),
                port=pg.get_exposed_port(5432),
                user=pg.username,
                password=pg.password,
                dbname=pg.dbname,
            ) as conn,
        ):
            conn.execute("create table refunds (customer text, amount numeric)")
            if insert:
                conn.execute("insert into refunds values ('cust-42', 84.5)")
            return conn.execute("select count(*) from refunds").fetchone()[0]

    print("run A sees", refunds_after(insert=True), "refund")
    print("run B sees", refunds_after(insert=False), "refunds")


def main() -> None:
    print("-- a temporary directory per run")
    temp_directory_per_run()
    print("-- a controlled clock")
    controlled_clock()
    print("-- an in-memory database per run")
    database_per_run()
    if "--container" in sys.argv:
        print("-- a throwaway Postgres container per run")
        throwaway_postgres()


if __name__ == "__main__":
    main()
