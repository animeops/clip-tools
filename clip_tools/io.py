import os
import sqlite3
import tempfile
from typing import Dict, Tuple

import pandas as pd


SQLITE_HEADER = b"SQLite format 3"


def split_clip_binary(clip_str: bytes) -> Tuple[bytes, bytes]:
    """Split a raw .clip file into (binary_chunks, sqlite_db) byte strings."""
    find_index = clip_str.find(SQLITE_HEADER)
    if find_index <= 0:
        raise Exception("Invalid CLIP Studio Paint file")
    return clip_str[:find_index], clip_str[find_index:]


def load_sqlite(sqlite_binary_str: bytes) -> Dict[str, pd.DataFrame]:
    """Load all tables from an in-memory SQLite blob into a dict of DataFrames."""
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    sqlite_path = f.name
    try:
        f.write(sqlite_binary_str)
        f.close()

        with sqlite3.connect(sqlite_path) as connect:
            cursor = connect.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            dfs: Dict[str, pd.DataFrame] = {}
            for (table_name,) in tables:
                dfs[table_name] = pd.read_sql_query(
                    f"SELECT * from {table_name}", connect
                )
        return dfs
    finally:
        try:
            os.unlink(sqlite_path)
        except OSError:
            pass
